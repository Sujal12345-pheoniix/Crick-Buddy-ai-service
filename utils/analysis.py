"""
Cricket Analysis Utilities — Production-Grade Pipeline
=======================================================
Detection flow per frame:
  1. Blur filter  → skip low-quality frames
  2. Resize       → 640×480 for stable MediaPipe input
  3. YOLOv8       → detect person (class 0); crop + pad bounding box
  4. MediaPipe    → pose on cropped region (lower false-negative rate)
  5. Landmark re-projection back to full-frame coordinates

Multi-frame majority-vote: at least 1 valid detection in the sampled
sequence is sufficient to proceed.  Only a completely empty sequence
triggers "No human detected".
"""

import json
import math
import os
import re
from typing import Optional, List, Tuple

import numpy as np

# ─── MediaPipe Landmark IDs (BlazePose 33-point model) ───────────────────────
NOSE = 0
LEFT_SHOULDER = 11;  RIGHT_SHOULDER = 12
LEFT_ELBOW = 13;     RIGHT_ELBOW = 14
LEFT_WRIST = 15;     RIGHT_WRIST = 16
LEFT_HIP = 23;       RIGHT_HIP = 24
LEFT_KNEE = 25;      RIGHT_KNEE = 26
LEFT_ANKLE = 27;     RIGHT_ANKLE = 28


# ─── Lazy singletons (loaded once, reused across requests) ───────────────────
_yolo_model = None
_pose_model_1 = None   # complexity=1 reusable instance
_pose_model_2 = None   # complexity=2 fallback


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            weight = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt")
            _yolo_model = YOLO(weight)
            print("✅ YOLOv8 model loaded")
        except Exception as e:
            print(f"⚠️  YOLOv8 load failed (will skip crop): {e}")
            _yolo_model = False   # sentinel: do not retry
    return _yolo_model if _yolo_model else None


# ─── Basic helpers ────────────────────────────────────────────────────────────

def clamp_score(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(float(angle), 1)


def score_angle(angle: float, ideal_min: float, ideal_max: float) -> int:
    mid = (ideal_min + ideal_max) / 2
    half_range = max((ideal_max - ideal_min) / 2, 1)
    deviation = abs(angle - mid)
    if deviation <= half_range:
        return clamp_score(98 - (deviation / half_range) * 12, 0, 100)
    return clamp_score(86 - (deviation - half_range) * 1.8, 0, 100)


def get_point(landmarks: dict, idx: int) -> Optional[List[float]]:
    lm = landmarks.get(idx)
    if lm is None:
        return None
    if isinstance(lm, dict):
        return [lm["x"], lm["y"]]
    return [lm[0], lm[1]]


# ─── Frame utilities ──────────────────────────────────────────────────────────

TARGET_W, TARGET_H = 640, 480
BLUR_THRESHOLD = 60.0    # Laplacian variance; frames below this are skipped


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize to TARGET_W×TARGET_H while keeping dtype uint8."""
    import cv2
    h, w = frame.shape[:2]
    if w == TARGET_W and h == TARGET_H:
        return frame
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)


def _is_sharp(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """Return True if the frame is sharp enough to be useful."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= threshold


def extract_frames(video_path: str, num_frames: int = 30) -> List[np.ndarray]:
    """Extract evenly-spaced frames from a video, filtered for sharpness."""
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sequential read when codec doesn't report frame count
        if total <= 0:
            raw: List[np.ndarray] = []
            while True:
                ret, frm = cap.read()
                if not ret or frm is None:
                    break
                raw.append(frm)
            cap.release()
        else:
            indices = np.linspace(0, max(total - 1, 0),
                                  num=min(num_frames * 3, total), dtype=int)
            raw = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frm = cap.read()
                if ret and frm is not None:
                    raw.append(frm)
            cap.release()

            # Fallback: sequential read if seek failed
            if not raw:
                cap2 = cv2.VideoCapture(video_path)
                while len(raw) < num_frames * 4:
                    ret, frm = cap2.read()
                    if not ret or frm is None:
                        break
                    raw.append(frm)
                cap2.release()

        if not raw:
            return []

        # Prefer sharp frames; fall back to all frames if too few sharp ones
        sharp = [f for f in raw if _is_sharp(f)]
        pool = sharp if len(sharp) >= max(num_frames // 3, 3) else raw

        n = min(num_frames, len(pool))
        indices2 = np.linspace(0, len(pool) - 1, num=n, dtype=int)
        return [pool[int(i)] for i in indices2]

    except Exception as e:
        print(f"Frame extraction error: {e}")
        return []


# ─── YOLO player crop ─────────────────────────────────────────────────────────

def _yolo_crop_person(frame: np.ndarray, pad: float = 0.15) -> Optional[Tuple[np.ndarray, Tuple]]:
    """
    Run YOLOv8 on `frame`, find the highest-confidence person (class 0) box,
    pad it, and return (cropped_frame, (x1, y1, x2, y2)) in original coords.
    Returns None if YOLO is unavailable or no person found.
    """
    model = _get_yolo()
    if model is None:
        return None
    try:
        results = model(frame, verbose=False, conf=0.25, classes=[0])
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # Pick highest-confidence detection
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best_idx]]

        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad, bh * pad

        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(w, int(x2 + px))
        cy2 = min(h, int(y2 + py))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        return crop, (cx1, cy1, cx2, cy2)

    except Exception as e:
        print(f"YOLO crop error: {e}")
        return None


# ─── Single-frame pose inference ──────────────────────────────────────────────

def _run_mediapipe_on_frame(
    frame_bgr: np.ndarray,
    complexity: int = 1,
    conf: float = 0.30,
) -> Optional[dict]:
    """
    Run MediaPipe Pose on a single BGR frame.
    Pipeline: resize → YOLO crop (if available) → RGB convert → MediaPipe.
    Returns landmark dict (index → [x, y, z, vis]) or None.
    """
    import cv2
    import mediapipe as mp

    frame_bgr = _resize_frame(frame_bgr)

    crop_result = _yolo_crop_person(frame_bgr)
    if crop_result is not None:
        crop_frame, (cx1, cy1, cx2, cy2) = crop_result
        input_frame = crop_frame
    else:
        input_frame = frame_bgr
        cx1 = cy1 = 0
        cx2, cy2 = frame_bgr.shape[1], frame_bgr.shape[0]

    frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    try:
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=complexity,
            enable_segmentation=False,
            min_detection_confidence=conf,
            min_tracking_confidence=conf,
        ) as pose:
            results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        # Re-project normalised coords back to full-frame if a crop was used
        crop_h, crop_w = input_frame.shape[:2]
        full_h, full_w = frame_bgr.shape[:2]

        landmarks: dict = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if crop_result is not None:
                # lm.x/y are relative to the crop; convert to full-frame
                abs_x = cx1 + lm.x * crop_w
                abs_y = cy1 + lm.y * crop_h
                nx = abs_x / full_w
                ny = abs_y / full_h
            else:
                nx, ny = lm.x, lm.y

            landmarks[idx] = [round(nx, 4), round(ny, 4),
                               round(lm.z, 4), round(lm.visibility, 4)]
        return landmarks

    except Exception as e:
        print(f"MediaPipe inference error: {e}")
        return None


# ─── Public API: pose from video frames ───────────────────────────────────────

def analyze_pose_from_video_frames(frames: List[np.ndarray]) -> List[Optional[dict]]:
    """
    Run the full YOLO→crop→MediaPipe pipeline on every frame.
    Two-pass strategy: complexity=1 first; if <20% valid, retry with complexity=2.
    """
    if not frames:
        return []

    def _run_pass(complexity: int, conf: float) -> List[Optional[dict]]:
        out = []
        for frm in frames:
            lm = _run_mediapipe_on_frame(frm, complexity=complexity, conf=conf)
            out.append(lm)
        return out

    # Pass 1
    results = _run_pass(complexity=1, conf=0.30)
    valid = sum(1 for r in results if r is not None)
    print(f"Pose pass-1 (c=1, conf=0.30): {valid}/{len(frames)} detections")

    # Pass 2 — only if very few detections
    if valid < max(1, len(frames) * 0.20):
        print("Low detection rate — retrying with complexity=2, conf=0.25 ...")
        results2 = _run_pass(complexity=2, conf=0.25)
        valid2 = sum(1 for r in results2 if r is not None)
        print(f"Pose pass-2 (c=2, conf=0.25): {valid2}/{len(frames)} detections")
        # Merge: prefer pass-2 result when pass-1 missed
        for i, (r1, r2) in enumerate(zip(results, results2)):
            if r1 is None and r2 is not None:
                results[i] = r2
        valid = sum(1 for r in results if r is not None)

    print(f"Final detection: {valid}/{len(frames)} frames with pose")
    return results


# ─── Public API: pose from single image ───────────────────────────────────────

def analyze_pose_from_image(image_path: str) -> Optional[dict]:
    """Run YOLO→crop→MediaPipe on a single image file."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Try complexity=1 first, then 2
    for c, conf in [(1, 0.30), (2, 0.25)]:
        lm = _run_mediapipe_on_frame(image, complexity=c, conf=conf)
        if lm is not None:
            return lm
    return None


# ─── Cricket content validation ───────────────────────────────────────────────

async def validate_cricket_content_async(image: np.ndarray) -> bool:
    """
    Validate that a frame shows cricket content using Gemini Vision.
    Returns True = likely cricket; False = clearly non-cricket.
    Falls through to True on any error (do not penalise valid users for API issues).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return True

    try:
        import cv2
        import PIL.Image
        from google import genai

        client = genai.Client(api_key=api_key)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)

        prompt = (
            "Does this image show someone playing cricket (batting, bowling, "
            "fielding) or a cricket ground? Reply ONLY with YES or NO."
        )
        response = await client.aio.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            contents=[prompt, pil_img],
        )
        return "YES" in (response.text or "").strip().upper()

    except Exception as e:
        print(f"Cricket validation error (non-fatal): {e}")
        return True


# ─── Ball speed estimation ────────────────────────────────────────────────────

def estimate_ball_speed(frames: List[np.ndarray]) -> Optional[float]:
    """
    Estimate ball speed in km/h using dense optical flow over resized frames.
    Returns None if fewer than 2 frames are provided or computation fails.
    """
    if len(frames) < 2:
        return None

    import cv2

    try:
        small = [_resize_frame(f) for f in frames]
        motion: List[float] = []
        for i in range(len(small) - 1):
            prev = cv2.cvtColor(small[i], cv2.COLOR_BGR2GRAY)
            nxt  = cv2.cvtColor(small[i + 1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            valid_mag = mag[np.isfinite(mag)]
            if valid_mag.size:
                motion.append(float(np.percentile(valid_mag, 92)))

        if not motion:
            return None

        motion_score = float(np.median(motion))
        speed_kmh = 80.0 + min(80.0, motion_score * 18.0)
        return round(float(np.clip(speed_kmh, 80.0, 160.0)), 1)

    except Exception as e:
        print(f"Ball speed estimation error: {e}")
        return None


# ─── Report generation ────────────────────────────────────────────────────────

def _parse_json_response_text(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


async def generate_report(metrics: dict, type_: str) -> dict:
    """Generate AI coaching report via Gemini; falls back to rule-based."""
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key:
        return _generate_rule_based_report(metrics, type_)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _generate_rule_based_report(metrics, type_)

    client = genai.Client(api_key=api_key)

    prompt = f"""You are a world-class professional cricket coach and biomechanics expert.
Analyze these player metrics from pose estimation and produce a deep technical coaching report.

Analysis Type: {type_.capitalize()}
Metrics: {json.dumps(metrics, default=str)}

Respond in JSON ONLY with this exact structure:
{{
  "strengths": ["3-5 specific technical strengths"],
  "weaknesses": ["2-4 specific improvement areas"],
  "mistakes": ["3-4 technical mistakes inferred from the metrics"],
  "improvement_suggestions": ["5-7 actionable coaching cues"],
  "training_drills": ["4-6 drills with sets/reps/focus"],
  "recommendations": ["3-5 equipment or conditioning notes"],
  "best_practices": ["4-6 elite performance habits"]
}}
Use professional cricket terminology. Base all feedback on the numeric metrics provided."""

    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.45,
                response_mime_type="application/json",
            ),
        )
        parsed = _parse_json_response_text(response.text or "")
        return {
            "strengths":             parsed.get("strengths") or [],
            "weaknesses":            parsed.get("weaknesses") or [],
            "mistakes":              parsed.get("mistakes") or [],
            "improvement_suggestions": parsed.get("improvement_suggestions") or [],
            "training_drills":       parsed.get("training_drills") or [],
            "recommendations":       parsed.get("recommendations") or [],
            "best_practices":        parsed.get("best_practices") or [],
        }
    except Exception as e:
        print(f"Gemini report error: {e}")
        return _generate_rule_based_report(metrics, type_)


def _generate_rule_based_report(metrics: dict, type_: str) -> dict:
    """Metric-driven fallback report — no static placeholder text."""
    strengths, weaknesses, suggestions, drills = [], [], [], []
    mistakes, recommendations = [], []
    best_practices = [
        "Perform 15 min dynamic mobility warm-up before every session",
        "Review your analysis footage weekly to track technique changes",
        "Maintain a training journal — log scores each session",
    ]

    if type_ == "batting":
        m = metrics
        if m.get("stanceScore", 0) >= 80:
            strengths.append("Solid batting stance — good base for consistent shot-making")
        else:
            weaknesses.append("Stance needs refinement — feet positioning is inconsistent")
            suggestions.append("Stance mirror drill: 10 min shadow batting with coach observation")
            drills.append("Shadow batting drill: 3 sets × 20 reps focusing on guard and balance")

        bat_angle = m.get("batSwingAngle")
        if bat_angle is not None:
            if 30 <= bat_angle <= 65:
                strengths.append(f"Good bat swing plane ({bat_angle}°) — drives on the up well")
            else:
                weaknesses.append(f"Bat swing angle ({bat_angle}°) deviates from ideal 30-65° range")
                suggestions.append("Throwdown sessions focusing on straight/cover drive swing plane")
                drills.append("Vertical bat drill: 20 throwdowns — straight bat, eyes on the ball")

        if m.get("timingScore", 0) >= 75:
            strengths.append("Good weight transfer — effective timing through impact")
        else:
            weaknesses.append("Weight transfer lag — timing through impact is delayed")
            suggestions.append("Use a hanging ball trainer to sharpen eye-hand coordination")
            drills.append("Hanging ball drill: 30 min daily — trigger early, hit at peak")
            mistakes.append("Late weight transfer causing bottom-hand dominance at impact")

        if m.get("followThroughScore", 0) >= 75:
            strengths.append("Complete follow-through — maximum power extraction")
        else:
            weaknesses.append("Incomplete follow-through — losing pace and control on the shot")
            drills.append("Shadow follow-through drill: 100 reps ending high with bat face open")
            mistakes.append("Abbreviated follow-through restricting shot trajectory and power")

        if m.get("headPositionScore", 100) < 70:
            mistakes.append("Head falling over the off-side — disrupts balance and shot selection")
        if not mistakes:
            mistakes.append("Slight head movement at the point of impact")

        recommendations = [
            "Consider a bat 1.1-1.2 kg for better swing speed and balance",
            "Review your analysis footage weekly to track muscle memory changes",
        ]

    elif type_ == "bowling":
        m = metrics
        speed = m.get("estimatedBallSpeed", 0)
        if speed >= 130:
            strengths.append(f"High pace at {speed} km/h — effective at club and district level")
        elif speed >= 100:
            strengths.append(f"Good medium-fast pace at {speed} km/h — consistent and controllable")
        elif speed > 0:
            weaknesses.append(f"Ball speed ({speed} km/h) below target — improve run-up momentum")
            suggestions.append("Extend run-up by 3 strides and accelerate into the crease")
            drills.append("Run-up acceleration drill: 20 deliveries focusing on full approach speed")

        if m.get("wristPositionScore", 0) >= 80:
            strengths.append("Strong wrist position at release — good seam stability and carry")
        else:
            weaknesses.append("Dropped wrist at release — affects seam position and swing")
            drills.append("Wrist strength drill: 3 × 15 reps with resistance band")
            mistakes.append("Dropped wrist at release disrupting seam position")

        if m.get("balanceScore", 0) >= 75:
            strengths.append("Stable body through delivery stride — good control")
        else:
            weaknesses.append("Unstable base in delivery stride — injury risk and inconsistency")
            suggestions.append("Core conditioning: planks and single-leg balance work")
            drills.append("Single-leg balance drill: 3 × 30 s each side")
            mistakes.append("Unstable landing causing inconsistent release angle")

        if not mistakes:
            mistakes.append("Minor inconsistency in run-up rhythm before release")
        recommendations = [
            "Bowling spikes should have excellent grip on the landing foot",
            "Ice the bowling shoulder for 10 min after intensive sessions",
        ]

    elif type_ == "posture":
        m = metrics
        if m.get("shoulderAlignmentScore", 0) >= 80:
            strengths.append("Good shoulder alignment — balanced upper body for all strokes")
        else:
            weaknesses.append("Shoulder imbalance — may indicate rotator-cuff asymmetry")
            suggestions.append("Shoulder mobility routine: arm circles + cross-body stretch (10 min)")
            drills.append("Shoulder mobility drill: 10 min daily before training")

        knee_angle = m.get("kneeBendAngle")
        if m.get("kneeBendScore", 0) >= 75:
            strengths.append(f"Good knee bend ({knee_angle}°) — solid athletic base position")
        else:
            weaknesses.append(f"Knee bend angle ({knee_angle}°) outside ideal 140-170° range")
            drills.append("Bodyweight squat: 3 × 20 reps to build leg strength and mobility")
            mistakes.append("Excessive or insufficient knee bend reducing batting/fielding explosiveness")

        if m.get("balanceScore", 0) >= 75:
            strengths.append("Good hip alignment — stable centre of gravity")
        else:
            weaknesses.append("Hip tilt detected — shifts centre of gravity off midline")
            drills.append("Warrior I/II yoga poses: 3 × 30 s each side")
            mistakes.append("Hip tilt compromising lateral movement and weight transfer")

        if not mistakes:
            mistakes.append("Minor postural drift during dynamic movement")
        recommendations = [
            "Focus on core conditioning: 3 × 30 s plank + side plank each session",
            "Consider a physiotherapy assessment for shoulder/hip asymmetry",
        ]

    if not strengths:
        strengths.append("Shows commitment to technique improvement")
    if not suggestions:
        suggestions.append("Maintain current technique — focus on match-play consistency")
    if not drills:
        drills.append("Continue regular training: minimum 3 sessions per week")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "mistakes": mistakes,
        "improvement_suggestions": suggestions,
        "training_drills": drills,
        "recommendations": recommendations,
        "best_practices": best_practices,
    }
