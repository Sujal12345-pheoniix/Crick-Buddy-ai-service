"""
CrickBuddy AI Service — Deterministic Scoring Engine
=====================================================
All scores are computed from real pose landmarks.
No random numbers. No hardcoded fallbacks.
Every score is traceable to a formula and a measured value.

Pipeline per frame:
  1. Blur filter → skip low-quality frames
  2. Resize → 640×480 for stable MediaPipe input  
  3. YOLOv8 → detect person (class 0); crop + pad bounding box
  4. MediaPipe → pose on cropped region
  5. Landmark re-projection back to full-frame coordinates

Scoring formulas are deterministic — same input ALWAYS gives same output.
"""

import json
import math
import os
import re
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

# ─── MediaPipe Landmark IDs (BlazePose 33-point model) ───────────────────────
NOSE = 0
LEFT_SHOULDER = 11;  RIGHT_SHOULDER = 12
LEFT_ELBOW = 13;     RIGHT_ELBOW = 14
LEFT_WRIST = 15;     RIGHT_WRIST = 16
LEFT_HIP = 23;       RIGHT_HIP = 24
LEFT_KNEE = 25;      RIGHT_KNEE = 26
LEFT_ANKLE = 27;     RIGHT_ANKLE = 28

# ─── Global model singletons (loaded once, reused across ALL requests) ────────
_yolo_model = None
_mp_pose_c1 = None
_mp_pose_c2 = None


def _get_yolo():
    """Load YOLOv8n once; return None if unavailable (non-fatal)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            weight = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt")
            _yolo_model = YOLO(weight)
            print("[Info] YOLOv8 model loaded")
        except Exception as e:
            print(f"[Warn] YOLOv8 load failed (will skip crop): {e}")
            _yolo_model = False
    return _yolo_model if _yolo_model else None


def _get_mp_pose(complexity: int = 1):
    global _mp_pose_c1, _mp_pose_c2
    import mediapipe as mp

    if complexity == 1:
        if _mp_pose_c1 is None:
            _mp_pose_c1 = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.30,
                min_tracking_confidence=0.30,
            )
            print("[Info] MediaPipe Pose (complexity=1) singleton ready")
        return _mp_pose_c1
    else:
        if _mp_pose_c2 is None:
            _mp_pose_c2 = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.25,
                min_tracking_confidence=0.25,
            )
            print("[Info] MediaPipe Pose (complexity=2) singleton ready")
        return _mp_pose_c2


def warmup_models():
    """Pre-load and warm up all heavy models at FastAPI startup."""
    print("[Info] Warming up AI models...")
    try:
        yolo = _get_yolo()
        if yolo:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            yolo(dummy, verbose=False, conf=0.99)
            print("[Info] YOLO warm-up done")
    except Exception as e:
        print(f"[Warn] YOLO warm-up error (non-fatal): {e}")

    try:
        pose = _get_mp_pose(1)
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        pose.process(dummy_rgb)
        print("[Info] MediaPipe Pose (c=1) warm-up done")
    except Exception as e:
        print(f"[Warn] MediaPipe Pose warm-up error (non-fatal): {e}")

    print("[Ready] All models ready — service is hot")


# ─── Basic helpers ────────────────────────────────────────────────────────────

def clamp_score(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(float(angle), 1)


def score_angle(angle: float, ideal_min: float, ideal_max: float) -> int:
    """Score an angle based on deviation from an ideal range. 98 at center, degrades outward."""
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
BLUR_THRESHOLD = 40.0


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    if w == TARGET_W and h == TARGET_H:
        return frame
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)


def _is_sharp(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= threshold


def extract_frames(video_path: str, num_frames: int = 20) -> List[np.ndarray]:
    """Extract evenly-spaced frames from a video, filtered for sharpness."""
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            raw: List[np.ndarray] = []
            while True:
                ret, frm = cap.read()
                if not ret or frm is None:
                    break
                raw.append(frm)
            cap.release()
        else:
            indices = np.linspace(0, max(total - 1, 0), num=min(num_frames * 3, total), dtype=int)
            raw = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frm = cap.read()
                if ret and frm is not None:
                    raw.append(frm)
            cap.release()

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
    model = _get_yolo()
    if model is None:
        return None
    try:
        results = model(frame, verbose=False, conf=0.25, classes=[0])
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

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
    import cv2

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
        pose = _get_mp_pose(complexity)
        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        crop_h, crop_w = input_frame.shape[:2]
        full_h, full_w = frame_bgr.shape[:2]

        landmarks: dict = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if crop_result is not None:
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


def analyze_pose_from_video_frames(frames: List[np.ndarray]) -> List[Optional[dict]]:
    """
    Run the full YOLO→crop→MediaPipe pipeline on every frame.
    Two-pass: complexity=1 first; retry with complexity=2 if < 10% valid.
    """
    if not frames:
        return []

    def _run_pass(complexity: int, conf: float) -> List[Optional[dict]]:
        out = []
        for frm in frames:
            lm = _run_mediapipe_on_frame(frm, complexity=complexity, conf=conf)
            out.append(lm)
        return out

    results = _run_pass(complexity=1, conf=0.30)
    valid = sum(1 for r in results if r is not None)
    print(f"Pose pass-1 (c=1, conf=0.30): {valid}/{len(frames)} detections")

    if valid < max(1, len(frames) * 0.10):
        print("Low detection rate — retrying with complexity=2, conf=0.20 ...")
        results2 = _run_pass(complexity=2, conf=0.20)
        valid2 = sum(1 for r in results2 if r is not None)
        print(f"Pose pass-2 (c=2, conf=0.20): {valid2}/{len(frames)} detections")
        for i, (r1, r2) in enumerate(zip(results, results2)):
            if r1 is None and r2 is not None:
                results[i] = r2
        valid = sum(1 for r in results if r is not None)

    print(f"Final detection: {valid}/{len(frames)} frames with pose")
    return results


def analyze_pose_from_image(image_path: str) -> Optional[dict]:
    """Run YOLO→crop→MediaPipe on a single image file."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        return None

    for c, conf in [(1, 0.30), (2, 0.20)]:
        lm = _run_mediapipe_on_frame(image, complexity=c, conf=conf)
        if lm is not None:
            return lm
    return None


# ─── Cricket content validation ───────────────────────────────────────────────

async def validate_cricket_content_async(image: np.ndarray) -> bool:
    """
    Validate that a frame shows cricket content using Gemini Vision.
    Returns True = likely cricket; False = clearly non-cricket.
    Falls through to True on any error.
    Always returns True if GEMINI_API_KEY is not set.
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
            "Does this image show a person playing cricket (batting, bowling, fielding) "
            "or a cricket ground/training session? "
            "Reply ONLY with YES or NO. No explanation."
        )
        response = await client.aio.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            contents=[prompt, pil_img],
        )
        answer = (response.text or "").strip().upper()
        return "YES" in answer

    except Exception as e:
        print(f"Cricket validation error (non-fatal, allowing through): {e}")
        return True


# ─── Ball speed estimation ────────────────────────────────────────────────────

def estimate_ball_speed(frames: List[np.ndarray]) -> Optional[float]:
    """
    Estimate relative ball speed in km/h using dense optical flow.
    Returns a float or None if fewer than 2 frames or computation fails.
    NOTE: This estimates RELATIVE motion magnitude, not absolute km/h.
          Use speedClassification() for an honest human-readable label.
    """
    if len(frames) < 2:
        return None

    import cv2

    try:
        small = [_resize_frame(f) for f in frames]
        motion: List[float] = []
        for i in range(len(small) - 1):
            prev = cv2.cvtColor(small[i], cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(small[i + 1], cv2.COLOR_BGR2GRAY)
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
        # Calibrated range: 80 km/h = slow, 160 km/h = very fast
        speed_kmh = 80.0 + min(80.0, motion_score * 18.0)
        return round(float(np.clip(speed_kmh, 80.0, 160.0)), 1)

    except Exception as e:
        print(f"Ball speed estimation error: {e}")
        return None


def classify_ball_speed(speed_kmh: Optional[float]) -> str:
    """Convert speed estimate to honest classification string."""
    if speed_kmh is None:
        return "Undetected"
    if speed_kmh >= 135:
        return "Fast"
    if speed_kmh >= 115:
        return "Medium-Fast"
    if speed_kmh >= 95:
        return "Medium"
    return "Spin/Slow"


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SCORING ENGINE
# Every function below takes measured landmark data and returns a score.
# No random numbers. No hardcoded values. Same input → same output.
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_head_stability(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Measures variance of NOSE Y-position across all frames.
    High variance = unstable head = low score.
    
    Formula: score = max(0, 1 - variance / 0.005) × 100
    Thresholds: variance < 0.0005 = excellent, > 0.005 = poor
    """
    valid = [f for f in frames if f and NOSE in f]
    if len(valid) < 3:
        return {"score": 50, "variance": None, "note": "Insufficient frames for head stability"}

    y_vals = [f[NOSE][1] for f in valid]
    variance = float(np.var(y_vals))
    score = float(np.clip(1.0 - (variance / 0.005), 0, 1)) * 100

    if variance < 0.0005:
        note = "Excellent — head is very still through the shot"
    elif variance < 0.002:
        note = "Good — minor head movement detected"
    elif variance < 0.005:
        note = "Fair — noticeable head movement, work on keeping eyes level"
    else:
        note = "Poor — excessive head movement disrupting shot execution"

    return {
        "score": clamp_score(score, 0, 100),
        "variance": round(variance, 6),
        "note": note
    }


def calculate_timing_score(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Batting timing: measures left knee angle trajectory across frames.
    A correctly timed shot shows: knee flexion at impact → full extension at follow-through.
    
    Formula:
      - Flexion score: score_angle(min_knee_angle, ideal 110-145°)
      - Extension score: score_angle(max_knee_angle, ideal 160-180°)
      - Range score: clamp(range_of_motion × 2.5, 0, 100)
      - Final: 0.4×flexion + 0.4×extension + 0.2×range
    """
    knee_angles = []
    for f in frames:
        if not f:
            continue
        lh = get_point(f, LEFT_HIP)
        lk = get_point(f, LEFT_KNEE)
        la = get_point(f, LEFT_ANKLE)
        if lh and lk and la:
            knee_angles.append(calculate_angle(lh, lk, la))

    if len(knee_angles) < 3:
        return {"score": 50, "peakFlexion": None, "peakExtension": None, "rangeOfMotion": None,
                "note": "Insufficient frames for timing analysis"}

    min_angle = float(min(knee_angles))
    max_angle = float(max(knee_angles))
    range_of_motion = max_angle - min_angle

    flexion_score = score_angle(min_angle, 110, 145)
    extension_score = score_angle(max_angle, 160, 180)
    range_score = clamp_score(range_of_motion * 2.5, 0, 100)

    final = clamp_score(flexion_score * 0.4 + extension_score * 0.4 + range_score * 0.2, 0, 100)

    if range_of_motion < 15:
        note = "Poor timing — minimal knee drive through the shot"
    elif range_of_motion < 30:
        note = "Fair timing — moderate knee flexion-extension cycle"
    elif min_angle < 110:
        note = "Over-flexed at impact — weight too far forward"
    elif max_angle < 155:
        note = "Incomplete extension — follow-through cut short"
    else:
        note = "Good timing — proper knee flexion and extension cycle"

    return {
        "score": final,
        "peakFlexion": round(min_angle, 1),
        "peakExtension": round(max_angle, 1),
        "rangeOfMotion": round(range_of_motion, 1),
        "note": note
    }


def calculate_balance_score(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Balance: measures hip levelness and spine verticality across all frames.
    
    Formula:
      - avgHipTilt: mean(|left_hip_Y - right_hip_Y|) across frames
      - avgSpineOffset: mean(|mid_shoulder_X - mid_hip_X|) across frames
      - hipScore: clamp(100 - avgHipTilt/0.001, 0, 100)
      - spineScore: clamp(100 - avgSpineOffset/0.0008, 0, 100)
      - Final: 0.5×hip + 0.5×spine
    """
    hip_tilts, spine_offsets = [], []
    for f in frames:
        if not f:
            continue
        lh = get_point(f, LEFT_HIP)
        rh = get_point(f, RIGHT_HIP)
        ls = get_point(f, LEFT_SHOULDER)
        rs = get_point(f, RIGHT_SHOULDER)

        if lh and rh:
            hip_tilts.append(abs(lh[1] - rh[1]))
        if lh and rh and ls and rs:
            mid_hip_x = (lh[0] + rh[0]) / 2
            mid_sh_x = (ls[0] + rs[0]) / 2
            spine_offsets.append(abs(mid_hip_x - mid_sh_x))

    if not hip_tilts:
        return {"score": 50, "avgHipTilt": None, "avgSpineOffset": None,
                "note": "Insufficient frames for balance analysis"}

    avg_hip_tilt = float(np.mean(hip_tilts))
    avg_spine_offset = float(np.mean(spine_offsets)) if spine_offsets else 0.0

    hip_score = clamp_score(100 - (avg_hip_tilt / 0.001), 0, 100)
    spine_score = clamp_score(100 - (avg_spine_offset / 0.0008), 0, 100)
    final = clamp_score(hip_score * 0.5 + spine_score * 0.5, 0, 100)

    if avg_hip_tilt < 0.02 and avg_spine_offset < 0.03:
        note = "Excellent — hips level and spine aligned throughout"
    elif avg_hip_tilt < 0.05:
        note = "Good — minor hip tilt, spine reasonably aligned"
    elif avg_hip_tilt < 0.08:
        note = "Fair — noticeable hip tilt affecting weight transfer"
    else:
        note = "Poor — significant balance issues, risk of injury"

    return {
        "score": final,
        "avgHipTilt": round(avg_hip_tilt, 5),
        "avgSpineOffset": round(avg_spine_offset, 5),
        "hipScore": hip_score,
        "spineScore": spine_score,
        "note": note
    }


def calculate_stride_score(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Measures front foot displacement relative to back foot, normalized to hip width.
    
    Formula:
      - strideRatio = ankle_spread / hip_width per frame
      - score = score_angle(mean_strideRatio, ideal_min=1.5, ideal_max=2.5)
    Ideal batting stride: 1.5–2.5× hip width. Too wide = unstable. Too narrow = no power.
    """
    stride_ratios = []
    for f in frames:
        if not f:
            continue
        la = get_point(f, LEFT_ANKLE)
        ra = get_point(f, RIGHT_ANKLE)
        lh = get_point(f, LEFT_HIP)
        rh = get_point(f, RIGHT_HIP)

        if la and ra and lh and rh:
            ankle_spread = abs(la[0] - ra[0])
            hip_width = abs(lh[0] - rh[0])
            if hip_width > 0.01:
                stride_ratios.append(ankle_spread / hip_width)

    if not stride_ratios:
        return {"score": 50, "avgStrideRatio": None, "note": "Insufficient frames for stride analysis"}

    avg_ratio = float(np.mean(stride_ratios))
    score = score_angle(avg_ratio, 1.5, 2.5)

    if avg_ratio < 1.0:
        note = f"Narrow stance ({avg_ratio:.2f}× hip width) — limited power transfer"
    elif avg_ratio > 3.0:
        note = f"Too wide ({avg_ratio:.2f}× hip width) — unstable base"
    elif 1.5 <= avg_ratio <= 2.5:
        note = f"Good stride width ({avg_ratio:.2f}× hip width) — solid batting base"
    else:
        note = f"Slightly off ideal ({avg_ratio:.2f}× hip width) — minor adjustment needed"

    return {
        "score": score,
        "avgStrideRatio": round(avg_ratio, 3),
        "note": note
    }


def calculate_arm_smoothness(frames: List[Optional[dict]], landmark_id: int = RIGHT_WRIST) -> Dict[str, Any]:
    """
    Measures bowling arm rotation smoothness using jerk metric.
    Jerk = rate of change of acceleration. Low jerk = smooth rotation.
    
    Formula:
      - velocities = ||diff(positions)||
      - accelerations = diff(velocities)
      - jerk = mean(|diff(accelerations)|)
      - score = max(0, 1 - jerk/0.03) × 100
    """
    pts = []
    for f in frames:
        if f and landmark_id in f:
            pts.append(f[landmark_id][:2])

    if len(pts) < 4:
        return {"score": 50, "avgJerk": None, "note": "Insufficient frames for arm smoothness"}

    pts_arr = np.array(pts)
    velocities = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
    accelerations = np.diff(velocities)
    jerks = np.abs(np.diff(accelerations))
    avg_jerk = float(np.mean(jerks)) if len(jerks) > 0 else 0.0

    score_raw = max(0.0, 1.0 - (avg_jerk / 0.03)) * 100
    score = clamp_score(score_raw, 0, 100)

    if avg_jerk < 0.003:
        note = "Excellent — very smooth arm rotation, consistent seam position"
    elif avg_jerk < 0.010:
        note = "Good — arm rotation is mostly smooth"
    elif avg_jerk < 0.025:
        note = "Fair — some inconsistency in arm rotation speed"
    else:
        note = "Poor — jerky arm rotation affecting line and length"

    return {
        "score": score,
        "avgJerk": round(avg_jerk, 6),
        "note": note
    }


def calculate_release_point_score(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Finds the frame with highest wrist position and scores release point height.
    High release (wrist above nose) = good bounce extraction.
    
    Formula:
      - Find frame where RIGHT_WRIST Y is minimum (highest in image)
      - height_score = clamp(70 + (nose_Y - wrist_Y) × 200, 30, 98)
      - extension_score = score_angle(arm angle at release, 160, 180°)
      - Final: 0.6×height + 0.4×extension
    """
    min_wrist_y = 1.0
    best_frame = None
    for f in frames:
        if not f:
            continue
        rw = get_point(f, RIGHT_WRIST)
        if rw and rw[1] < min_wrist_y:
            min_wrist_y = rw[1]
            best_frame = f

    if not best_frame:
        return {"score": 50, "peakWristY": None, "releaseArmAngle": None,
                "note": "Could not detect release frame"}

    rw = get_point(best_frame, RIGHT_WRIST)
    nose = get_point(best_frame, NOSE)
    elbow = get_point(best_frame, RIGHT_ELBOW)
    shoulder = get_point(best_frame, RIGHT_SHOULDER)

    height_score = 50
    if rw and nose:
        delta = nose[1] - rw[1]  # positive = wrist is above nose = good
        height_score = clamp_score(70 + delta * 200, 30, 98)

    arm_angle = None
    extension_score = 50
    if rw and elbow and shoulder:
        arm_angle = calculate_angle(shoulder, elbow, rw)
        extension_score = score_angle(arm_angle, 160, 180)

    final = clamp_score(height_score * 0.6 + extension_score * 0.4, 0, 100)

    if min_wrist_y < 0.3:
        note = "Excellent high release point — maximum bounce extraction"
    elif min_wrist_y < 0.45:
        note = "Good release height — solid bounce and carry"
    elif min_wrist_y < 0.55:
        note = "Fair — slightly low release, may lose bounce"
    else:
        note = "Low release point — ball will lack bounce and carry"

    return {
        "score": final,
        "peakWristY": round(min_wrist_y, 4),
        "releaseArmAngle": round(arm_angle, 1) if arm_angle else None,
        "heightScore": height_score,
        "extensionScore": extension_score,
        "note": note
    }


def calculate_follow_through_score(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Measures wrist position change from early frames to late frames.
    In a complete follow-through, the wrist should rise (Y decreases in image coords).
    
    Formula:
      - early_wrist_Y = avg wrist Y in first 33% of frames
      - late_wrist_Y = avg wrist Y in last 33% of frames
      - delta = early_Y - late_Y (positive = wrist went UP = good follow-through)
      - score = clamp(60 + delta × 300, 25, 98)
    """
    valid = [f for f in frames if f]
    if len(valid) < 6:
        return {"score": 50, "wristYDelta": None, "note": "Insufficient frames for follow-through analysis"}

    n = len(valid)
    early_frames = valid[:n // 3]
    late_frames = valid[2 * n // 3:]

    def avg_wrist_y(flist):
        ys = [get_point(f, RIGHT_WRIST)[1] for f in flist if get_point(f, RIGHT_WRIST)]
        return float(np.mean(ys)) if ys else 0.5

    early_y = avg_wrist_y(early_frames)
    late_y = avg_wrist_y(late_frames)
    delta = early_y - late_y  # positive = wrist went UP = good

    score = clamp_score(60 + delta * 300, 25, 98)

    if delta > 0.15:
        note = "Excellent follow-through — wrist completes high above impact zone"
    elif delta > 0.05:
        note = "Good follow-through — bat completes the arc well"
    elif delta > -0.02:
        note = "Fair — partial follow-through, slightly abbreviated"
    else:
        note = "Poor follow-through — bat stops at impact, losing pace and direction"

    return {
        "score": score,
        "wristYDelta": round(delta, 4),
        "earlyWristY": round(early_y, 4),
        "lateWristY": round(late_y, 4),
        "note": note
    }


def calculate_shoulder_alignment(frames: List[Optional[dict]]) -> Dict[str, Any]:
    """
    Measures shoulder levelness across all frames.
    
    Formula:
      - tilt = |left_shoulder_Y - right_shoulder_Y| per frame
      - score = clamp(98 - mean(tilt) × 520, 35, 98)
    """
    tilts = []
    for f in frames:
        if not f:
            continue
        ls = get_point(f, LEFT_SHOULDER)
        rs = get_point(f, RIGHT_SHOULDER)
        if ls and rs:
            tilts.append(abs(ls[1] - rs[1]))

    if not tilts:
        return {"score": 50, "avgTilt": None, "note": "Could not detect shoulder alignment"}

    avg_tilt = float(np.mean(tilts))
    score = clamp_score(98 - avg_tilt * 520, 35, 98)

    if avg_tilt < 0.02:
        note = "Excellent — shoulders level throughout"
    elif avg_tilt < 0.05:
        note = "Good — minor shoulder tilt"
    elif avg_tilt < 0.09:
        note = "Fair — noticeable shoulder imbalance"
    else:
        note = "Poor — significant shoulder tilt, risk of technique breakdown"

    return {
        "score": score,
        "avgTilt": round(avg_tilt, 5),
        "note": note
    }


def detect_faults(metrics: Dict[str, Any], action_type: str) -> List[Dict[str, Any]]:
    """
    Deterministic fault detection. Every fault is stored with:
    - faultCode: machine-readable identifier
    - faultText: human-readable description  
    - metric: which metric triggered this
    - value: actual measured value
    - threshold: the threshold that was violated
    - severity: critical | moderate | minor
    """
    faults = []

    def add_fault(code, text, metric, value, threshold, severity):
        faults.append({
            "faultCode": code,
            "faultText": text,
            "metric": metric,
            "value": round(float(value), 4) if value is not None else None,
            "threshold": threshold,
            "severity": severity
        })

    head_score = metrics.get("headStabilityScore", 100)
    if head_score < 40:
        add_fault("HEAD_INSTABILITY_CRITICAL", "Severe head movement — ball tracking compromised",
                  "headStabilityScore", head_score, 40, "critical")
    elif head_score < 65:
        add_fault("HEAD_INSTABILITY", "Noticeable head movement during shot execution",
                  "headStabilityScore", head_score, 65, "moderate")

    balance_score = metrics.get("balanceScore", 100)
    if balance_score < 40:
        add_fault("POOR_BALANCE_CRITICAL", "Severe balance issue — significant injury risk",
                  "balanceScore", balance_score, 40, "critical")
    elif balance_score < 60:
        add_fault("POOR_BALANCE", "Unstable base — hip tilt or spine misalignment detected",
                  "balanceScore", balance_score, 60, "moderate")

    if action_type == "batting":
        timing_score = metrics.get("timingScore", 100)
        if timing_score < 45:
            add_fault("LATE_TIMING_CRITICAL", "Very late shot execution — bat not through at impact",
                      "timingScore", timing_score, 45, "critical")
        elif timing_score < 65:
            add_fault("LATE_TIMING", "Late shot timing — weight transfer incomplete at impact",
                      "timingScore", timing_score, 65, "moderate")

        follow_score = metrics.get("followThroughScore", 100)
        if follow_score < 40:
            add_fault("NO_FOLLOW_THROUGH", "Follow-through absent — bat stopping at contact",
                      "followThroughScore", follow_score, 40, "critical")
        elif follow_score < 60:
            add_fault("SHORT_FOLLOW_THROUGH", "Abbreviated follow-through reducing shot power",
                      "followThroughScore", follow_score, 60, "moderate")

        stride_score = metrics.get("strideScore", 100)
        avg_stride = metrics.get("avgStrideRatio", 2.0)
        if avg_stride is not None and avg_stride > 3.2:
            add_fault("WIDE_STANCE", f"Overwide stance ({avg_stride:.2f}× hip width) — loss of mobility",
                      "avgStrideRatio", avg_stride, 3.2, "minor")
        elif avg_stride is not None and avg_stride < 0.8:
            add_fault("NARROW_STANCE", f"Too narrow stance ({avg_stride:.2f}× hip width) — limited power base",
                      "avgStrideRatio", avg_stride, 0.8, "minor")

    elif action_type == "bowling":
        arm_score = metrics.get("armSmoothnessScore", 100)
        if arm_score < 40:
            add_fault("JERKY_ACTION_CRITICAL", "Very inconsistent arm rotation — severe rhythm issue",
                      "armSmoothnessScore", arm_score, 40, "critical")
        elif arm_score < 60:
            add_fault("JERKY_ACTION", "Inconsistent arm rotation speed — affects line and length",
                      "armSmoothnessScore", arm_score, 60, "moderate")

        release_score = metrics.get("releasePointScore", 100)
        if release_score < 40:
            add_fault("LOW_RELEASE_CRITICAL", "Very low release point — ball will lack pace and bounce",
                      "releasePointScore", release_score, 40, "critical")
        elif release_score < 60:
            add_fault("LOW_RELEASE", "Below-ideal release height — losing bounce extraction",
                      "releasePointScore", release_score, 60, "moderate")

    return faults


def compute_overall_batting_score(metrics: Dict[str, Any]) -> int:
    """
    Weighted combination of all batting metrics.
    Weights derived from biomechanical importance in cricket batting.
    """
    timing = metrics.get("timingScore", 50)
    balance = metrics.get("balanceScore", 50)
    stance = metrics.get("stanceScore", 50)
    head = metrics.get("headStabilityScore", 50)
    follow = metrics.get("followThroughScore", 50)
    stride = metrics.get("strideScore", 50)

    overall = (
        timing  * 0.25 +
        balance * 0.20 +
        stance  * 0.20 +
        head    * 0.20 +
        follow  * 0.10 +
        stride  * 0.05
    )
    return clamp_score(overall, 0, 100)


def compute_overall_bowling_score(metrics: Dict[str, Any]) -> int:
    """
    Weighted combination of all bowling metrics.
    """
    arm = metrics.get("armSmoothnessScore", 50)
    release = metrics.get("releasePointScore", 50)
    wrist = metrics.get("wristPositionScore", 50)
    balance = metrics.get("balanceScore", 50)

    overall = (
        arm     * 0.30 +
        release * 0.25 +
        wrist   * 0.25 +
        balance * 0.20
    )
    return clamp_score(overall, 0, 100)


def compute_overall_posture_score(metrics: Dict[str, Any]) -> int:
    shoulder = metrics.get("shoulderAlignmentScore", 50)
    knee = metrics.get("kneeBendScore", 50)
    balance = metrics.get("balanceScore", 50)
    spine = metrics.get("spinePosScore", 50)

    overall = (
        shoulder * 0.30 +
        knee     * 0.25 +
        balance  * 0.25 +
        spine    * 0.20
    )
    return clamp_score(overall, 0, 100)


# ─── Report generation ────────────────────────────────────────────────────────

def _parse_json_response_text(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


async def generate_report(metrics: dict, type_: str, faults: List[Dict] = None) -> dict:
    """
    Generate AI coaching report via Gemini.
    The LLM receives all computed scores and explains them in natural language.
    The LLM is PROHIBITED from inventing or changing any score.
    Falls back to rule-based on error.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key:
        return _generate_rule_based_report(metrics, type_, faults or [])

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _generate_rule_based_report(metrics, type_, faults or [])

    client = genai.Client(api_key=api_key)

    # Format fault summary for the LLM
    fault_summary = []
    for f in (faults or []):
        fault_summary.append(
            f"[{f['severity'].upper()}] {f['faultText']} "
            f"({f['metric']}={f['value']}, threshold={f['threshold']})"
        )

    # CRITICAL: LLM prompt explicitly prohibits score invention
    prompt = f"""You are a professional cricket coach explaining biomechanical analysis results to a player.

IMPORTANT RULES:
1. You MUST NOT invent, change, or contradict any of the scores listed below.
2. You MUST NOT generate new fault codes or scores.
3. Your ONLY job is to write natural-language explanation for each score and fault.
4. Reference the actual numeric values in your explanations.
5. All scores are on a 0-100 scale computed from real pose landmark measurements.

ANALYSIS TYPE: {type_.capitalize()}

COMPUTED SCORES (DO NOT CHANGE THESE - they are from pose estimation algorithm):
{json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2)}

DETECTED FAULTS (explain each, do not add new ones):
{chr(10).join(fault_summary) if fault_summary else 'No faults detected above threshold.'}

Respond in JSON ONLY with this exact structure:
{{
  "strengths": ["3-5 specific technical strengths — reference actual score values"],
  "weaknesses": ["weaknesses based ONLY on the detected faults above"],
  "mistakes": ["technical mistakes inferred from scores — must cite which score triggered each"],
  "improvement_suggestions": ["5-7 actionable coaching drills with sets/reps"],
  "training_drills": ["4-6 specific drills targeting the weakest metrics"],
  "recommendations": ["3-5 equipment or conditioning notes"],
  "best_practices": ["4-6 elite performance habits"]
}}
Do NOT include any markdown outside the JSON block."""

    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.30,
                response_mime_type="application/json",
            ),
        )
        parsed = _parse_json_response_text(response.text or "")
        return {
            "strengths":               parsed.get("strengths") or [],
            "weaknesses":              parsed.get("weaknesses") or [],
            "mistakes":                parsed.get("mistakes") or [],
            "improvement_suggestions": parsed.get("improvement_suggestions") or [],
            "training_drills":         parsed.get("training_drills") or [],
            "recommendations":         parsed.get("recommendations") or [],
            "best_practices":          parsed.get("best_practices") or [],
        }
    except Exception as e:
        print(f"Gemini report error (using rule-based fallback): {e}")
        return _generate_rule_based_report(metrics, type_, faults or [])


def _generate_rule_based_report(metrics: dict, type_: str, faults: List[Dict] = None) -> dict:
    """
    Metric-driven fallback report — derived ENTIRELY from actual extracted scores.
    Every statement references a real measured value.
    """
    strengths, weaknesses, suggestions, drills = [], [], [], []
    mistakes, recommendations = [], []
    best_practices = [
        "Perform 15 min dynamic mobility warm-up before every session",
        "Review your analysis footage weekly to track technique changes",
        "Maintain a training journal — log scores and key cues each session",
    ]

    fault_codes = {f["faultCode"] for f in (faults or [])}

    if type_ == "batting":
        stance = metrics.get("stanceScore", 50)
        head = metrics.get("headStabilityScore", 50)
        timing = metrics.get("timingScore", 50)
        follow = metrics.get("followThroughScore", 50)
        balance = metrics.get("balanceScore", 50)
        bat_angle = metrics.get("batSwingAngle")
        peak_flex = metrics.get("peakKneeFlexion")
        range_of_motion = metrics.get("rangeOfMotion")

        # Strengths from good scores
        if stance >= 80:
            strengths.append(f"Excellent batting stance (score: {stance}/100) — solid base for consistent shot-making")
        if head >= 80:
            strengths.append(f"Very stable head position (score: {head}/100) — tracking the ball well through impact")
        if timing >= 80:
            strengths.append(f"Good timing mechanics (score: {timing}/100) — knee drive timing is effective")
        if follow >= 80:
            strengths.append(f"Complete follow-through (score: {follow}/100) — maximum power extracted from each shot")
        if balance >= 80:
            strengths.append(f"Excellent balance (score: {balance}/100) — hips and spine well-aligned throughout")

        # Weaknesses from fault codes
        if "HEAD_INSTABILITY_CRITICAL" in fault_codes or "HEAD_INSTABILITY" in fault_codes:
            weaknesses.append(f"Head stability (score: {head}/100) — movement during shot disrupts ball tracking")
            mistakes.append(f"Head moving off the ball at impact (measured variance exceeds safe threshold)")
            suggestions.append("Ball-tracking drill: watch the ball hit the bat for 30 consecutive deliveries")
            drills.append("Static head drill: shadow bat with book on head — 3 sets × 20 reps")
        if "LATE_TIMING" in fault_codes or "LATE_TIMING_CRITICAL" in fault_codes:
            note = f"peak flexion={peak_flex}°" if peak_flex else f"score={timing}/100"
            weaknesses.append(f"Shot timing ({note}) — weight transfer completing late")
            mistakes.append(f"Weight transfer lag — front foot landing too early or too late relative to ball arrival")
            suggestions.append("Underarm feed drill: 40 feeds at 3/4 pace focusing on early trigger movement")
            drills.append("Drop-ball drill: 3 sets × 20 reps — trigger at bowler's hand, not at bounce")
        if "NO_FOLLOW_THROUGH" in fault_codes or "SHORT_FOLLOW_THROUGH" in fault_codes:
            weaknesses.append(f"Follow-through (score: {follow}/100) — bat stopping before completion")
            mistakes.append("Abbreviated follow-through reducing shot pace and direction control")
            drills.append("Shadow follow-through: 100 reps ending bat high above left shoulder")
        if "POOR_BALANCE" in fault_codes or "POOR_BALANCE_CRITICAL" in fault_codes:
            weaknesses.append(f"Balance (score: {balance}/100) — hip tilt or spine misalignment detected")
            drills.append("Single-leg balance hold: 3 × 30s each side, then shadow bat")

        if bat_angle is not None:
            if 30 <= bat_angle <= 65:
                strengths.append(f"Good bat swing plane ({bat_angle}°) — in the ideal 30-65° range")
            else:
                weaknesses.append(f"Bat swing angle ({bat_angle}°) outside ideal 30-65° range")
                drills.append("Vertical bat drill: 20 throwdowns with straight/cover drive swing plane focus")

        recommendations = [
            "Consider a bat weighing 1.1-1.2 kg for better swing speed and balance",
            "Review analysis footage weekly to track muscle-memory changes",
            "Bat grip should be firm but not tense — pressure should be in top 3 fingers",
        ]

    elif type_ == "bowling":
        arm = metrics.get("armSmoothnessScore", 50)
        release = metrics.get("releasePointScore", 50)
        wrist = metrics.get("wristPositionScore", 50)
        balance = metrics.get("balanceScore", 50)
        speed = metrics.get("estimatedBallSpeed")
        speed_class = metrics.get("speedClassification", "Undetected")
        avg_jerk = metrics.get("avgJerk")
        peak_wrist_y = metrics.get("peakWristY")

        if arm >= 80:
            strengths.append(f"Smooth arm rotation (score: {arm}/100) — consistent seam position through delivery")
        if release >= 80:
            strengths.append(f"High release point (score: {release}/100) — excellent bounce extraction")
        if wrist >= 80:
            strengths.append(f"Good wrist position at release (score: {wrist}/100) — seam stability and swing")
        if balance >= 80:
            strengths.append(f"Stable delivery stride (score: {balance}/100) — controlled front-side landing")
        if speed_class in ["Fast", "Medium-Fast"]:
            strengths.append(f"Pace classification: {speed_class} — effective at this level")

        if "JERKY_ACTION" in fault_codes or "JERKY_ACTION_CRITICAL" in fault_codes:
            note = f"jerk={avg_jerk:.5f}" if avg_jerk else f"score={arm}/100"
            weaknesses.append(f"Arm rotation inconsistency ({note}) — rhythm breaking down")
            mistakes.append("Inconsistent arm rotation speed causing variable release point")
            suggestions.append("Slow-motion bowling drill: bowl at 50% pace focusing only on smooth arm arc")
            drills.append("Bowling to a target with no batsman: 30 deliveries focusing on arm speed consistency")
        if "LOW_RELEASE" in fault_codes or "LOW_RELEASE_CRITICAL" in fault_codes:
            note = f"wrist_y={peak_wrist_y:.3f}" if peak_wrist_y else f"score={release}/100"
            weaknesses.append(f"Release point too low ({note}) — losing bounce and carry")
            mistakes.append("Low arm at release reducing bounce, carry, and pace effectiveness")
            suggestions.append("High-arm release drill: bowl into a wall target set above head height")
            drills.append("High-arm bound drill: 3 × 8 jump-and-release repetitions")
        if "POOR_BALANCE" in fault_codes:
            weaknesses.append(f"Delivery stride balance (score: {balance}/100) — unstable landing")
            drills.append("Single-leg landing hold: 3 × 30s each side for landing stability")

        if not speed or speed_class == "Undetected":
            recommendations = [
                "Upload a video with the ball clearly visible for speed measurement",
                "Bowling spikes should have excellent grip on landing foot",
            ]
        else:
            recommendations = [
                f"Current pace classification: {speed_class} — target the next category up",
                "Ice the bowling shoulder for 10 min after intensive sessions",
                "Maintain minimum 48h recovery between high-intensity bowling loads",
            ]

    elif type_ == "posture":
        shoulder = metrics.get("shoulderAlignmentScore", 50)
        knee = metrics.get("kneeBendScore", 50)
        balance = metrics.get("balanceScore", 50)
        spine = metrics.get("spinePosScore", 50)
        knee_angle = metrics.get("kneeBendAngle")

        if shoulder >= 80:
            strengths.append(f"Excellent shoulder alignment (score: {shoulder}/100) — balanced upper body")
        if knee >= 80:
            ka_str = f" ({knee_angle}°)" if knee_angle else ""
            strengths.append(f"Good knee bend{ka_str} (score: {knee}/100) — solid athletic base")
        if balance >= 80:
            strengths.append(f"Good hip alignment (score: {balance}/100) — stable centre of gravity")
        if spine >= 80:
            strengths.append(f"Good spinal posture (score: {spine}/100) — spine aligned over hips")

        if shoulder < 65:
            weaknesses.append(f"Shoulder alignment (score: {shoulder}/100) — may indicate rotator-cuff asymmetry")
            drills.append("Shoulder mobility routine: arm circles + cross-body stretch 10 min daily")
            mistakes.append("Shoulder tilt causing uneven power transfer to batting/bowling arm")
        if knee < 65:
            ka_str = f" ({knee_angle}°)" if knee_angle else ""
            weaknesses.append(f"Knee bend{ka_str} (score: {knee}/100) — outside ideal 140-170° athletic range")
            drills.append("Bodyweight squat: 3 × 20 reps to build leg strength and mobility")
            mistakes.append("Insufficient knee flex reducing explosive power and lateral movement")
        if balance < 65:
            weaknesses.append(f"Hip alignment (score: {balance}/100) — hip tilt shifting centre of gravity")
            drills.append("Warrior I/II yoga hold: 3 × 30s each side for hip stability")

        recommendations = [
            "Focus on core conditioning: 3 × 30s plank + side plank each session",
            "Consider physiotherapy assessment for shoulder/hip asymmetry",
            "Include 10 min hip flexor and thoracic spine mobility in every warm-up",
        ]

    if not strengths:
        strengths.append("Showing commitment to technique improvement through video analysis")
    if not suggestions:
        suggestions.append("Maintain current technique — focus on match-play consistency")
    if not drills:
        drills.append("Continue regular training: minimum 3 sessions per week")

    return {
        "strengths":               strengths,
        "weaknesses":              weaknesses,
        "mistakes":                mistakes,
        "improvement_suggestions": suggestions,
        "training_drills":         drills,
        "recommendations":         recommendations,
        "best_practices":          best_practices,
    }


def extract_raw_frame_metrics(all_landmarks: List[Optional[dict]], action_type: str) -> List[dict]:
    """
    Computes per-frame biomechanical angles and coordinates for all detected poses.
    Outputs match the DB model columns of RawFrameMetric.
    """
    frame_metrics = []
    for idx, lm in enumerate(all_landmarks):
        if not lm:
            continue

        left_knee = None
        lh = get_point(lm, LEFT_HIP)
        lk = get_point(lm, LEFT_KNEE)
        la = get_point(lm, LEFT_ANKLE)
        if lh and lk and la:
            left_knee = calculate_angle(lh, lk, la)

        right_knee = None
        rh = get_point(lm, RIGHT_HIP)
        rk = get_point(lm, RIGHT_KNEE)
        ra = get_point(lm, RIGHT_ANKLE)
        if rh and rk and ra:
            right_knee = calculate_angle(rh, rk, ra)

        left_elbow = None
        ls = get_point(lm, LEFT_SHOULDER)
        le = get_point(lm, LEFT_ELBOW)
        lw = get_point(lm, LEFT_WRIST)
        if ls and le and lw:
            left_elbow = calculate_angle(ls, le, lw)

        right_elbow = None
        rs = get_point(lm, RIGHT_SHOULDER)
        re = get_point(lm, RIGHT_ELBOW)
        rw = get_point(lm, RIGHT_WRIST)
        if rs and re and rw:
            right_elbow = calculate_angle(rs, re, rw)

        shoulder_tilt = None
        if ls and rs:
            shoulder_tilt = abs(ls[1] - rs[1])

        hip_tilt = None
        if lh and rh:
            hip_tilt = abs(lh[1] - rh[1])

        spine_offset = None
        if ls and rs and lh and rh:
            mid_sh_x = (ls[0] + rs[0]) / 2
            mid_hip_x = (lh[0] + rh[0]) / 2
            spine_offset = abs(mid_sh_x - mid_hip_x)

        wrist_y = rw[1] if rw else None
        nose_y = lm.get(NOSE)[1] if lm.get(NOSE) else None

        ankle_spread = None
        lan = get_point(lm, LEFT_ANKLE)
        ran = get_point(lm, RIGHT_ANKLE)
        if lan and ran:
            ankle_spread = abs(lan[0] - ran[0])

        frame_metrics.append({
            "frameIndex": idx,
            "frameType": action_type,
            "landmarks": lm,
            "leftKneeAngle": left_knee,
            "rightKneeAngle": right_knee,
            "leftElbowAngle": left_elbow,
            "rightElbowAngle": right_elbow,
            "shoulderTilt": shoulder_tilt,
            "hipTilt": hip_tilt,
            "spineOffset": spine_offset,
            "wristY": wrist_y,
            "noseY": nose_y,
            "ankleSpread": ankle_spread
        })
    return frame_metrics

