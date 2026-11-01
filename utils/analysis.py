"""
CrickBuddy AI Service — shared vision primitives and report generation
======================================================================
This module holds the pieces every analysis type shares:

  * model singletons (YOLOv8n person/ball detector, MediaPipe Pose)
  * geometry helpers (angles, scoring curves, aspect-correct landmark access)
  * single-frame pose inference (YOLO crop → MediaPipe → re-projection)
  * coaching report generation (Gemini narration with a rule-based fallback)

The batting / bowling / posture pipelines themselves live in utils/pipeline.py.

Coordinates
-----------
Landmarks are stored as MediaPipe-normalised [x, y, z, visibility] plus an
"_aspect" entry (frame width / height). get_point() multiplies x by the aspect
so every distance and angle is measured in true "frame height" units. Without
this, portrait phone videos squashed to a fixed size distort every joint angle.
"""

import json
import os
import re
from typing import Optional, List, Tuple, Dict

import numpy as np

# ─── MediaPipe Landmark IDs (BlazePose 33-point model) ───────────────────────
NOSE = 0
LEFT_SHOULDER = 11;  RIGHT_SHOULDER = 12
LEFT_ELBOW = 13;     RIGHT_ELBOW = 14
LEFT_WRIST = 15;     RIGHT_WRIST = 16
LEFT_HIP = 23;       RIGHT_HIP = 24
LEFT_KNEE = 25;      RIGHT_KNEE = 26
LEFT_ANKLE = 27;     RIGHT_ANKLE = 28

# COCO class ids used with YOLOv8
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
COCO_BASEBALL_BAT = 34   # closest COCO proxy for a cricket bat

# ─── Global model singletons (loaded once, reused across ALL requests) ────────
_yolo_model = None
_yolo_pose_model = None
_mp_pose_c1 = None
_mp_pose_c2 = None


def _load_yolo(env_var: str, default: str):
    from ultralytics import YOLO
    weight = os.getenv(env_var, default)
    if not os.path.isabs(weight):
        # Resolve against the service folder, not the launch directory, so the
        # weights are downloaded once and found on every start.
        service_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weight = os.path.join(service_dir, weight)
    return YOLO(weight)


def _get_yolo():
    """Load YOLOv8n once; return None if unavailable (non-fatal)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            _yolo_model = _load_yolo("YOLO_WEIGHTS_PATH", "yolov8n.pt")
            print("[Info] YOLOv8 model loaded")
        except Exception as e:
            print(f"[Warn] YOLOv8 load failed (will skip crop): {e}")
            _yolo_model = False
    return _yolo_model if _yolo_model else None


def _get_yolo_pose():
    """YOLOv8n-pose: coarse keypoints for every person in one pass (used to find the bowler)."""
    global _yolo_pose_model
    if _yolo_pose_model is None:
        try:
            _yolo_pose_model = _load_yolo("YOLO_POSE_WEIGHTS_PATH", "yolov8n-pose.pt")
            print("[Info] YOLOv8-pose model loaded")
        except Exception as e:
            print(f"[Warn] YOLOv8-pose load failed: {e}")
            _yolo_pose_model = False
    return _yolo_pose_model if _yolo_pose_model else None


def _get_mp_pose(complexity: int = 1):
    global _mp_pose_c1, _mp_pose_c2
    try:
        import mediapipe.solutions.pose as mp_pose
    except ModuleNotFoundError:
        try:
            import mediapipe.python.solutions.pose as mp_pose
        except ModuleNotFoundError as e:
            print(f"[Error] Critical MediaPipe import failed: {e}")
            raise e

    if complexity == 1:
        if _mp_pose_c1 is None:
            _mp_pose_c1 = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.30,
            )
            print("[Info] MediaPipe Pose (complexity=1) singleton ready")
        return _mp_pose_c1
    else:
        if _mp_pose_c2 is None:
            _mp_pose_c2 = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.25,
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
            yolo(dummy, verbose=False, conf=0.99, imgsz=320)
            print("[Info] YOLO warm-up done")
    except Exception as e:
        print(f"[Warn] YOLO warm-up error (non-fatal): {e}")

    try:
        pose_model = _get_yolo_pose()
        if pose_model:
            pose_model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, conf=0.99, imgsz=320)
    except Exception as e:
        print(f"[Warn] YOLO-pose warm-up error (non-fatal): {e}")

    try:
        pose = _get_mp_pose(1)
        pose.process(np.zeros((480, 640, 3), dtype=np.uint8))
        print("[Info] MediaPipe Pose (c=1) warm-up done")
    except Exception as e:
        print(f"[Warn] MediaPipe Pose warm-up error (non-fatal): {e}")

    print("[Ready] All models ready — service is hot")


# ─── Basic helpers ────────────────────────────────────────────────────────────

def clamp_score(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """Angle ABC in degrees (0–180)."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(float(angle), 1)


def score_angle(angle: float, ideal_min: float, ideal_max: float) -> int:
    """Score an angle (degrees) by deviation from an ideal range. 98 at centre, degrades outward."""
    mid = (ideal_min + ideal_max) / 2
    half_range = max((ideal_max - ideal_min) / 2, 1)
    deviation = abs(angle - mid)
    if deviation <= half_range:
        return clamp_score(98 - (deviation / half_range) * 12, 0, 100)
    return clamp_score(86 - (deviation - half_range) * 1.8, 0, 100)


def range_score(value: Optional[float], lo: float, hi: float, falloff: float) -> Optional[int]:
    """
    Unit-agnostic version of score_angle.
    Inside [lo, hi]: 100 at the centre down to 86 at the edges.
    Outside: drops linearly, reaching 26 one `falloff` beyond the edge, then 0.
    """
    if value is None or not np.isfinite(value):
        return None
    mid = (lo + hi) / 2
    half = max((hi - lo) / 2, 1e-9)
    dev = abs(value - mid)
    if dev <= half:
        return clamp_score(100 - 14 * dev / half)
    return clamp_score(86 - 60 * (dev - half) / falloff)


def get_point(landmarks: dict, idx: int) -> Optional[List[float]]:
    """Landmark (x, y) in aspect-correct frame-height units, or None."""
    lm = landmarks.get(idx)
    if lm is None:
        lm = landmarks.get(str(idx))   # landmarks that went through JSON
    if lm is None:
        return None
    aspect = float(landmarks.get("_aspect", 1.0) or 1.0)
    if isinstance(lm, dict):
        return [lm["x"] * aspect, lm["y"]]
    return [lm[0] * aspect, lm[1]]


# ─── Frame utilities ──────────────────────────────────────────────────────────

MAX_SIDE = 640
BLUR_THRESHOLD = 40.0


def resize_keep_aspect(frame: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    # INTER_LINEAR is ~10× faster than INTER_AREA on 1080p input and accurate enough for pose.
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)


def _is_sharp(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= threshold


def extract_frames(video_path: str, num_frames: int = 20) -> List[np.ndarray]:
    """Evenly spaced, sharpness-filtered frames across a whole video (used by train_models.py)."""
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw: List[np.ndarray] = []
        if total <= 0:
            while len(raw) < num_frames * 4:
                ret, frm = cap.read()
                if not ret or frm is None:
                    break
                raw.append(frm)
        else:
            for idx in np.linspace(0, total - 1, num=min(num_frames * 3, total), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frm = cap.read()
                if ret and frm is not None:
                    raw.append(frm)
        cap.release()
        if not raw:
            return []
        raw = [resize_keep_aspect(f) for f in raw]
        sharp = [f for f in raw if _is_sharp(f)]
        pool = sharp if len(sharp) >= max(num_frames // 3, 3) else raw
        n = min(num_frames, len(pool))
        return [pool[int(i)] for i in np.linspace(0, len(pool) - 1, num=n, dtype=int)]
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return []


# ─── Person detection + pose inference ────────────────────────────────────────

def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_people(frame: np.ndarray, conf: float = 0.25, imgsz: int = 416) -> List[Tuple[List[float], float]]:
    """All person detections as ([x1, y1, x2, y2], confidence), largest first."""
    model = _get_yolo()
    if model is None:
        return []
    try:
        results = model(frame, verbose=False, conf=conf, classes=[COCO_PERSON], imgsz=imgsz)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        people = [([float(v) for v in b.xyxy[0]], float(b.conf[0])) for b in boxes]
        return sorted(people, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]), reverse=True)
    except Exception as e:
        print(f"YOLO person detection error: {e}")
        return []


def match_box(people: List[Tuple[List[float], float]], prev_box) -> Optional[List[float]]:
    """
    The detection that continues `prev_box`: best overlap, else the nearest centre
    within about one box size. None when nothing plausible is found — the caller
    keeps the previous box rather than jumping to another player.
    """
    if not people:
        return None
    best = max(people, key=lambda p: _box_iou(p[0], prev_box))
    if _box_iou(best[0], prev_box) > 0.1:
        return best[0]
    pcx, pcy = (prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2
    size = max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1])

    def dist(p):
        b = p[0]
        return np.hypot((b[0] + b[2]) / 2 - pcx, (b[1] + b[3]) / 2 - pcy)

    near = min(people, key=dist)
    return near[0] if dist(near) < size else None


def detect_person_box(frame: np.ndarray, prev_box=None, conf: float = 0.25) -> Optional[List[float]]:
    """
    Pixel box [x1, y1, x2, y2] of the player. With a previous box the same player
    is followed; otherwise the largest confident person wins.
    """
    people = detect_people(frame, conf)
    if prev_box is not None:
        return match_box(people, prev_box)
    if not people:
        return None
    return max(people, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]) * p[1])[0]


def people_arm_heights(frame: np.ndarray, conf: float = 0.25) -> List[Tuple[List[float], float]]:
    """
    ([x1, y1, x2, y2], arm height) for every person, from one YOLOv8-pose pass.
    Arm height = highest wrist above its shoulder in torso lengths (COCO keypoints).
    """
    model = _get_yolo_pose()
    if model is None:
        return []
    try:
        r = model(frame, verbose=False, conf=conf, imgsz=640)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        kps = r.keypoints.data.cpu().numpy()          # (N, 17, 3): x, y, confidence
        out = []
        for box, k in zip(r.boxes.xyxy.cpu().numpy(), kps):
            ok = k[:, 2] > 0.3
            if not (ok[5] and ok[6] and ok[11] and ok[12]):
                continue
            torso = np.linalg.norm((k[5, :2] + k[6, :2]) / 2 - (k[11, :2] + k[12, :2]) / 2)
            if torso <= 0:
                continue
            heights = [(k[s, 1] - k[w, 1]) / torso for s, w in ((5, 9), (6, 10)) if ok[s] and ok[w]]
            if heights:
                out.append(([float(v) for v in box], float(max(heights))))
        return out
    except Exception as e:
        print(f"YOLO-pose error: {e}")
        return []


def detect_balls(frame: np.ndarray, conf: float = 0.08) -> List[Tuple[float, float, float]]:
    """Sports-ball detections as (cx, cy, size) in pixels."""
    model = _get_yolo()
    if model is None:
        return []
    try:
        results = model(frame, verbose=False, conf=conf, classes=[COCO_SPORTS_BALL], imgsz=640)
        out = []
        for b in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)))
        return out
    except Exception as e:
        print(f"YOLO ball detection error: {e}")
        return []


def pose_on_frame(frame_bgr: np.ndarray, box=None, complexity: int = 1, pad: float = 0.2) -> Optional[dict]:
    """
    MediaPipe pose for one frame. With a person box the crop is padded and the
    landmarks re-projected to full-frame normalised coordinates.
    """
    import cv2

    full_h, full_w = frame_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = 0, 0, full_w, full_h
    if box is not None:
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(full_w, int(x2 + px)), min(full_h, int(y2 + py))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            cx1, cy1, cx2, cy2 = 0, 0, full_w, full_h

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    try:
        results = _get_mp_pose(complexity).process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"MediaPipe inference error: {e}")
        return None
    if not results.pose_landmarks:
        return None

    crop_h, crop_w = crop.shape[:2]
    landmarks: dict = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        nx = (cx1 + lm.x * crop_w) / full_w
        ny = (cy1 + lm.y * crop_h) / full_h
        landmarks[idx] = [round(nx, 4), round(ny, 4), round(lm.z, 4), round(lm.visibility, 4)]
    landmarks["_aspect"] = round(full_w / full_h, 5)
    return landmarks


def _run_mediapipe_on_frame(frame_bgr: np.ndarray, complexity: int = 1) -> Optional[dict]:
    """YOLO crop → MediaPipe on a single frame; falls back to the full frame."""
    frame_bgr = resize_keep_aspect(frame_bgr)
    box = detect_person_box(frame_bgr)
    lm = pose_on_frame(frame_bgr, box, complexity) if box is not None else None
    if lm is None:
        lm = pose_on_frame(frame_bgr, None, complexity)
    return lm


def analyze_pose_from_video_frames(frames: List[np.ndarray]) -> List[Optional[dict]]:
    """Independent per-frame pose (used by train_models.py)."""
    return [_run_mediapipe_on_frame(f) for f in frames]


def analyze_pose_from_image(image_path: str) -> Optional[dict]:
    """Run YOLO→crop→MediaPipe on a single image file (complexity 2 retry)."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        return None
    for complexity in (1, 2):
        lm = _run_mediapipe_on_frame(image, complexity=complexity)
        if lm is not None:
            return lm
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


async def generate_report(metrics: dict, type_: str, faults: List[Dict] = None) -> dict:
    """
    Generate the coaching narrative via Gemini.
    The LLM receives the computed scores and only explains them; it may not
    invent or change scores. Falls back to the rule-based report on any error.
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

    fault_summary = [
        f"[{f['severity'].upper()}] {f['faultText']} ({f['metric']}={f['value']}, threshold={f['threshold']})"
        for f in (faults or [])
    ]
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    notes = {k: v for k, v in metrics.items() if k.endswith("Note") and v}

    prompt = f"""You are a professional cricket coach explaining biomechanical analysis results to a player.

IMPORTANT RULES:
1. You MUST NOT invent, change, or contradict any of the scores listed below.
2. You MUST NOT add faults that are not listed.
3. Only call something a strength if its score is 70 or higher; lists may be short or empty.
4. Reference the actual numeric values in your explanations.
5. Scores are 0-100 and come from 2D pose estimation on a single camera view.

ANALYSIS TYPE: {type_.capitalize()}

COMPUTED METRICS (DO NOT CHANGE THESE):
{json.dumps(numeric, indent=2)}

MEASUREMENT NOTES:
{json.dumps(notes, indent=2)}

DETECTED FAULTS (explain each, do not add new ones):
{chr(10).join(fault_summary) if fault_summary else 'No faults detected above threshold.'}

Respond in JSON ONLY with this exact structure:
{{
  "strengths": ["genuine technical strengths, citing score values"],
  "weaknesses": ["one entry per detected fault, citing its value"],
  "mistakes": ["technical mistakes implied by the faults — cite which metric"],
  "improvement_suggestions": ["3-6 actionable coaching cues"],
  "training_drills": ["3-5 specific drills with sets/reps targeting the weakest metrics"],
  "recommendations": ["2-4 conditioning or equipment notes"],
  "best_practices": ["3-5 habits that support this technique"]
}}"""

    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.30,
                response_mime_type="application/json",
                max_output_tokens=1200,
            ),
        )
        parsed = _parse_json_response_text(response.text or "")
        keys = ["strengths", "weaknesses", "mistakes", "improvement_suggestions",
                "training_drills", "recommendations", "best_practices"]
        return {k: [str(x) for x in (parsed.get(k) or []) if x] for k in keys}
    except Exception as e:
        print(f"Gemini report error (using rule-based fallback): {e}")
        return _generate_rule_based_report(metrics, type_, faults or [])


# Coaching content for each fault code emitted by utils/fault_detector.py.
# (weakness template, mistake, suggestion, drill) — templates may use {value}.
FAULT_COACHING = {
    # batting
    "HEAD_INSTABILITY": (
        "Head movement through the shot ({value} torso-lengths of vertical bob)",
        "Head moving off the line of the ball before contact",
        "Keep eyes level and the head still until after contact — let the body move under it",
        "Static-head shadow batting: 3 × 20 reps watching a fixed point on a wall",
    ),
    "LATE_TIMING": (
        "Front foot lands too late relative to the bat swing ({value} ms before contact)",
        "Bat coming down before the front foot is planted, so weight transfer is incomplete",
        "Trigger earlier: start the forward press as the bowler's arm comes over",
        "Drop-ball drill: 3 × 15 feeds, plant the front foot before the ball bounces",
    ),
    "EARLY_TIMING": (
        "Front foot plants long before contact ({value} ms) — the shot is waiting on the ball",
        "Committing to the front foot too early and losing rhythm into the swing",
        "Delay the stride slightly and keep the weight balanced until length is picked",
        "Throwdowns at varied lengths: 3 × 12, stride only once length is read",
    ),
    "STIFF_FRONT_KNEE": (
        "Front knee stays almost straight ({value}° at its most bent)",
        "Little knee flex in the stride, so the head cannot get over the ball",
        "Bend the front knee into the shot to bring the head over the line",
        "Lunge-and-hold front-foot drives: 3 × 10 each, hold the finish for 2 s",
    ),
    "POOR_BALANCE": (
        "Balance through the action (score {value}/100)",
        "Head falling outside the base or excessive spine lean at the key moment",
        "Finish every rep balanced — hold the final position for two seconds",
        "Single-leg balance holds: 3 × 30 s each side, then shadow reps",
    ),
    "SHORT_FOLLOW_THROUGH": (
        "Follow-through is cut short (hands rise only {value} torso-lengths after contact)",
        "Checking the swing at contact instead of letting the bat finish",
        "Let the hands finish high over the front shoulder on attacking shots",
        "Shadow follow-through: 2 × 25 reps finishing with hands above the head",
    ),
    "STANCE_WIDTH": (
        "Stride/stance width is outside the ideal range ({value}× torso length)",
        "Base too narrow for power or too wide to move freely",
        "Set feet about shoulder-width at setup and stride roughly one torso length",
        "Footwork ladder: 3 × 10 reps setting up and striding to markers",
    ),
    # bowling
    "LOW_RELEASE": (
        "Low release point (wrist {value} torso-lengths above the shoulder)",
        "Arm dropping at release, reducing height, bounce and carry",
        "Reach up tall at release — bowl 'over the top' with a high front arm pull",
        "Wall-target drill: 3 × 8 deliveries releasing above a mark set at full stretch",
    ),
    "BENT_ARM_AT_RELEASE": (
        "Bowling arm bent at release ({value}° at the elbow)",
        "Elbow not fully extended at release — check with a side-on video",
        "Keep the arm straight through the arc; get a coach to review the action",
        "Slow-motion arm circles with a straight arm: 3 × 10 before each session",
    ),
    "JERKY_ACTION": (
        "Arm rotation rhythm (smoothness score {value}/100)",
        "Arm speed rising and falling during the delivery arc instead of one smooth acceleration",
        "Build pace gradually through the crease; let the arm accelerate in one continuous arc",
        "Walk-through bowling at 50% pace: 3 × 6 balls focusing on a single smooth arm swing",
    ),
    "WRIST_DROP": (
        "Wrist behind/below the elbow at release (delta {value})",
        "Wrist collapsing at release, losing seam position",
        "Keep the wrist firm and behind the ball with fingers on top at release",
        "Seam-up catching drill: flick the ball to a partner with a vertical seam, 3 × 20",
    ),
    # posture
    "SHOULDER_TILT": (
        "Shoulders not level ({value}° tilt)",
        "Dropping one shoulder in the set-up position",
        "Set up with level shoulders and eyes",
        "Mirror set-up checks: 3 × 10 holds of 5 s",
    ),
    "KNEE_ANGLE": (
        "Knee bend outside the athletic range ({value}°)",
        "Either too upright or too crouched in the ready position",
        "Soft knees: roughly 140–170° at the knee in the ready position",
        "Split-step ready-position drill: 3 × 12 reps",
    ),
    "SPINE_LEAN": (
        "Spine leaning away from vertical ({value}°)",
        "Upper body falling to one side in the set-up",
        "Stack the head and shoulders over the hips",
        "Wall posture holds: 3 × 30 s",
    ),
}


def _generate_rule_based_report(metrics: dict, type_: str, faults: List[Dict] = None) -> dict:
    """Metric-driven fallback report — every statement references a measured value."""
    strengths: List[str] = []
    weaknesses: List[str] = []
    mistakes: List[str] = []
    suggestions: List[str] = []
    drills: List[str] = []
    recommendations: List[str] = []
    best_practices = [
        "Film from side-on at hip height with the whole body in frame",
        "Review your analysis footage weekly to track technique changes",
        "Keep a training journal — log scores and one key cue per session",
    ]

    for f in faults or []:
        coaching = FAULT_COACHING.get(f.get("faultCode"))
        if not coaching:
            weaknesses.append(f.get("faultText", ""))
            continue
        weak, mistake, cue, drill = coaching
        weaknesses.append(weak.format(value=f.get("value")))
        mistakes.append(mistake)
        suggestions.append(cue)
        drills.append(drill)

    def praise(key: str, label: str, note_key: str):
        """Strength line for a score ≥ 75, quoting the engine's own measurement note."""
        v = metrics.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 75:
            note = metrics.get(note_key)
            strengths.append(f"{label} ({v}/100)" + (f" — {note}" if note else ""))

    if type_ == "batting":
        praise("headStabilityScore", "Head stability", "headStabilityNote")
        praise("timingScore", "Timing", "timingNote")
        praise("balanceScore", "Balance at contact", "balanceNote")
        praise("followThroughScore", "Follow-through", "followThroughNote")
        praise("stanceScore", "Set-up stance", None)
        praise("strideScore", "Stride", "strideNote")
        recommendations = [
            "Use a bat you can pick up and swing without the top hand dominating",
            "Include hip-mobility work so you can get a full stride to the pitch of the ball",
        ]
    elif type_ == "bowling":
        praise("armSmoothnessScore", "Arm action", "armSmoothnessNote")
        praise("releasePointScore", "Release point", "releasePointNote")
        praise("wristPositionScore", "Wrist position", "wristPositionNote")
        praise("balanceScore", "Balance at release", "balanceNote")
        speed = metrics.get("estimatedBallSpeed")
        if speed:
            recommendations.append(
                f"Estimated pace {speed} km/h ({metrics.get('speedClassification')}) — measured from a single "
                "camera, so treat it as approximate"
            )
        else:
            recommendations.append("Ball was not tracked — film side-on with the ball visible after release for a pace estimate")
        recommendations.append("Allow 48 h recovery between high-intensity bowling sessions")
    elif type_ == "posture":
        praise("shoulderAlignmentScore", "Shoulders", "shoulderAlignmentNote")
        praise("kneeBendScore", "Knee bend", "kneeBendNote")
        praise("balanceScore", "Hips", "balanceNote")
        praise("spinePosScore", "Spine", "spinePosNote")
        recommendations = [
            "Core conditioning: 3 × 30 s plank and side plank each session",
            "Add 10 min hip-flexor and thoracic mobility to every warm-up",
        ]

    if not suggestions:
        suggestions.append("No faults above threshold — keep repeating this technique under match-like pressure")
    if not drills:
        drills.append("Maintain 3 quality sessions per week with one filmed for review")

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
    """Per-frame joint angles and positions (columns of the RawFrameMetric table)."""
    frame_metrics = []
    for idx, lm in enumerate(all_landmarks):
        if not lm:
            continue

        def ang(a, b, c):
            pa, pb, pc = get_point(lm, a), get_point(lm, b), get_point(lm, c)
            return calculate_angle(pa, pb, pc) if pa and pb and pc else None

        ls, rs = get_point(lm, LEFT_SHOULDER), get_point(lm, RIGHT_SHOULDER)
        lh, rh = get_point(lm, LEFT_HIP), get_point(lm, RIGHT_HIP)
        la, ra = get_point(lm, LEFT_ANKLE), get_point(lm, RIGHT_ANKLE)
        rw, nose = get_point(lm, RIGHT_WRIST), get_point(lm, NOSE)

        spine_offset = None
        if ls and rs and lh and rh:
            spine_offset = round(abs((ls[0] + rs[0]) / 2 - (lh[0] + rh[0]) / 2), 5)

        frame_metrics.append({
            "frameIndex": idx,
            "frameType": action_type,
            "landmarks": lm,
            "leftKneeAngle": ang(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            "rightKneeAngle": ang(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            "leftElbowAngle": ang(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            "rightElbowAngle": ang(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            "shoulderTilt": round(abs(ls[1] - rs[1]), 5) if ls and rs else None,
            "hipTilt": round(abs(lh[1] - rh[1]), 5) if lh and rh else None,
            "spineOffset": spine_offset,
            "wristY": round(rw[1], 5) if rw else None,
            "noseY": round(nose[1], 5) if nose else None,
            "ankleSpread": round(abs(la[0] - ra[0]), 5) if la and ra else None,
        })
    return frame_metrics
