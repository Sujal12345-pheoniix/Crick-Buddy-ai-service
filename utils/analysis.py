"""
Cricket Analysis Utilities
Shared helper functions for MediaPipe pose analysis, OpenCV frame extraction,
angle calculations, and report generation.
"""

import json
import math
import os
import random
import re
import numpy as np
from typing import Optional, List


def clamp_score(value: float, low: int = 0, high: int = 100) -> int:
    """Clamp a numeric score to an integer range."""
    return int(max(low, min(high, round(value))))


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """Calculate angle at point B given three points A, B, C."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(float(angle), 1)


def score_angle(angle: float, ideal_min: float, ideal_max: float) -> int:
    """Score an angle based on ideal range. Returns 0-100."""
    mid = (ideal_min + ideal_max) / 2
    half_range = max((ideal_max - ideal_min) / 2, 1)
    deviation = abs(angle - mid)

    if deviation <= half_range:
        # Inside ideal range: smoothly scale 86-98 by distance from center.
        ratio = deviation / half_range
        return clamp_score(98 - ratio * 12, 0, 100)

    # Outside ideal range: decay from 86 as deviation grows.
    outside = deviation - half_range
    return clamp_score(86 - outside * 1.8, 0, 100)


def extract_frames(video_path: str, num_frames: int = 30) -> List[np.ndarray]:
    """Extract evenly spaced frames from a video file using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Many codecs report 0 frames until decoded; read sequentially and sample.
        if total_frames <= 0:
            raw: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                raw.append(frame)
            cap.release()
            if not raw:
                return []
            n = min(num_frames, len(raw))
            indices = np.linspace(0, len(raw) - 1, num=n, dtype=int)
            return [raw[int(i)] for i in indices]

        indices = np.linspace(0, max(total_frames - 1, 0), num=min(num_frames, total_frames), dtype=int)
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()

        if not frames:
            cap2 = cv2.VideoCapture(video_path)
            raw = []
            while len(raw) < num_frames * 4:
                ret, frame = cap2.read()
                if not ret or frame is None:
                    break
                raw.append(frame)
            cap2.release()
            if not raw:
                return []
            n = min(num_frames, len(raw))
            idx2 = np.linspace(0, len(raw) - 1, num=n, dtype=int)
            return [raw[int(i)] for i in idx2]

        return frames
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return []


def analyze_pose_from_image(image_path: str) -> Optional[dict]:
    """Run MediaPipe Pose on a single image and return landmark dict."""
    try:
        import cv2
        import mediapipe as mp
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Check for cricket context (basic heuristic or detector)
        if not is_cricket_content(image):
            return {"error": "wrong_content"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4
        ) as pose:
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = {}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 4)
                }
            return landmarks
    except Exception as e:
        print(f"MediaPipe pose error: {e}")
        return None


def is_cricket_content(image: np.ndarray) -> bool:
    """
    Validate if the image/frame contains cricket-related content.
    Requires at least a person detected (cricket bat/ball detection is bonus).
    Returns True = likely cricket, False = clearly non-cricket.
    """
    try:
        from ultralytics import YOLO
        
        if not hasattr(is_cricket_content, "_model"):
            yolo_path = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt")
            is_cricket_content._model = YOLO(yolo_path)
        
        results = is_cricket_content._model(image, verbose=False)
        
        found_person = False
        found_cricket_item = False 
        
        # COCO classes: 0: person, 32: sports ball, 34: baseball bat
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < 0.25:  # Lower threshold to catch more cases
                    continue
                
                if cls_id == 0: 
                    found_person = True
                if cls_id in [32, 34]:  # sports ball or baseball bat (close to cricket bat)
                    found_cricket_item = True
        
        # Primary requirement: must have a person in frame.
        # If a person is found + bat/ball detected → definitely cricket.
        # If only person found → likely cricket (YOLO often misses cricket-specific gear).
        # If NO person → reject.
        if not found_person:
            return False
        
        # If we detect a sports ball or bat with a person, that's very strong signal.
        # If only a person, still allow (cricket bats are often classified as other objects).
        return True
    except Exception as e:
        print(f"Cricket validation error: {e}")
        return True  # Fallback: do not block valid users due to model errors


def analyze_pose_from_video_frames(frames: List[np.ndarray]) -> List[Optional[dict]]:
    """Run MediaPipe Pose on multiple frames and return list of landmark data."""
    try:
        import mediapipe as mp
        import cv2
        
        all_landmarks = []
        
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        ) as pose:
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    landmarks = {}
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        landmarks[idx] = [lm.x, lm.y, lm.z, lm.visibility]
                    all_landmarks.append(landmarks)
                else:
                    all_landmarks.append(None)
        
        return all_landmarks
    except Exception as e:
        print(f"MediaPipe video analysis error: {e}")
        return []


def estimate_ball_speed(frames: List[np.ndarray]) -> Optional[float]:
    """
    Estimate ball speed in km/h using lightweight optical-flow motion.
    Optional YOLO refinement can be enabled with ENABLE_YOLO_SPEED=true.
    """
    if len(frames) < 2:
        return None

    try:
        import cv2

        motion = []
        for idx in range(len(frames) - 1):
            prev = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frames[idx + 1], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev,
                nxt,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            valid = mag[np.isfinite(mag)]
            if valid.size:
                motion.append(float(np.percentile(valid, 92)))

        if not motion:
            return None

        # Convert robust motion proxy to cricket-speed range.
        motion_score = float(np.median(motion))
        speed_kmh = 80 + min(80.0, motion_score * 18.0)

        # Optional refinement: YOLO tracking if explicitly enabled.
        if str(os.getenv("ENABLE_YOLO_SPEED", "false")).lower() == "true":
            try:
                from ultralytics import YOLO

                weight_path = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt")
                model = YOLO(weight_path)
                points = []

                for frame in frames:
                    results = model(frame, verbose=False)
                    best = None
                    best_conf = 0.0
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id in [32, 0] and conf > best_conf:
                            x1, y1, x2, y2 = box.xyxy[0]
                            best_conf = conf
                            best = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
                    if best:
                        points.append(best)

                if len(points) >= 2:
                    first = points[0]
                    last = points[-1]
                    pixel_dist = math.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)
                    yolo_speed = (pixel_dist / max(len(points), 1)) * 30 * 0.05 * 3.6
                    # Blend stable optical-flow estimate with detector-based estimate.
                    speed_kmh = (speed_kmh * 0.65) + (yolo_speed * 0.35)
            except Exception as yolo_err:
                print(f"YOLO speed refinement skipped: {yolo_err}")

        return round(float(max(80.0, min(160.0, speed_kmh))), 1)
    except Exception as e:
        print(f"Ball speed estimation error: {e}")
        return None


# ─── MediaPipe Landmark IDs (BlazePose) ────────────────────────────────────
NOSE = 0
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13;    RIGHT_ELBOW = 14
LEFT_WRIST = 15;    RIGHT_WRIST = 16
LEFT_HIP = 23;      RIGHT_HIP = 24
LEFT_KNEE = 25;     RIGHT_KNEE = 26
LEFT_ANKLE = 27;    RIGHT_ANKLE = 28


def get_point(landmarks: dict, idx: int) -> Optional[List[float]]:
    lm = landmarks.get(idx)
    if lm is None:
        return None
    if isinstance(lm, dict):
        return [lm['x'], lm['y']]
    return [lm[0], lm[1]]


def _parse_json_response_text(text: str) -> dict:
    """Parse model output that should be JSON; tolerate markdown fences."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


async def generate_report(metrics: dict, type_: str) -> dict:
    """Generate high-quality AI coaching feedback using Gemini API (google-genai SDK)."""
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key:
        return _generate_rule_based_report(metrics, type_)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("google-genai is not installed; install with: pip install google-genai")
        return _generate_rule_based_report(metrics, type_)

    client = genai.Client(api_key=api_key)

    prompt = f"""You are a world-class professional cricket coach and biomechanics expert. 
Analyze these player metrics extracted from pose estimation and provide a deep, accurate technical report.

Analysis Type: {type_.capitalize()}
Detailed Metrics: {json.dumps(metrics, default=str)}

Focus on:
1. Identifying subtle technical flaws in {type_} technique.
2. Providing actionable, elite-level coaching cues.
3. Suggesting drills that directly address the numeric deviations in the metrics.

Respond in JSON ONLY with this exact structure:
{{
  "strengths": ["3-5 high-quality technical strengths"],
  "weaknesses": ["2-4 specific technical areas for improvement"],
  "mistakes": ["Detailed explanation of 3-4 likely technical mistakes based on the metrics"],
  "improvement_suggestions": ["5-7 specific technical and tactical tips for immediate improvement"],
  "training_drills": ["4-6 high-intensity drills with specific sets, reps, and focus points"],
  "recommendations": ["3-5 personalized notes on equipment, physical conditioning, or match strategy"],
  "best_practices": ["4-6 professional habits for elite performance"]
}}

Use professional cricket terminology (e.g., 'falling over the off-side', 'closed face', 'unstable base', 'loading phase'). 
Ensure the feedback feels personalized to the numbers provided. Do not use generic advice."""

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
        # Normalize keys expected by the backend
        return {
            "strengths": parsed.get("strengths") or [],
            "weaknesses": parsed.get("weaknesses") or [],
            "mistakes": parsed.get("mistakes") or [],
            "improvement_suggestions": parsed.get("improvement_suggestions") or parsed.get("improvementSuggestions") or [],
            "training_drills": parsed.get("training_drills") or parsed.get("trainingDrills") or [],
            "recommendations": parsed.get("recommendations") or [],
            "best_practices": parsed.get("best_practices") or parsed.get("bestPractices") or [],
        }
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return _generate_rule_based_report(metrics, type_)


def _generate_rule_based_report(metrics: dict, type_: str) -> dict:
    """Fallback rule-based report generation."""
    strengths = []
    weaknesses = []
    suggestions = []
    drills = []

    if type_ == 'batting':
        m = metrics
        if m.get('stanceScore', 0) >= 80:
            strengths.append("Excellent batting stance — solid foundation for shot-making")
        else:
            weaknesses.append("Batting stance needs work — feet positioning inconsistent")
            suggestions.append("Practice the 'stance mirror drill' — stand before a mirror and adjust feet to shoulder-width")
            drills.append("10 min daily stance drill: shadow batting with coach checkpoints")

        if m.get('batSwingAngle') and 30 <= m['batSwingAngle'] <= 60:
            strengths.append(f"Good bat swing angle ({m['batSwingAngle']}°) — ideal for driving shots")
        elif m.get('batSwingAngle'):
            weaknesses.append(f"Bat swing angle ({m['batSwingAngle']}°) is off-ideal — affects shot power")
            suggestions.append("Work on vertical bat shots to correct the swing plane")
            drills.append("20 throwdown sessions focusing on straight drive and cover drive")

        if m.get('timingScore', 0) >= 75:
            strengths.append("Good ball-hitting timing — transfer of weight is effective")
        else:
            weaknesses.append("Timing issues detected — weight transfer needs improvement")
            suggestions.append("Use a hanging ball trainer to improve eye-hand coordination")
            drills.append("Hanging ball drill: 30 minutes daily")

        if m.get('followThroughScore', 0) >= 75:
            strengths.append("Complete follow-through — maximizing shot power correctly")
        else:
            weaknesses.append("Incomplete follow-through — losing power and control")
            drills.append("Shadow batting with full follow-through emphasis — 100 reps")

    elif type_ == 'bowling':
        m = metrics
        speed = m.get('estimatedBallSpeed', 0)
        if speed >= 130:
            strengths.append(f"Excellent pace — bowling at {speed} km/h, above average club level")
        elif speed >= 100:
            strengths.append(f"Good pace at {speed} km/h — consistent with medium-fast category")
        else:
            weaknesses.append(f"Ball speed ({speed} km/h) is below target — need to improve run-up momentum")
            suggestions.append("Add 3 more approach steps to build momentum before delivery")
            drills.append("Run-up acceleration drill: 20 deliveries focusing on full run-up speed")

        if m.get('wristPositionScore', 0) >= 80:
            strengths.append("Strong wrist position at release — good seam stability")
        else:
            weaknesses.append("Wrist position at release needs work — may affect seam position")
            drills.append("Wrist strengthening exercises: 3 sets x 15 reps with resistance band")

        if m.get('balanceScore', 0) >= 75:
            strengths.append("Good body balance through delivery stride")
        else:
            weaknesses.append("Balance issues in delivery stride — risk of injury and inconsistency")
            suggestions.append("Strengthen core muscles to improve balance")
            drills.append("Single-leg balance drill: 3 x 30 seconds each leg")

    elif type_ == 'posture':
        m = metrics
        if m.get('shoulderAlignmentScore', 0) >= 80:
            strengths.append("Good shoulder alignment — balanced upper body positioning")
        else:
            weaknesses.append("Shoulder imbalance detected — may indicate muscle weakness")
            suggestions.append("Work on shoulder mobility exercises before practice sessions")
            drills.append("Shoulder mobility routine: 10 min daily (arm circles, cross-body stretch)")

        if m.get('kneeBendScore', 0) >= 75:
            strengths.append("Correct knee bend angle — good athletic position")
        else:
            weaknesses.append(f"Knee bend angle ({m.get('kneeBendAngle', 0)}°) needs adjustment")
            drills.append("Squat training: 3 x 20 bodyweight squats to strengthen legs")

        if m.get('balanceScore', 0) >= 75:
            strengths.append("Good posture balance — solid centre of gravity")
        else:
            weaknesses.append("Posture imbalance detected — may affect performance")
            drills.append("Yoga poses: Warrior I and II, 3 x 30 seconds each side")

    if not suggestions:
        suggestions.append("Maintain your current technique — focus on consistency")
    if not drills:
        drills.append("Continue regular training schedule — 3 sessions per week minimum")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "mistakes": ["High center of gravity during impact", "Static footwork", "Limited follow-through extension"],
        "improvement_suggestions": suggestions,
        "training_drills": drills,
        "recommendations": ["Consider a professional bat weighing between 1.1kg - 1.2kg for better swing balance", "Invest in high-quality batting gloves with extra finger protection"],
        "best_practices": ["Perform 15 min of dynamic mobility drills before every session", "Maintain a hydration level of at least 500ml per hour of play", "Review your analysis footage weekly to track muscle memory changes"]
    }
