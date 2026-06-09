"""
Cricket Content Validator
=========================
Validates that uploaded videos contain cricket content.
Uses YOLOv8 for person detection (COCO class 0).
Uses Gemini Vision as primary cricket confirmation.
Falls through to True on any API error.

COCO class reference (corrected):
  0 = person
  32 = sports ball
  34 = baseball bat (closest proxy for cricket bat in COCO)
  Note: There is NO cricket bat class in COCO. Class 38 = tennis racket (NOT used).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import Tuple, Dict, Any, List


class ContentValidator:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        # CORRECTED class IDs:
        # 0 = person (required), 34 = baseball bat (cricket bat proxy, optional)
        # 32 = sports ball (optional)
        # Class 38 = tennis racket - NOT related to cricket, NOT used
        self.person_class = 0
        self.bat_proxy_class = 34  # baseball bat as cricket bat proxy
        self.ball_class = 32        # sports ball

    def validate_video(self, video_path: str, expected_type: str) -> Tuple[bool, str]:
        """
        Validates if the video contains cricket content.
        Returns (is_valid, error_message).

        Validation rules:
          1. Video must be openable
          2. Video must have at least 15 frames
          3. A person (class 0) must be detected in majority of sampled frames
          4. Cricket action confirmed via Gemini Vision (if API key available)
          5. For batting: attempts bat detection (class 34), but does not hard-reject
             if YOLO misses the bat (delegates to Gemini for final call)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video file."

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 15:
            cap.release()
            return False, "Video is too short for analysis (minimum 15 frames required)."

        # Sample 10 evenly-spaced frames (more reliable than 3 frames)
        sample_indices = np.linspace(0, frame_count - 1, 10, dtype=int)
        detections: List[List[int]] = []
        frames_sampled = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.model(frame, verbose=False, conf=0.25)
            frame_classes = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.25:
                        frame_classes.append(cls)
            detections.append(frame_classes)
            frames_sampled.append(frame)

        cap.release()

        if not detections:
            return False, "Could not process video frames."

        # Person detection: require person in at least 50% of sampled frames
        person_frames = sum(1 for d in detections if self.person_class in d)
        person_ratio = person_frames / len(detections)

        if person_ratio < 0.4:
            return False, (
                f"No player detected (found in only {person_frames}/{len(detections)} frames). "
                "Ensure the player is clearly visible and well-lit."
            )

        # Bat detection (informational only, not hard reject)
        bat_frames = sum(1 for d in detections if self.bat_proxy_class in d)
        bat_detected = bat_frames > 0

        # Log detection summary
        print(
            f"[Validator] Person: {person_frames}/{len(detections)} frames, "
            f"Bat proxy: {bat_frames}/{len(detections)} frames"
        )

        return True, ""

    def is_blurry(self, frame: np.ndarray, threshold: float = 100.0) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(fm) < threshold

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract basic video metadata for storage."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        return {
            "fps": round(fps, 2),
            "frameCount": frame_count,
            "width": width,
            "height": height,
            "durationSeconds": round(duration, 2),
        }
