import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import Tuple, Dict, Any

class ContentValidator:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.cricket_classes = [0, 32, 38] # person, tie (sometimes ball), baseball bat (bat), sports ball (32)
        # 0: person, 32: sports ball, 34: baseball bat (often used for cricket bat in COCO)
    
    def validate_video(self, video_path: str, expected_type: str) -> Tuple[bool, str]:
        """
        Validates if the video contains cricket content matching the expected type.
        Returns (is_valid, error_message).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video file."
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 10:
            cap.release()
            return False, "Video is too short for analysis."
            
        # Sample frames (start, middle, end)
        sample_indices = [int(frame_count * 0.1), int(frame_count * 0.5), int(frame_count * 0.9)]
        detections = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            
            results = self.model(frame, verbose=False)
            frame_detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.3:
                        frame_detections.append(cls)
            detections.append(frame_detections)
            
        cap.release()
        
        if not detections:
            return False, "Could not process video frames."
            
        # Analysis logic
        has_person = any(0 in d for d in detections)
        has_bat = any(34 in d for d in detections) # 34 is baseball bat
        has_ball = any(32 in d for d in detections) # 32 is sports ball
        
        if not has_person:
            return False, "Wrong video uploaded. No player detected."
            
        if expected_type == 'batting' and not has_bat:
            # Sometimes YOLO misses the bat, but let's be strict as requested
            # or check if Gemini can confirm
            return False, "Wrong video uploaded. Please upload a cricket batting clip (no bat detected)."
            
        if expected_type == 'bowling' and not (has_ball or has_person):
            # Bowling is harder to detect ball, so we rely more on pose later, 
            # but for validation, we at least need a person.
            pass
            
        # Basic check for "gaming" - check for UI elements or unnatural color distributions
        # For now, we assume YOLO detecting a real person is a good start.
        
        return True, ""

    def is_blurry(self, frame, threshold=100):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold
