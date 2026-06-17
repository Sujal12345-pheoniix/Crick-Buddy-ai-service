import torch
import torch.nn as nn
import numpy as np
import cv2
import os

class VideoValidatorNet(nn.Module):
    """
    Lightweight video validation network.
    Uses temporal features: mean optical flow magnitudes, human confidence scores,
    and frame-to-frame difference variances.
    """
    def __init__(self, input_dim=5, hidden_dim=16):
        super(VideoValidatorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2) # [Non-Cricket, Cricket]
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def extract_validation_features(frames, yolo_model=None) -> np.ndarray:
    """
    Extracts a feature vector from video frames:
    [avg_motion, max_motion, human_present_ratio, pose_confidence_avg, motion_variance]
    """
    if not frames:
        return np.zeros((1, 5), dtype=np.float32)
        
    # 1. Motion features (dense optical flow)
    motion_mags = []
    for i in range(len(frames) - 1):
        try:
            prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mags.append(float(np.mean(mag)))
        except Exception:
            motion_mags.append(0.0)
            
    avg_motion = np.mean(motion_mags) if motion_mags else 0.0
    max_motion = np.max(motion_mags) if motion_mags else 0.0
    motion_var = np.var(motion_mags) if motion_mags else 0.0
    
    # 2. YOLO Human Detection & Confidence
    human_present_count = 0
    total_conf = 0.0
    
    if yolo_model is not None:
        for frame in frames[:5]: # Sample first 5 frames for speed
            try:
                results = yolo_model(frame, verbose=False, conf=0.15)
                boxes = results[0].boxes
                has_human = False
                for box in boxes:
                    if int(box.cls[0]) == 0: # Person
                        has_human = True
                        total_conf += float(box.conf[0])
                        break
                if has_human:
                    human_present_count += 1
            except Exception:
                pass
                
    human_ratio = human_present_count / min(len(frames), 5) if frames else 0.0
    avg_conf = total_conf / max(human_present_count, 1)
    
    return np.array([[avg_motion, max_motion, human_ratio, avg_conf, motion_var]], dtype=np.float32)

class VideoValidator:
    def __init__(self, model_path="video_validator.pth", yolo_model=None):
        self.yolo_model = yolo_model
        self.model = VideoValidatorNet()
        # Find absolute path if loaded in FastAPI
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_model_path = os.path.join(base_dir, model_path)
        if os.path.exists(abs_model_path):
            try:
                self.model.load_state_dict(torch.load(abs_model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"[Info] Video Validator model loaded from {abs_model_path}")
            except Exception as e:
                print(f"[Warn] Failed to load Video Validator: {e}")
        else:
            print(f"[Info] Video Validator weights not found at {abs_model_path}. Using default initialization.")
            
    def validate(self, frames) -> tuple[bool, str]:
        """
        Classifies the video as Cricket or Non-Cricket.
        Returns (is_valid, message)
        """
        if not frames or len(frames) < 3:
            return False, "ERR_INVALID_VIDEO: Could not extract enough frames from the video."
            
        features = extract_validation_features(frames, self.yolo_model)
        x = torch.tensor(features)
        
        with torch.no_grad():
            outputs = self.model(x)
            prediction = torch.argmax(outputs, dim=1).item()
            
        # If prediction is 1, it's Cricket. Otherwise Non-Cricket.
        if prediction == 1:
            return True, ""
        else:
            # Check if there is zero motion or no human to provide a detailed message
            if features[0, 2] < 0.2:
                return False, "ERR_NO_HUMAN_DETECTED: No player detected in the video."
            if features[0, 0] < 0.1:
                return False, "ERR_NOT_CRICKET_ACTION: No athletic action or movement detected."
            return False, "ERR_NON_CRICKET: The uploaded video does not contain valid cricket movements."
