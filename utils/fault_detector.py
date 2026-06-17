import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Any

class FaultMLP(nn.Module):
    """
    MLP to predict binary indicators for 5 faults:
    [poor balance, head movement, stance issue, weak follow-through, poor timing]
    Input features:
    [headStabilityVariance, avgHipTilt, avgSpineOffset, rangeOfMotion, wristYDelta, batSwingAngle, shoulderTilt, kneeBendAngle]
    """
    def __init__(self, input_dim=8, hidden_dim=32, num_faults=5):
        super(FaultMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_faults),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def extract_fault_features(metrics: Dict[str, Any], action_type: str) -> np.ndarray:
    """
    Standardizes a metric dictionary to a fixed-size 8-dimensional feature vector:
    1. headStabilityVariance
    2. avgHipTilt
    3. avgSpineOffset
    4. rangeOfMotion
    5. wristYDelta
    6. batSwingAngle
    7. shoulderTilt
    8. kneeBendAngle
    """
    # Use defaults if not present
    feat = [
        metrics.get("headStabilityVariance", 0.001) or 0.001,
        metrics.get("avgHipTilt", 0.02) or 0.02,
        metrics.get("avgSpineOffset", 0.02) or 0.02,
        metrics.get("rangeOfMotion", 45.0) or 45.0,
        metrics.get("wristYDelta", 0.1) or 0.1,
        metrics.get("batSwingAngle", 45.0) or 45.0,
        metrics.get("shoulderTilt", 0.02) or 0.02,
        metrics.get("kneeBendAngle", 150.0) or 150.0
    ]
    return np.array([feat], dtype=np.float32)

class FaultDetector:
    def __init__(self, model_path="fault_detector.pth"):
        self.model = FaultMLP()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_model_path = os.path.join(base_dir, model_path)
        if os.path.exists(abs_model_path):
            try:
                self.model.load_state_dict(torch.load(abs_model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"[Info] Fault Detector model loaded from {abs_model_path}")
            except Exception as e:
                print(f"[Warn] Failed to load Fault Detector: {e}")
        else:
            print(f"[Info] Fault Detector weights not found at {abs_model_path}. Using default initialization.")
            
    def detect(self, metrics: Dict[str, Any], action_type: str) -> List[Dict[str, Any]]:
        """
        Uses the MLP model to predict which faults are active.
        Returns a list of structured faults.
        """
        feat = extract_fault_features(metrics, action_type)
        x = torch.tensor(feat)
        
        with torch.no_grad():
            probs = self.model(x).squeeze(0).numpy()
            
        fault_names = [
            ("POOR_BALANCE", "poor balance", "avgHipTilt", 0.04, "moderate"),
            ("HEAD_MOVEMENT", "head movement", "headStabilityVariance", 0.002, "moderate"),
            ("STANCE_ISSUE", "stance issue", "batSwingAngle", 30.0, "minor"),
            ("WEAK_FOLLOW_THROUGH", "weak follow-through", "wristYDelta", 0.05, "moderate"),
            ("POOR_TIMING", "poor timing", "rangeOfMotion", 25.0, "moderate")
        ]
        
        detected_faults = []
        for i, prob in enumerate(probs):
            # If probability is > 0.5, we flag it as an active fault
            if prob > 0.5:
                code, desc, metric_name, threshold, severity = fault_names[i]
                
                # Check for critical severity if metric is significantly off
                val = metrics.get(metric_name)
                if val is not None:
                    if metric_name == "headStabilityVariance" and val > 0.005:
                        severity = "critical"
                    elif metric_name == "avgHipTilt" and val > 0.08:
                        severity = "critical"
                        
                detected_faults.append({
                    "faultCode": f"{code}_ML",
                    "faultText": f"{desc.capitalize()} detected by analysis model.",
                    "metric": metric_name,
                    "value": round(float(val), 4) if val is not None else None,
                    "threshold": threshold,
                    "severity": severity
                })
                
        return detected_faults
