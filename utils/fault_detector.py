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
        print("[Info] Fault Detector initialized in deterministic rule-based mode.")
            
    def detect(self, metrics: Dict[str, Any], action_type: str) -> List[Dict[str, Any]]:
        """
        Uses explicit, mathematically verifiable rules to detect biomechanical faults.
        Returns a list of structured faults.
        """
        # Read the metrics, using standard safe defaults if not computed
        head_var = metrics.get("headStabilityVariance", 0.001) or 0.001
        hip_tilt = metrics.get("avgHipTilt", 0.02) or 0.02
        spine_offset = metrics.get("avgSpineOffset", 0.02) or 0.02
        rom = metrics.get("rangeOfMotion", 45.0) or 45.0
        wrist_delta = metrics.get("wristYDelta", 0.1) or 0.1
        bat_angle = metrics.get("batSwingAngle", 45.0) or 45.0
        
        fault_rules = [
            ("POOR_BALANCE", "poor balance", "avgHipTilt", hip_tilt, 0.05, hip_tilt > 0.05 or spine_offset > 0.05, "moderate"),
            ("HEAD_MOVEMENT", "head movement", "headStabilityVariance", head_var, 0.003, head_var > 0.003, "moderate"),
            ("STANCE_ISSUE", "stance issue", "batSwingAngle", bat_angle, 35.0, bat_angle < 35.0 or bat_angle > 75.0, "minor"),
            ("WEAK_FOLLOW_THROUGH", "weak follow-through", "wristYDelta", wrist_delta, 0.05, wrist_delta < 0.05, "moderate"),
            ("POOR_TIMING", "poor timing", "rangeOfMotion", rom, 25.0, rom < 25.0, "moderate")
        ]
        
        detected_faults = []
        for code, desc, metric_name, val, threshold, is_active, severity in fault_rules:
            if is_active:
                # Upgrade severity if metric is critically off
                if metric_name == "headStabilityVariance" and val > 0.005:
                    severity = "critical"
                elif metric_name == "avgHipTilt" and val > 0.08:
                    severity = "critical"
                    
                detected_faults.append({
                    "faultCode": f"{code}_ML",
                    "faultText": f"{desc.capitalize()} detected by kinematics engine.",
                    "metric": metric_name,
                    "value": round(float(val), 4) if val is not None else None,
                    "threshold": threshold,
                    "severity": severity
                })
                
        return detected_faults

