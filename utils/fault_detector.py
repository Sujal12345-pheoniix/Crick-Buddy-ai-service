"""
Rule-based technique fault detection for batting, bowling and posture.
Kept free of PyTorch so the analysis service stays small; the experimental
FaultMLP network lives in train_models.py.
"""

from typing import Any, Dict, List

import numpy as np


# ─── Runtime rule engine ──────────────────────────────────────────────────────
# Each rule: (code, text, metric, threshold, test(value) -> severity | None).
# Metric units match utils/pipeline.py: "torso" values are multiples of the
# player's shoulder-to-hip length, angles are degrees, scores are 0-100.

def _above(moderate, critical=None):
    return lambda v: ("critical" if critical is not None and v > critical else "moderate") if v > moderate else None


def _below(moderate, critical=None):
    return lambda v: ("critical" if critical is not None and v < critical else "moderate") if v < moderate else None


def _outside(lo, hi, severity="minor"):
    return lambda v: severity if v < lo or v > hi else None


FAULT_RULES = {
    "batting": [
        ("HEAD_INSTABILITY", "Head moves too much during the downswing", "headStabilityVariance", 0.08, _above(0.08, 0.12)),
        ("LATE_TIMING", "Front foot lands at or after the bat swing peaks", "plantToImpactMs", 50, _below(50, 0)),
        ("EARLY_TIMING", "Front foot plants long before contact", "plantToImpactMs", 600, _above(600)),
        ("STIFF_FRONT_KNEE", "Front knee barely bends in the stride", "peakKneeFlexion", 165, _above(165)),
        ("POOR_BALANCE", "Balance breaks down at contact", "balanceScore", 55, _below(55, 40)),
        ("SHORT_FOLLOW_THROUGH", "Follow-through is cut short", "wristYDelta", 0.3, _below(0.3, 0.0)),
        ("STANCE_WIDTH", "Stride width outside the ideal range", "avgStrideRatio", 0.6, _outside(0.6, 2.0)),
    ],
    "bowling": [
        ("LOW_RELEASE", "Release point is low", "releaseHeight", 0.6, _below(0.6, 0.3)),
        ("BENT_ARM_AT_RELEASE", "Bowling arm visibly bent at release", "releaseArmAngle", 150, _below(150)),
        ("JERKY_ACTION", "Arm rotation is not smooth", "armSmoothnessScore", 60, _below(60, 40)),
        ("POOR_BALANCE", "Balance breaks down at release", "balanceScore", 55, _below(55, 40)),
        ("WRIST_DROP", "Wrist drops below the line of the forearm at release", "wristElbowDelta", 0.15, _below(0.15)),
    ],
    "posture": [
        ("SHOULDER_TILT", "Shoulders are not level", "shoulderTiltDeg", 6, _above(6, 12)),
        ("KNEE_ANGLE", "Knee bend outside the athletic range", "kneeBendAngle", 130, _outside(130, 175, "moderate")),
        ("SPINE_LEAN", "Spine leans away from vertical", "spineLeanDeg", 20, _above(20, 35)),
    ],
}


class FaultDetector:
    def __init__(self, model_path="fault_detector.pth"):
        print("[Info] Fault Detector initialized in deterministic rule-based mode.")

    def detect(self, metrics: Dict[str, Any], action_type: str) -> List[Dict[str, Any]]:
        """
        Applies the rules for `action_type`. A rule only fires when its metric was
        actually measured — missing metrics are never replaced by defaults.
        """
        faults = []
        for code, text, metric, threshold, test in FAULT_RULES.get(action_type, []):
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                continue
            severity = test(value)
            if severity:
                faults.append({
                    "faultCode": code,
                    "faultText": text,
                    "metric": metric,
                    "value": round(float(value), 3),
                    "threshold": threshold,
                    "severity": severity,
                })
        return faults
