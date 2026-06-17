import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.action_classifier import ActionClassifier
from utils.fault_detector import FaultDetector

def test_action_classification_bowling():
    print("Testing action classification: Bowling...")
    classifier = ActionClassifier()
    
    # Mock a sequence of 10 frames where:
    # Right wrist (16) starts high (y=0.1, above shoulder) and ends low (y=0.8)
    # Right shoulder (12) is static at y=0.4
    all_landmarks = []
    for i in range(10):
        t = i / 9.0
        frame_lms = {
            16: [0.5, 0.1 + 0.7 * t], # Wrist X, Y
            12: [0.5, 0.4]            # Shoulder X, Y
        }
        all_landmarks.append(frame_lms)
        
    cls = classifier.predict(all_landmarks)
    assert cls == "bowling", f"Expected bowling, got {cls}"
    print("Success: Bowling trajectory classified correctly.")

def test_action_classification_batting():
    print("Testing action classification: Batting...")
    classifier = ActionClassifier()
    
    # Mock a sequence of 10 frames where:
    # Right wrist (16) swings horizontally (x goes from 0.3 to 0.7) below shoulder level (y=0.6, shoulder at y=0.4)
    all_landmarks = []
    for i in range(10):
        t = i / 9.0
        frame_lms = {
            16: [0.3 + 0.4 * t, 0.6], # Wrist X, Y
            12: [0.5, 0.4]            # Shoulder X, Y
        }
        all_landmarks.append(frame_lms)
        
    cls = classifier.predict(all_landmarks)
    assert cls == "batting", f"Expected batting, got {cls}"
    print("Success: Batting trajectory classified correctly.")

def test_fault_detection():
    print("Testing fault detection...")
    detector = FaultDetector()
    
    # Mock metrics with high head stability variance (0.006) and poor hip tilt (0.08)
    metrics = {
        "headStabilityVariance": 0.006,
        "avgHipTilt": 0.08,
        "avgSpineOffset": 0.01,
        "rangeOfMotion": 40.0,
        "wristYDelta": 0.1,
        "batSwingAngle": 45.0
    }
    
    faults = detector.detect(metrics, "batting")
    codes = [f["faultCode"] for f in faults]
    
    assert "HEAD_MOVEMENT_ML" in codes, "Should detect HEAD_MOVEMENT"
    assert "POOR_BALANCE_ML" in codes, "Should detect POOR_BALANCE"
    
    # Check that HEAD_MOVEMENT gets upgraded to critical severity due to high variance
    head_fault = next(f for f in faults if f["faultCode"] == "HEAD_MOVEMENT_ML")
    assert head_fault["severity"] == "critical", "Should upgrade high head movement to critical severity"
    
    print("Success: Faults detected and prioritized correctly.")

if __name__ == "__main__":
    try:
        test_action_classification_bowling()
        test_action_classification_batting()
        test_fault_detection()
        print("\nAll kinematics tests passed successfully! [OK]")
    except AssertionError as e:
        print(f"\nAssertion failed: {e} [FAIL]")
        sys.exit(1)
    except Exception as e:
        print(f"\nRuntime error during tests: {e} [ERROR]")
        sys.exit(1)
