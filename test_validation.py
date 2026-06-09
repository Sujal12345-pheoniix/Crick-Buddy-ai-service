import numpy as np
import sys
import os

# Ensure the parent directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.analysis import validate_video_locally

def test_validation_empty_frames():
    print("Testing validation with empty frames list...")
    is_valid, msg = validate_video_locally([], "batting")
    assert not is_valid, "Should reject empty frames"
    assert "ERR_INVALID_VIDEO" in msg, "Should return correct error code"
    print("Success: Empty frames rejected correctly.")

def test_validation_blank_frames():
    print("Testing validation with blank frames (no human)...")
    # Create 5 blank frames (black images)
    blank_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    is_valid, msg = validate_video_locally(blank_frames, "batting")
    
    assert not is_valid, "Should reject blank frames with no human"
    assert "ERR_NO_HUMAN_DETECTED" in msg, "Should return ERR_NO_HUMAN_DETECTED"
    print("Success: Blank frames rejected correctly.")

if __name__ == "__main__":
    try:
        test_validation_empty_frames()
        test_validation_blank_frames()
        print("\nAll local validation checks passed successfully! ✅")
    except AssertionError as e:
        print(f"\nAssertion failed: {e} ❌")
        sys.exit(1)
    except Exception as e:
        print(f"\nRuntime error during tests: {e} ❌")
        sys.exit(1)
