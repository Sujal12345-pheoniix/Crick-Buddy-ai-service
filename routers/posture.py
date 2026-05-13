"""
Body Posture Analysis Router
Uses MediaPipe Pose on images to analyze:
- Shoulder alignment, knee bend, balance, spine position
"""

import os
import random
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from utils.analysis import (
    analyze_pose_from_image, analyze_pose_from_video_frames, extract_frames,
    calculate_angle, score_angle, generate_report, clamp_score,
    is_cricket_content,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point
)

router = APIRouter()


def _tilt_to_score(tilt: float, scale: float, low: int = 35, high: int = 98) -> int:
    return clamp_score(high - (tilt * scale), low, high)


def analyze_posture_landmarks(landmarks: dict) -> dict:
    """Extract posture metrics from MediaPipe landmarks."""
    metrics = {}

    # Shoulder alignment (horizontal level check)
    l_shoulder = get_point(landmarks, LEFT_SHOULDER)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    if l_shoulder and r_shoulder:
        shoulder_tilt = abs(l_shoulder[1] - r_shoulder[1])
        metrics['shoulderTilt'] = round(shoulder_tilt, 4)
        metrics['shoulderAlignmentScore'] = _tilt_to_score(shoulder_tilt, 520)
        if shoulder_tilt < 0.03:
            metrics['shoulderAlignmentNote'] = "Excellent — shoulders are level"
        elif shoulder_tilt < 0.07:
            metrics['shoulderAlignmentNote'] = "Good — minor shoulder tilt detected"
        else:
            metrics['shoulderAlignmentNote'] = "Needs work — significant shoulder imbalance"

    # Knee bend angle (left leg)
    l_hip = get_point(landmarks, LEFT_HIP)
    l_knee = get_point(landmarks, LEFT_KNEE)
    l_ankle = get_point(landmarks, LEFT_ANKLE)
    if l_hip and l_knee and l_ankle:
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        metrics['kneeBendAngle'] = knee_angle
        metrics['kneeBendScore'] = score_angle(knee_angle, 140, 170)

    # Balance score (hip alignment)
    l_hip_pt = get_point(landmarks, LEFT_HIP)
    r_hip_pt = get_point(landmarks, RIGHT_HIP)
    if l_hip_pt and r_hip_pt:
        hip_tilt = abs(l_hip_pt[1] - r_hip_pt[1])
        metrics['hipTilt'] = round(hip_tilt, 4)
        metrics['balanceScore'] = _tilt_to_score(hip_tilt, 480)

    # Spine score (shoulder vs hip vertical alignment)
    if l_shoulder and r_shoulder and l_hip_pt and r_hip_pt:
        mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
        mid_hip_x = (l_hip_pt[0] + r_hip_pt[0]) / 2
        spine_offset = abs(mid_shoulder_x - mid_hip_x)
        metrics['spineOffset'] = round(spine_offset, 4)
        metrics['spinePosScore'] = _tilt_to_score(spine_offset, 520)

    return metrics


@router.post("/posture")
async def analyze_posture(
    file: UploadFile = File(...),
    upload_id: Optional[str] = Form(None)
):
    """Analyze body posture from image using MediaPipe Pose."""

    suffix = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        posture_metrics = {}
        landmarks_data = None

        # First try MediaPipe image inference.
        landmarks = analyze_pose_from_image(tmp_path)

        if isinstance(landmarks, dict) and landmarks.get("error") == "wrong_content":
            raise HTTPException(
                status_code=400, 
                detail="Wrong image/video uploaded. Please upload a cricket-related posture image or video."
            )

        # If upload is a video, analyze sampled frames and take a stable middle detection.
        if not landmarks:
            frames = extract_frames(tmp_path, num_frames=24)
            if frames:
                # Check first frame for cricket content
                if not is_cricket_content(frames[0]):
                    raise HTTPException(
                        status_code=400, 
                        detail="Wrong video uploaded. Please upload a cricket-related video."
                    )
                
                sequence_landmarks = analyze_pose_from_video_frames(frames)
                valid = [lm for lm in sequence_landmarks if lm is not None]
                if valid:
                    landmarks = valid[len(valid) // 2]

        if landmarks:
            posture_metrics = analyze_posture_landmarks(landmarks)
            landmarks_data = landmarks
        else:
            raise HTTPException(status_code=400, detail="No human detected in the image. Please ensure the player is clearly visible.")

        # Overall posture score
        scores = [
            posture_metrics.get('shoulderAlignmentScore', 70),
            posture_metrics.get('kneeBendScore', 70),
            posture_metrics.get('balanceScore', 70),
            posture_metrics.get('spinePosScore', 70),
        ]
        overall_score = round(sum(scores) / len(scores))
        posture_metrics['overallPostureScore'] = overall_score

        report = await generate_report(posture_metrics, 'posture')

        return {
            "success": True,
            "type": "posture",
            "upload_id": upload_id,
            "posture_metrics": {
                "shoulderAlignmentScore": posture_metrics.get('shoulderAlignmentScore', 70),
                "shoulderAlignmentNote": posture_metrics.get('shoulderAlignmentNote', ''),
                "kneeBendAngle": posture_metrics.get('kneeBendAngle', 150.0),
                "kneeBendScore": posture_metrics.get('kneeBendScore', 70),
                "balanceScore": posture_metrics.get('balanceScore', 70),
                "spinePosScore": posture_metrics.get('spinePosScore', 70),
                "overallPostureScore": overall_score
            },
            "overall_score": overall_score,
            "landmarks": landmarks_data,
            **report
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass



