"""
Batting Video Analysis Router
Uses MediaPipe + OpenCV to analyze:
- Stance, bat swing angle, head position, timing, follow-through
- Shot recognition (cover drive, pull, cut, sweep)
"""

import os
import random
import tempfile
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from utils.analysis import (
    extract_frames, analyze_pose_from_video_frames, analyze_pose_from_image,
    calculate_angle, score_angle, generate_report, clamp_score,
    validate_cricket_content_async,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    get_point
)

router = APIRouter()

SHOT_TYPES = ["Cover Drive", "Pull Shot", "Cut Shot", "Sweep", "Straight Drive", "Flick"]


def _head_position_score(head_offset: float) -> int:
    # Typical normalized x-offset sits between ~0.0 and ~0.2 for most user videos.
    return clamp_score(98 - (head_offset * 280), 40, 98)


def _follow_through_score(wrist_y: float, nose_y: float) -> int:
    # Smaller y means higher landmark in image coordinates.
    delta = nose_y - wrist_y
    if delta >= 0:
        return clamp_score(76 + delta * 220, 50, 98)
    return clamp_score(68 + delta * 180, 35, 90)


def classify_shot(landmarks_list: list) -> str:
    """Deterministic shot classifier using pose geometry and motion trends."""
    valid = [lm for lm in landmarks_list if lm]
    if not valid:
        return "Straight Drive"

    try:
        wrist_x = []
        shoulder_x = []
        knee_angles = []

        for lm in valid:
            rw = get_point(lm, RIGHT_WRIST)
            rs = get_point(lm, RIGHT_SHOULDER)
            rh = get_point(lm, RIGHT_HIP)
            rk = get_point(lm, RIGHT_KNEE)
            ra = get_point(lm, RIGHT_ANKLE)

            if rw and rs:
                wrist_x.append(rw[0])
                shoulder_x.append(rs[0])

            if rh and rk and ra:
                knee_angles.append(calculate_angle(rh, rk, ra))

        if not wrist_x or not shoulder_x:
            return "Straight Drive"

        x_travel = wrist_x[-1] - wrist_x[0]
        offset_from_shoulder = wrist_x[-1] - shoulder_x[-1]
        avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else 150.0

        if avg_knee < 128:
            return "Sweep"
        if offset_from_shoulder > 0.15 and x_travel > 0.12:
            return "Pull Shot"
        if offset_from_shoulder < -0.12 and x_travel < -0.10:
            return "Cut Shot"
        if abs(offset_from_shoulder) < 0.06 and abs(x_travel) < 0.08:
            return "Straight Drive"
        if x_travel > 0.04:
            return "Flick"
        return "Cover Drive"
    except Exception as e:
        print(f"Heuristic shot classification error: {e}")
        return "Straight Drive"


def analyze_batting_landmarks(landmarks: dict) -> dict:
    """Extract batting-specific metrics from a single frame landmarks."""
    metrics = {}

    # Head position relative to shoulders
    nose = get_point(landmarks, NOSE)
    l_shoulder = get_point(landmarks, LEFT_SHOULDER)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)

    if nose and l_shoulder and r_shoulder:
        mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
        head_offset = abs(nose[0] - mid_shoulder_x)
        metrics['headOffset'] = round(head_offset, 4)
        if head_offset < 0.05:
            metrics['headPosition'] = "Excellent — aligned over off stump"
        elif head_offset < 0.12:
            metrics['headPosition'] = "Good — slight lateral movement"
        else:
            metrics['headPosition'] = "Needs work — head falling over"
        metrics['headPositionScore'] = _head_position_score(head_offset)

    # Bat swing angle (using right wrist → right elbow → right shoulder)
    r_wrist = get_point(landmarks, RIGHT_WRIST)
    r_elbow = get_point(landmarks, RIGHT_ELBOW)
    r_shoulder_pt = get_point(landmarks, RIGHT_SHOULDER)

    if r_wrist and r_elbow and r_shoulder_pt:
        swing_angle = calculate_angle(r_wrist, r_elbow, r_shoulder_pt)
        metrics['batSwingAngle'] = swing_angle
        metrics['stanceScore'] = score_angle(swing_angle, 30, 70)

    # Timing / weight transfer (knee bend)
    l_hip = get_point(landmarks, LEFT_HIP)
    l_knee = get_point(landmarks, LEFT_KNEE)
    l_ankle = get_point(landmarks, LEFT_ANKLE)

    if l_hip and l_knee and l_ankle:
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        metrics['kneeBendAngle'] = knee_angle
        metrics['timingScore'] = score_angle(knee_angle, 120, 160)

    # Follow-through (right wrist above head = good follow-through)
    if r_wrist and nose:
        metrics['followThroughScore'] = _follow_through_score(r_wrist[1], nose[1])

    return metrics


import requests

@router.post("/batting")
async def analyze_batting(
    file: Optional[UploadFile] = File(None),
    fileUrl: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None)
):
    """Analyze batting video using MediaPipe pose estimation."""
    if not file and not fileUrl:
        raise HTTPException(status_code=400, detail="No file or fileUrl provided")

    # Save uploaded file temporarily
    suffix = '.mp4'
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1]
    elif fileUrl and (fileUrl.endswith('.mp4') or fileUrl.endswith('.mov')):
        suffix = '.mp4'

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if file:
            content = await file.read()
            tmp.write(content)
        elif fileUrl:
            response = requests.get(fileUrl, stream=True)
            if response.status_code == 200:
                for chunk in response.iter_content(1024 * 1024):
                    tmp.write(chunk)
            else:
                raise HTTPException(status_code=400, detail="Failed to download file from URL")
        tmp_path = tmp.name

    try:
        landmarks_data = None
        batting_metrics = {}

        # Try real MediaPipe analysis
        frames = extract_frames(tmp_path, num_frames=30)

        if frames:
            # Validate if it's cricket content
            is_valid = await validate_cricket_content_async(frames[0])
            if not is_valid:
                raise HTTPException(
                    status_code=400, 
                    detail="Wrong video uploaded. Please upload a cricket batting or bowling clip."
                )

            all_landmarks = analyze_pose_from_video_frames(frames)
            valid = [lm for lm in all_landmarks if lm is not None]

            if valid:
                # Use middle frame for primary analysis
                mid_lm = valid[len(valid) // 2]
                batting_metrics = analyze_batting_landmarks(mid_lm)
                landmarks_data = mid_lm

                # Shot classification
                batting_metrics['shotType'] = classify_shot(valid)
            else:
                raise HTTPException(status_code=400, detail="No human detected in the video. Please ensure the player is clearly visible.")
        else:
            raise HTTPException(status_code=400, detail="Invalid video: Could not extract frames.")

        # Calculate overall score
        scores = [
            batting_metrics.get('stanceScore', 70),
            batting_metrics.get('headPositionScore', 70),
            batting_metrics.get('timingScore', 70),
            batting_metrics.get('followThroughScore', 70),
        ]
        overall_score = round(sum(scores) / len(scores))
        batting_metrics['overallBattingScore'] = overall_score

        # Generate AI report
        report = await generate_report(batting_metrics, 'batting')

        return {
            "success": True,
            "type": "batting",
            "upload_id": upload_id,
            "batting_metrics": {
                "stanceScore": batting_metrics.get('stanceScore', 70),
                "batSwingAngle": batting_metrics.get('batSwingAngle', 45.0),
                "headPosition": batting_metrics.get('headPosition', 'Analysis in progress'),
                "headPositionScore": batting_metrics.get('headPositionScore', 70),
                "timingScore": batting_metrics.get('timingScore', 70),
                "followThroughScore": batting_metrics.get('followThroughScore', 70),
                "shotType": batting_metrics.get('shotType', 'Cover Drive'),
                "overallBattingScore": overall_score
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



