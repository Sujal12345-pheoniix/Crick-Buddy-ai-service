"""
Bowling Video Analysis Router
MediaPipe + YOLO pipeline:
  Frame sampling → blur filter → YOLO crop → MediaPipe pose → feature extraction
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

import httpx

from utils.analysis import (
    extract_frames, analyze_pose_from_video_frames, estimate_ball_speed,
    calculate_angle, score_angle, generate_report, clamp_score,
    validate_cricket_content_async,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point,
)

router = APIRouter()


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _wrist_position_score(wrist_y: float, elbow_y: float) -> int:
    delta = elbow_y - wrist_y
    if delta >= 0:
        return clamp_score(74 + delta * 240, 45, 98)
    return clamp_score(68 + delta * 180, 30, 90)


def _release_point_score(wrist_y: float, nose_y: float) -> int:
    delta = nose_y - wrist_y
    if delta >= 0:
        return clamp_score(78 + delta * 220, 45, 98)
    return clamp_score(72 + delta * 170, 30, 90)


def classify_bowling_style(ball_speed: float, arm_angle: float) -> str:
    if ball_speed >= 135:    return "Fast Bowler"
    if ball_speed >= 120:    return "Medium-Fast"
    if ball_speed >= 100:
        return "Swing Bowler" if arm_angle > 80 else "Medium Pace"
    return "Off Spinner" if arm_angle > 75 else "Leg Spinner"


def analyze_bowling_landmarks(landmarks: dict) -> dict:
    metrics = {}

    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    r_elbow    = get_point(landmarks, RIGHT_ELBOW)
    r_wrist    = get_point(landmarks, RIGHT_WRIST)
    if r_shoulder and r_elbow and r_wrist:
        arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        metrics["armRotationAngle"] = arm_angle
        metrics["armRotationScore"] = score_angle(arm_angle, 160, 180)

    if r_wrist and r_elbow:
        metrics["wristPositionScore"] = _wrist_position_score(r_wrist[1], r_elbow[1])
        metrics["wristPositionNote"] = (
            "Good — wrist over the ball at release"
            if r_wrist[1] < r_elbow[1]
            else "Needs work — dropped wrist affects seam position"
        )

    l_hip   = get_point(landmarks, LEFT_HIP)
    l_knee  = get_point(landmarks, LEFT_KNEE)
    l_ankle = get_point(landmarks, LEFT_ANKLE)
    if l_hip and l_knee and l_ankle:
        bal = calculate_angle(l_hip, l_knee, l_ankle)
        metrics["balanceScore"] = score_angle(bal, 170, 180)

    nose = get_point(landmarks, NOSE)
    if r_wrist and nose:
        metrics["releasePointScore"] = _release_point_score(r_wrist[1], nose[1])
        metrics["releasePointNote"] = (
            "High release point — good for bounce extraction"
            if r_wrist[1] < nose[1]
            else "Low release point — ball may lack bounce"
        )

    return metrics


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post("/bowling")
async def analyze_bowling(
    file: Optional[UploadFile] = File(None),
    fileUrl: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
):
    if not file and not fileUrl:
        raise HTTPException(400, "No file or fileUrl provided")

    suffix = ".mp4"
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if file:
            tmp.write(await file.read())
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("GET", fileUrl) as resp:
                    if resp.status_code != 200:
                        raise HTTPException(400, "Failed to download file from URL")
                    async for chunk in resp.aiter_bytes(1024 * 1024):
                        tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # ── 1. Extract frames ─────────────────────────────────────────────────────────
        frames = extract_frames(tmp_path, num_frames=15)
        if not frames:
            raise HTTPException(400, "Invalid video: could not extract frames.")

        # ── 2. Cricket content validation ─────────────────────────────────────
        mid_frame = frames[len(frames) // 2]
        if not await validate_cricket_content_async(mid_frame):
            raise HTTPException(
                400,
                "Content validation failed. Please upload a cricket bowling video."
            )

        # ── 3. Ball speed via optical flow ──────────────────────────────────
        ball_speed = estimate_ball_speed(frames)
        if ball_speed is None:
            # Fallback — estimate from typical medium-pace; still produce a full report
            ball_speed = 115.0

        # ── 4. Multi-frame pose detection ─────────────────────────────────────
        bowling_metrics: dict = {"estimatedBallSpeed": ball_speed}
        all_landmarks = analyze_pose_from_video_frames(frames)
        valid = [lm for lm in all_landmarks if lm is not None]

        if not valid:
            raise HTTPException(
                400,
                "No human detected in the video. Ensure the bowler is clearly "
                "visible, well-lit, and in the centre of the frame."
            )

        mid_lm = valid[len(valid) // 2]
        bowling_metrics.update(analyze_bowling_landmarks(mid_lm))
        bowling_metrics["bowlingStyle"] = classify_bowling_style(
            ball_speed, bowling_metrics.get("armRotationAngle", 160)
        )

        # ── 5. Score aggregation ──────────────────────────────────────────────
        scores = [
            bowling_metrics.get("wristPositionScore", 70),
            bowling_metrics.get("armRotationScore", 70),
            bowling_metrics.get("releasePointScore", 70),
            bowling_metrics.get("balanceScore", 70),
        ]
        overall = round(sum(scores) / len(scores))
        bowling_metrics["overallBowlingScore"] = overall

        report = await generate_report(bowling_metrics, "bowling")

        return {
            "success": True,
            "type": "bowling",
            "upload_id": upload_id,
            "frames_analysed": len(frames),
            "frames_with_pose": len(valid),
            "bowling_metrics": {
                "wristPositionScore":  bowling_metrics.get("wristPositionScore", 70),
                "wristPositionNote":   bowling_metrics.get("wristPositionNote", ""),
                "armRotationAngle":    bowling_metrics.get("armRotationAngle", 165.0),
                "armRotationScore":    bowling_metrics.get("armRotationScore", 70),
                "releasePointScore":   bowling_metrics.get("releasePointScore", 70),
                "releasePointNote":    bowling_metrics.get("releasePointNote", ""),
                "estimatedBallSpeed":  ball_speed,
                "balanceScore":        bowling_metrics.get("balanceScore", 70),
                "bowlingStyle":        bowling_metrics.get("bowlingStyle", "Medium Pace"),
                "overallBowlingScore": overall,
            },
            "overall_score": overall,
            "landmarks": mid_lm,
            **report,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
