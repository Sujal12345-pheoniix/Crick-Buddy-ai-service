"""
Body Posture Analysis Router
MediaPipe + YOLO pipeline:
  Image/video → YOLO crop → MediaPipe pose → posture metrics
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

import httpx

from utils.analysis import (
    analyze_pose_from_image, analyze_pose_from_video_frames, extract_frames,
    calculate_angle, score_angle, generate_report, clamp_score,
    validate_cricket_content_async,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point,
)

router = APIRouter()


# ─── Scoring helper ───────────────────────────────────────────────────────────

def _tilt_to_score(tilt: float, scale: float, low: int = 35, high: int = 98) -> int:
    return clamp_score(high - tilt * scale, low, high)


def analyze_posture_landmarks(landmarks: dict) -> dict:
    metrics = {}

    l_shoulder = get_point(landmarks, LEFT_SHOULDER)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    if l_shoulder and r_shoulder:
        tilt = abs(l_shoulder[1] - r_shoulder[1])
        metrics["shoulderTilt"] = round(tilt, 4)
        metrics["shoulderAlignmentScore"] = _tilt_to_score(tilt, 520)
        if tilt < 0.03:
            metrics["shoulderAlignmentNote"] = "Excellent — shoulders are level"
        elif tilt < 0.07:
            metrics["shoulderAlignmentNote"] = "Good — minor shoulder tilt"
        else:
            metrics["shoulderAlignmentNote"] = "Needs work — significant shoulder imbalance"

    l_hip   = get_point(landmarks, LEFT_HIP)
    l_knee  = get_point(landmarks, LEFT_KNEE)
    l_ankle = get_point(landmarks, LEFT_ANKLE)
    if l_hip and l_knee and l_ankle:
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        metrics["kneeBendAngle"] = knee_angle
        metrics["kneeBendScore"] = score_angle(knee_angle, 140, 170)

    r_hip = get_point(landmarks, RIGHT_HIP)
    if l_hip and r_hip:
        hip_tilt = abs(l_hip[1] - r_hip[1])
        metrics["hipTilt"] = round(hip_tilt, 4)
        metrics["balanceScore"] = _tilt_to_score(hip_tilt, 480)

    if l_shoulder and r_shoulder and l_hip and r_hip:
        mid_sx = (l_shoulder[0] + r_shoulder[0]) / 2
        mid_hx = (l_hip[0] + r_hip[0]) / 2
        offset = abs(mid_sx - mid_hx)
        metrics["spineOffset"] = round(offset, 4)
        metrics["spinePosScore"] = _tilt_to_score(offset, 520)

    return metrics


# ─── Route ────────────────────────────────────────────────────────────────────

@router.post("/posture")
async def analyze_posture(
    file: Optional[UploadFile] = File(None),
    fileUrl: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
):
    if not file and not fileUrl:
        raise HTTPException(400, "No file or fileUrl provided")

    suffix = ".jpg"
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
    elif fileUrl:
        low = fileUrl.lower()
        if ".mp4" in low or ".mov" in low or ".avi" in low:
            suffix = ".mp4"

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
        import cv2

        is_video = suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        landmarks = None

        if not is_video:
            # ── Image path ────────────────────────────────────────────────────
            img = cv2.imread(tmp_path)
            if img is not None:
                if not await validate_cricket_content_async(img):
                    raise HTTPException(
                        400,
                        "Wrong file uploaded. Please upload a cricket posture image."
                    )
            landmarks = analyze_pose_from_image(tmp_path)

        else:
            # ── Video path ────────────────────────────────────────────────────
            frames = extract_frames(tmp_path, num_frames=12)
            if not frames:
                raise HTTPException(400, "Invalid video: could not extract frames.")

            mid_frame = frames[len(frames) // 2]
            if not await validate_cricket_content_async(mid_frame):
                raise HTTPException(
                    400,
                    "Content validation failed. Please upload a cricket posture video."
                )

            all_lm = analyze_pose_from_video_frames(frames)
            valid = [lm for lm in all_lm if lm is not None]
            if valid:
                landmarks = valid[len(valid) // 2]

        if landmarks is None:
            raise HTTPException(
                400,
                "No human detected. Ensure the player is clearly visible, "
                "well-lit, and unobstructed in the frame."
            )

        posture_metrics = analyze_posture_landmarks(landmarks)

        scores = [
            posture_metrics.get("shoulderAlignmentScore", 70),
            posture_metrics.get("kneeBendScore", 70),
            posture_metrics.get("balanceScore", 70),
            posture_metrics.get("spinePosScore", 70),
        ]
        overall = round(sum(scores) / len(scores))
        posture_metrics["overallPostureScore"] = overall

        report = await generate_report(posture_metrics, "posture")

        return {
            "success": True,
            "type": "posture",
            "upload_id": upload_id,
            "posture_metrics": {
                "shoulderAlignmentScore": posture_metrics.get("shoulderAlignmentScore", 70),
                "shoulderAlignmentNote":  posture_metrics.get("shoulderAlignmentNote", ""),
                "kneeBendAngle":          posture_metrics.get("kneeBendAngle", 150.0),
                "kneeBendScore":          posture_metrics.get("kneeBendScore", 70),
                "balanceScore":           posture_metrics.get("balanceScore", 70),
                "spinePosScore":          posture_metrics.get("spinePosScore", 70),
                "overallPostureScore":    overall,
            },
            "overall_score": overall,
            "landmarks": landmarks,
            **report,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
