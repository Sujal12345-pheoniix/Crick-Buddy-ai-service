"""
Body Posture Analysis Router — Deterministic Pipeline
=====================================================
All scores derived from real pose landmarks (multi-frame temporal where possible).
No random values. No hardcoded scores.
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

import httpx

from utils.analysis import (
    analyze_pose_from_image, analyze_pose_from_video_frames, extract_frames,
    calculate_angle, score_angle, generate_report, clamp_score,
    validate_cricket_content_async, validate_video_locally,
    calculate_balance_score,
    calculate_shoulder_alignment,
    detect_faults,
    compute_overall_posture_score,
    extract_raw_frame_metrics,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point,
)

router = APIRouter()


def analyze_posture_landmarks(landmarks: dict) -> dict:
    """
    Single-frame posture metrics.
    Used when only one image is available.
    """
    metrics = {}

    l_shoulder = get_point(landmarks, LEFT_SHOULDER)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    if l_shoulder and r_shoulder:
        tilt = abs(l_shoulder[1] - r_shoulder[1])
        metrics["shoulderTilt"] = round(tilt, 5)
        metrics["shoulderAlignmentScore"] = clamp_score(98 - tilt * 520, 35, 98)
        if tilt < 0.02:
            metrics["shoulderAlignmentNote"] = "Excellent — shoulders are perfectly level"
        elif tilt < 0.05:
            metrics["shoulderAlignmentNote"] = "Good — minor shoulder tilt"
        elif tilt < 0.09:
            metrics["shoulderAlignmentNote"] = "Fair — noticeable shoulder imbalance"
        else:
            metrics["shoulderAlignmentNote"] = "Needs work — significant shoulder tilt detected"

    l_hip = get_point(landmarks, LEFT_HIP)
    l_knee = get_point(landmarks, LEFT_KNEE)
    l_ankle = get_point(landmarks, LEFT_ANKLE)
    if l_hip and l_knee and l_ankle:
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        metrics["kneeBendAngle"] = knee_angle
        metrics["kneeBendScore"] = score_angle(knee_angle, 140, 170)
        if 140 <= knee_angle <= 170:
            metrics["kneeBendNote"] = f"Good knee bend ({knee_angle}°) — solid athletic base"
        elif knee_angle < 140:
            metrics["kneeBendNote"] = f"Over-flexed ({knee_angle}°) — too crouched, may restrict movement"
        else:
            metrics["kneeBendNote"] = f"Straight-legged ({knee_angle}°) — increase knee flex for explosiveness"

    r_hip = get_point(landmarks, RIGHT_HIP)
    if l_hip and r_hip:
        hip_tilt = abs(l_hip[1] - r_hip[1])
        metrics["hipTilt"] = round(hip_tilt, 5)
        metrics["balanceScore"] = clamp_score(98 - hip_tilt * 480, 35, 98)
        if hip_tilt < 0.02:
            metrics["balanceNote"] = "Excellent — hips level"
        elif hip_tilt < 0.05:
            metrics["balanceNote"] = "Good — minor hip tilt"
        else:
            metrics["balanceNote"] = "Needs work — significant hip tilt affecting weight distribution"

    if l_shoulder and r_shoulder and l_hip and r_hip:
        mid_sx = (l_shoulder[0] + r_shoulder[0]) / 2
        mid_hx = (l_hip[0] + r_hip[0]) / 2
        offset = abs(mid_sx - mid_hx)
        metrics["spineOffset"] = round(offset, 5)
        metrics["spinePosScore"] = clamp_score(98 - offset * 520, 35, 98)
        if offset < 0.03:
            metrics["spinePosNote"] = "Good — spine aligned over hips"
        elif offset < 0.07:
            metrics["spinePosNote"] = "Fair — slight lateral lean in spine"
        else:
            metrics["spinePosNote"] = "Poor — significant spine-hip misalignment"

    return metrics


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
        if any(ext in low for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]):
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
        all_landmarks_list = None

        if not is_video:
            # Image path
            img = cv2.imread(tmp_path)
            if img is not None:
                is_valid, err_msg = validate_video_locally([img], "posture")
                if not is_valid:
                    raise HTTPException(status_code=400, detail=err_msg)
            landmarks = analyze_pose_from_image(tmp_path)
        else:
            # Video path
            frames = extract_frames(tmp_path, num_frames=15)
            if not frames:
                raise HTTPException(400, "Invalid video: could not extract frames.")

            is_valid, err_msg = validate_video_locally(frames, "posture")
            if not is_valid:
                raise HTTPException(status_code=400, detail=err_msg)

            all_lm = analyze_pose_from_video_frames(frames)
            valid = [lm for lm in all_lm if lm is not None]
            if valid:
                # Check that action is not invalid
                from utils.analysis import get_action_classifier
                classifier = get_action_classifier()
                action_class = classifier.predict(all_lm)
                if action_class == "invalid":
                    raise HTTPException(
                        status_code=400,
                        detail="ERR_INVALID_ACTION: Uploaded video does not show valid cricket action or posture."
                    )

                # Use best-visibility frame as primary, but run temporal where possible
                landmarks = max(valid, key=lambda lm: sum(
                    lm[k][3] for k in lm if isinstance(lm[k], list) and len(lm[k]) > 3
                ))
                all_landmarks_list = all_lm


        if landmarks is None:
            raise HTTPException(
                400,
                "No human detected. Ensure the player is clearly visible, "
                "well-lit, and unobstructed in the frame."
            )

        # Single-frame metrics
        posture_metrics = analyze_posture_landmarks(landmarks)

        # Temporal metrics if video
        if all_landmarks_list:
            temporal_shoulder = calculate_shoulder_alignment(all_landmarks_list)
            temporal_balance = calculate_balance_score(all_landmarks_list)
            # Override with temporal (more reliable) values
            posture_metrics["shoulderAlignmentScore"] = temporal_shoulder["score"]
            posture_metrics["shoulderAlignmentNote"] = temporal_shoulder["note"]
            posture_metrics["balanceScore"] = temporal_balance["score"]
            posture_metrics["avgHipTilt"] = temporal_balance["avgHipTilt"]
            posture_metrics["balanceNote"] = temporal_balance["note"]

        # Deterministic overall score
        overall = compute_overall_posture_score(posture_metrics)
        posture_metrics["overallPostureScore"] = overall

        # Fault detection
        faults = detect_faults(posture_metrics, "posture")
        fault_codes = [f["faultCode"] for f in faults]

        report = await generate_report(posture_metrics, "posture", faults)

        return {
            "success": True,
            "type": "posture",
            "upload_id": upload_id,
            "posture_metrics": {
                "shoulderAlignmentScore": posture_metrics.get("shoulderAlignmentScore", 50),
                "shoulderAlignmentNote":  posture_metrics.get("shoulderAlignmentNote", ""),
                "kneeBendAngle":          posture_metrics.get("kneeBendAngle"),
                "kneeBendScore":          posture_metrics.get("kneeBendScore", 50),
                "kneeBendNote":           posture_metrics.get("kneeBendNote", ""),
                "balanceScore":           posture_metrics.get("balanceScore", 50),
                "balanceNote":            posture_metrics.get("balanceNote", ""),
                "spinePosScore":          posture_metrics.get("spinePosScore", 50),
                "spinePosNote":           posture_metrics.get("spinePosNote", ""),
                "overallPostureScore":    overall,
            },
            "overall_score": overall,
            "faults": faults,
            "fault_codes": fault_codes,
            "raw_frame_metrics": extract_raw_frame_metrics(all_landmarks_list or [landmarks], "posture"),
            "landmarks": landmarks,
            **report,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
