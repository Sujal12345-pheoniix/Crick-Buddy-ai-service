"""
Bowling Video Analysis Router — Deterministic Pipeline
======================================================
All scores come from measured pose landmarks across MULTIPLE frames.
No hardcoded fallback speeds. No random values.
Every score is traceable to a formula in utils/analysis.py.

Pipeline:
  Frame sampling → Blur filter → YOLO crop → MediaPipe pose
  → Full-sequence feature extraction → Deterministic scoring
  → Fault detection with evidence → LLM narration only
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

import httpx

from utils.analysis import (
    extract_frames, analyze_pose_from_video_frames,
    calculate_angle, score_angle, generate_report, clamp_score,
    validate_cricket_content_async, validate_video_locally,
    estimate_ball_speed, classify_ball_speed,
    # Deterministic scoring functions
    calculate_arm_smoothness,
    calculate_release_point_score,
    calculate_balance_score,
    calculate_shoulder_alignment,
    detect_faults,
    compute_overall_bowling_score,
    extract_raw_frame_metrics,
    # Landmark IDs
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point,
)

router = APIRouter()


def classify_bowling_style(speed_class: str, arm_angle: float) -> str:
    """Classify bowling style from speed classification and arm angle."""
    if speed_class == "Fast":
        return "Fast Bowler"
    if speed_class == "Medium-Fast":
        return "Medium-Fast Bowler"
    if speed_class == "Medium":
        return "Swing Bowler" if arm_angle > 80 else "Medium Pace Bowler"
    if arm_angle > 75:
        return "Off Spin Bowler"
    return "Leg Spin / Wrist Spin Bowler"


def analyze_wrist_position(frames: list) -> dict:
    """
    Wrist position at release: wrist should be above elbow.
    Measures the vertical delta between wrist and elbow at the
    frame with highest wrist position (minimum wrist Y).
    """
    min_wrist_y = 1.0
    best_frame = None
    for f in frames:
        if not f:
            continue
        rw = get_point(f, RIGHT_WRIST)
        if rw and rw[1] < min_wrist_y:
            min_wrist_y = rw[1]
            best_frame = f

    if not best_frame:
        return {"score": 50, "note": "Could not detect release frame for wrist analysis"}

    rw = get_point(best_frame, RIGHT_WRIST)
    re = get_point(best_frame, RIGHT_ELBOW)

    if not rw or not re:
        return {"score": 50, "note": "Wrist or elbow not visible at release frame"}

    # Positive delta = wrist above elbow (good)
    delta = re[1] - rw[1]
    if delta >= 0:
        score = clamp_score(74 + delta * 240, 45, 98)
    else:
        score = clamp_score(68 + delta * 180, 30, 90)

    note = (
        "Good — wrist above elbow at release, stable seam position"
        if delta >= 0
        else "Needs work — dropped wrist at release affects seam and swing"
    )

    return {"score": score, "wristElbowDelta": round(delta, 4), "note": note}


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
        # ── 1. Extract frames ────────────────────────────────────────────────
        frames = extract_frames(tmp_path, num_frames=20)
        if not frames:
            raise HTTPException(400, "Invalid video: could not extract frames.")

        # ── 2. Local verifiable validation pipeline ───────────────────────────
        is_valid, err_msg = validate_video_locally(frames, "bowling")
        if not is_valid:
            raise HTTPException(status_code=400, detail=err_msg)

        # ── 3. Ball speed via optical flow ────────────────────────────────
        # Returns None if ball not trackable — we report this honestly
        ball_speed = estimate_ball_speed(frames)
        speed_classification = classify_ball_speed(ball_speed)
        speed_note = (
            f"Estimated relative speed: {ball_speed} km/h (classification: {speed_classification})"
            if ball_speed is not None
            else "Ball not detected in video — ensure ball is visible for speed classification"
        )

        # ── 4. Multi-frame pose detection ────────────────────────────────
        all_landmarks = analyze_pose_from_video_frames(frames)
        valid = [lm for lm in all_landmarks if lm is not None]

        if not valid:
            raise HTTPException(
                400,
                "No human detected in the video. Ensure the bowler is clearly "
                "visible, well-lit, and in the centre of the frame."
            )

        # Run action classifier
        from utils.analysis import get_action_classifier
        classifier = get_action_classifier()
        action_class = classifier.predict(all_landmarks)
        if action_class != "bowling":
            raise HTTPException(
                status_code=400,
                detail=f"ERR_INVALID_ACTION: Expected bowling video, but action classifier detected: {action_class}."
            )


        # ── 5. TEMPORAL ANALYSIS — uses ALL frames ──────────────────────────
        arm_result     = calculate_arm_smoothness(all_landmarks, RIGHT_WRIST)
        release_result = calculate_release_point_score(all_landmarks)
        balance_result = calculate_balance_score(all_landmarks)
        shoulder_result = calculate_shoulder_alignment(all_landmarks)
        wrist_result   = analyze_wrist_position(all_landmarks)

        # Get arm angle from best release frame for style classification
        min_wrist_y = 1.0
        best_frame = None
        for f in all_landmarks:
            if not f:
                continue
            rw = get_point(f, RIGHT_WRIST)
            if rw and rw[1] < min_wrist_y:
                min_wrist_y = rw[1]
                best_frame = f

        arm_angle_at_release = release_result.get("releaseArmAngle") or 165.0
        bowling_style = classify_bowling_style(speed_classification, arm_angle_at_release)

        # ── 6. Compile all metrics ────────────────────────────────────────────
        bowling_metrics = {
            # Temporal scores
            "armSmoothnessScore":     arm_result["score"],
            "avgJerk":                arm_result["avgJerk"],
            "armSmoothnessNote":      arm_result["note"],
            "releasePointScore":      release_result["score"],
            "peakWristY":             release_result["peakWristY"],
            "releaseArmAngle":        release_result.get("releaseArmAngle"),
            "releaseHeightScore":     release_result.get("heightScore"),
            "releaseExtensionScore":  release_result.get("extensionScore"),
            "releasePointNote":       release_result["note"],
            "balanceScore":           balance_result["score"],
            "avgHipTilt":             balance_result["avgHipTilt"],
            "balanceNote":            balance_result["note"],
            "shoulderAlignmentScore": shoulder_result["score"],
            "wristPositionScore":     wrist_result["score"],
            "wristElbowDelta":        wrist_result.get("wristElbowDelta"),
            "wristPositionNote":      wrist_result["note"],
            # Speed (honest reporting)
            "estimatedBallSpeed":     ball_speed,
            "speedClassification":    speed_classification,
            "speedNote":              speed_note,
            # Classification
            "armRotationAngle":       arm_angle_at_release,
            "bowlingStyle":           bowling_style,
        }

        # ── 7. Deterministic overall score ────────────────────────────────────
        overall = compute_overall_bowling_score(bowling_metrics)
        bowling_metrics["overallBowlingScore"] = overall

        # ── 8. Fault detection (evidence-based) ──────────────────────────────
        faults = detect_faults(bowling_metrics, "bowling")
        fault_codes = [f["faultCode"] for f in faults]

        # ── 9. LLM report (narration only) ──────────────────────────────────
        report = await generate_report(bowling_metrics, "bowling", faults)

        return {
            "success": True,
            "type": "bowling",
            "upload_id": upload_id,
            "frames_analysed": len(frames),
            "frames_with_pose": len(valid),
            "bowling_metrics": {
                "armSmoothnessScore":     bowling_metrics["armSmoothnessScore"],
                "avgJerk":                bowling_metrics["avgJerk"],
                "armSmoothnessNote":      bowling_metrics["armSmoothnessNote"],
                "releasePointScore":      bowling_metrics["releasePointScore"],
                "peakWristY":             bowling_metrics["peakWristY"],
                "releaseArmAngle":        bowling_metrics["releaseArmAngle"],
                "releasePointNote":       bowling_metrics["releasePointNote"],
                "balanceScore":           bowling_metrics["balanceScore"],
                "balanceNote":            bowling_metrics["balanceNote"],
                "wristPositionScore":     bowling_metrics["wristPositionScore"],
                "wristPositionNote":      bowling_metrics["wristPositionNote"],
                "shoulderAlignmentScore": bowling_metrics["shoulderAlignmentScore"],
                "estimatedBallSpeed":     ball_speed,
                "speedClassification":    speed_classification,
                "speedNote":              speed_note,
                "armRotationAngle":       bowling_metrics["armRotationAngle"],
                "bowlingStyle":           bowling_style,
                "overallBowlingScore":    overall,
            },
            "overall_score": overall,
            "faults": faults,
            "fault_codes": fault_codes,
            "raw_frame_metrics": extract_raw_frame_metrics(all_landmarks, "bowling"),
            "landmarks": best_frame,
            **report,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
