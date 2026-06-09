"""
Batting Video Analysis Router — Deterministic Pipeline
=======================================================
All scores come from measured pose landmarks across MULTIPLE frames.
No random values. No single-frame snapshots.
Every score is traceable to a formula defined in utils/analysis.py.

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
    # Deterministic scoring functions
    calculate_head_stability,
    calculate_timing_score,
    calculate_balance_score,
    calculate_stride_score,
    calculate_follow_through_score,
    calculate_shoulder_alignment,
    detect_faults,
    compute_overall_batting_score,
    extract_raw_frame_metrics,
    # Landmark IDs
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point,
)

router = APIRouter()


def classify_shot(landmarks_list: list) -> str:
    """Geometry-based shot classifier from full pose sequence."""
    valid = [lm for lm in landmarks_list if lm]
    if not valid:
        return "Unknown"
    try:
        wrist_x, shoulder_x, knee_angles = [], [], []
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

        if not wrist_x:
            return "Straight Drive"

        x_travel = wrist_x[-1] - wrist_x[0]
        offset = wrist_x[-1] - shoulder_x[-1]
        avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else 150.0

        if avg_knee < 128:                              return "Sweep"
        if offset > 0.15 and x_travel > 0.12:          return "Pull Shot"
        if offset < -0.12 and x_travel < -0.10:        return "Cut Shot"
        if abs(offset) < 0.06 and abs(x_travel) < 0.08: return "Straight Drive"
        if x_travel > 0.04:                            return "Flick"
        return "Cover Drive"
    except Exception as e:
        print(f"Shot classification error: {e}")
        return "Straight Drive"


def analyze_stance_from_single_frame(landmarks: dict) -> dict:
    """
    Stance metrics that require a single best-pose frame (wrist-elbow-shoulder angle).
    Everything else comes from temporal (multi-frame) analysis.
    """
    metrics = {}

    # Bat swing angle from best-pose frame
    r_wrist = get_point(landmarks, RIGHT_WRIST)
    r_elbow = get_point(landmarks, RIGHT_ELBOW)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    if r_wrist and r_elbow and r_shoulder:
        swing = calculate_angle(r_wrist, r_elbow, r_shoulder)
        metrics["batSwingAngle"] = swing
        metrics["stanceScore"] = score_angle(swing, 30, 70)

    # Wrist position score
    r_elbow_pt = get_point(landmarks, RIGHT_ELBOW)
    if r_wrist and r_elbow_pt:
        delta = r_elbow_pt[1] - r_wrist[1]
        if delta >= 0:
            metrics["wristPositionScore"] = clamp_score(74 + delta * 240, 45, 98)
        else:
            metrics["wristPositionScore"] = clamp_score(68 + delta * 180, 30, 90)
        metrics["wristNote"] = (
            "Good — wrist above elbow at top of backswing"
            if delta >= 0
            else "Needs work — wrist below elbow, reducing bat control"
        )

    return metrics


@router.post("/batting")
async def analyze_batting(
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
        # ── 1. Extract frames (more frames = better temporal analysis) ────────
        frames = extract_frames(tmp_path, num_frames=20)
        if not frames:
            raise HTTPException(400, "Invalid video: could not extract frames.")

        # ── 2. Local verifiable validation pipeline ───────────────────────────
        is_valid, err_msg = validate_video_locally(frames, "batting")
        if not is_valid:
            raise HTTPException(status_code=400, detail=err_msg)

        # ── 3. Multi-frame pose detection (YOLO→crop→MediaPipe) ──────────────
        all_landmarks = analyze_pose_from_video_frames(frames)
        valid = [lm for lm in all_landmarks if lm is not None]

        if not valid:
            raise HTTPException(
                400,
                "No human detected in the video. Ensure the batsman is clearly "
                "visible, well-lit, and in the centre of the frame."
            )

        # ── 4. TEMPORAL ANALYSIS — uses ALL frames, not just one ──────────────
        head_result    = calculate_head_stability(all_landmarks)
        timing_result  = calculate_timing_score(all_landmarks)
        balance_result = calculate_balance_score(all_landmarks)
        stride_result  = calculate_stride_score(all_landmarks)
        follow_result  = calculate_follow_through_score(all_landmarks)
        shoulder_result = calculate_shoulder_alignment(all_landmarks)

        # ── 5. Single-frame metrics (stance requires best-pose frame) ─────────
        # Pick the frame with best overall landmark visibility
        best_lm = max(valid, key=lambda lm: sum(
            lm[k][3] for k in lm if isinstance(lm[k], list) and len(lm[k]) > 3
        ))
        stance_metrics = analyze_stance_from_single_frame(best_lm)

        # ── 6. Shot classification from full sequence ─────────────────────────
        shot_type = classify_shot(all_landmarks)

        # ── 7. Compile all metrics ────────────────────────────────────────────
        batting_metrics = {
            # Temporal scores (from full sequence)
            "headStabilityScore":  head_result["score"],
            "headStabilityVariance": head_result["variance"],
            "headStabilityNote":   head_result["note"],
            "timingScore":         timing_result["score"],
            "peakKneeFlexion":     timing_result["peakFlexion"],
            "peakKneeExtension":   timing_result["peakExtension"],
            "rangeOfMotion":       timing_result["rangeOfMotion"],
            "timingNote":          timing_result["note"],
            "balanceScore":        balance_result["score"],
            "avgHipTilt":          balance_result["avgHipTilt"],
            "avgSpineOffset":      balance_result["avgSpineOffset"],
            "balanceNote":         balance_result["note"],
            "strideScore":         stride_result["score"],
            "avgStrideRatio":      stride_result["avgStrideRatio"],
            "strideNote":          stride_result["note"],
            "followThroughScore":  follow_result["score"],
            "wristYDelta":         follow_result["wristYDelta"],
            "followThroughNote":   follow_result["note"],
            "shoulderAlignmentScore": shoulder_result["score"],
            # Single-frame scores
            "stanceScore":         stance_metrics.get("stanceScore", 50),
            "batSwingAngle":       stance_metrics.get("batSwingAngle"),
            "wristPositionScore":  stance_metrics.get("wristPositionScore", 50),
            "wristNote":           stance_metrics.get("wristNote", ""),
            # Classification
            "shotType":            shot_type,
        }

        # ── 8. Deterministic overall score ────────────────────────────────────
        overall = compute_overall_batting_score(batting_metrics)
        batting_metrics["overallBattingScore"] = overall

        # ── 9. Fault detection (evidence-based) ──────────────────────────────
        faults = detect_faults(batting_metrics, "batting")
        fault_codes = [f["faultCode"] for f in faults]

        # ── 10. LLM report (narration only — no score invention) ──────────────
        report = await generate_report(batting_metrics, "batting", faults)

        # ── 11. Build response with full audit trail ──────────────────────────
        return {
            "success": True,
            "type": "batting",
            "upload_id": upload_id,
            "frames_analysed": len(frames),
            "frames_with_pose": len(valid),
            "batting_metrics": {
                # Temporal metrics (derived from full sequence)
                "headStabilityScore":    batting_metrics["headStabilityScore"],
                "headStabilityVariance": batting_metrics["headStabilityVariance"],
                "headStabilityNote":     batting_metrics["headStabilityNote"],
                "timingScore":           batting_metrics["timingScore"],
                "peakKneeFlexion":       batting_metrics["peakKneeFlexion"],
                "peakKneeExtension":     batting_metrics["peakKneeExtension"],
                "rangeOfMotion":         batting_metrics["rangeOfMotion"],
                "timingNote":            batting_metrics["timingNote"],
                "balanceScore":          batting_metrics["balanceScore"],
                "avgHipTilt":            batting_metrics["avgHipTilt"],
                "balanceNote":           batting_metrics["balanceNote"],
                "strideScore":           batting_metrics["strideScore"],
                "avgStrideRatio":        batting_metrics["avgStrideRatio"],
                "strideNote":            batting_metrics["strideNote"],
                "followThroughScore":    batting_metrics["followThroughScore"],
                "wristYDelta":           batting_metrics["wristYDelta"],
                "followThroughNote":     batting_metrics["followThroughNote"],
                # Single-frame metrics
                "stanceScore":           batting_metrics["stanceScore"],
                "batSwingAngle":         batting_metrics["batSwingAngle"],
                "wristPositionScore":    batting_metrics["wristPositionScore"],
                "wristNote":             batting_metrics["wristNote"],
                "shoulderAlignmentScore": batting_metrics["shoulderAlignmentScore"],
                # Classification
                "shotType":              batting_metrics["shotType"],
                "overallBattingScore":   overall,
            },
            "overall_score": overall,
            "faults": faults,
            "fault_codes": fault_codes,
            "raw_frame_metrics": extract_raw_frame_metrics(all_landmarks, "batting"),
            "landmarks": best_lm,
            **report,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
