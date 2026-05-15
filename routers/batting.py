"""
Batting Video Analysis Router
MediaPipe + YOLO pipeline:
  Frame sampling → blur filter → YOLO crop → MediaPipe pose → feature extraction
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

import httpx

from utils.analysis import (
    extract_frames, analyze_pose_from_video_frames,
    calculate_angle, score_angle, generate_report, clamp_score,
    validate_cricket_content_async,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, get_point,
)

router = APIRouter()


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _head_position_score(head_offset: float) -> int:
    return clamp_score(98 - head_offset * 280, 40, 98)


def _follow_through_score(wrist_y: float, nose_y: float) -> int:
    delta = nose_y - wrist_y
    if delta >= 0:
        return clamp_score(76 + delta * 220, 50, 98)
    return clamp_score(68 + delta * 180, 35, 90)


def classify_shot(landmarks_list: list) -> str:
    """Geometry-based shot classifier from pose sequence."""
    valid = [lm for lm in landmarks_list if lm]
    if not valid:
        return "Straight Drive"
    try:
        wrist_x, shoulder_x, knee_angles = [], [], []
        for lm in valid:
            rw = get_point(lm, RIGHT_WRIST)
            rs = get_point(lm, RIGHT_SHOULDER)
            rh = get_point(lm, RIGHT_HIP)
            rk = get_point(lm, RIGHT_KNEE)
            ra = get_point(lm, RIGHT_ANKLE)
            if rw and rs:
                wrist_x.append(rw[0]); shoulder_x.append(rs[0])
            if rh and rk and ra:
                knee_angles.append(calculate_angle(rh, rk, ra))

        if not wrist_x:
            return "Straight Drive"
        x_travel = wrist_x[-1] - wrist_x[0]
        offset   = wrist_x[-1] - shoulder_x[-1]
        avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else 150.0

        if avg_knee < 128:            return "Sweep"
        if offset > 0.15 and x_travel > 0.12:  return "Pull Shot"
        if offset < -0.12 and x_travel < -0.10: return "Cut Shot"
        if abs(offset) < 0.06 and abs(x_travel) < 0.08: return "Straight Drive"
        if x_travel > 0.04:          return "Flick"
        return "Cover Drive"
    except Exception as e:
        print(f"Shot classification error: {e}")
        return "Straight Drive"


def analyze_batting_landmarks(landmarks: dict) -> dict:
    metrics = {}

    nose       = get_point(landmarks, NOSE)
    l_shoulder = get_point(landmarks, LEFT_SHOULDER)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    if nose and l_shoulder and r_shoulder:
        mid_sx = (l_shoulder[0] + r_shoulder[0]) / 2
        offset = abs(nose[0] - mid_sx)
        metrics["headOffset"] = round(offset, 4)
        if offset < 0.05:
            metrics["headPosition"] = "Excellent — aligned over off stump"
        elif offset < 0.12:
            metrics["headPosition"] = "Good — slight lateral movement"
        else:
            metrics["headPosition"] = "Needs work — head falling over"
        metrics["headPositionScore"] = _head_position_score(offset)

    r_wrist    = get_point(landmarks, RIGHT_WRIST)
    r_elbow    = get_point(landmarks, RIGHT_ELBOW)
    r_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    if r_wrist and r_elbow and r_shoulder:
        swing = calculate_angle(r_wrist, r_elbow, r_shoulder)
        metrics["batSwingAngle"] = swing
        metrics["stanceScore"]   = score_angle(swing, 30, 70)

    l_hip   = get_point(landmarks, LEFT_HIP)
    l_knee  = get_point(landmarks, LEFT_KNEE)
    l_ankle = get_point(landmarks, LEFT_ANKLE)
    if l_hip and l_knee and l_ankle:
        knee = calculate_angle(l_hip, l_knee, l_ankle)
        metrics["kneeBendAngle"] = knee
        metrics["timingScore"]   = score_angle(knee, 120, 160)

    if r_wrist and nose:
        metrics["followThroughScore"] = _follow_through_score(r_wrist[1], nose[1])

    return metrics


# ─── Route ────────────────────────────────────────────────────────────────────

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
        # ── 1. Extract frames (blur-filtered) ────────────────────────────────
        frames = extract_frames(tmp_path, num_frames=30)
        if not frames:
            raise HTTPException(400, "Invalid video: could not extract frames.")

        # ── 2. Cricket content validation (non-blocking — skipped if no key) ─
        mid_frame = frames[len(frames) // 2]
        if not await validate_cricket_content_async(mid_frame):
            raise HTTPException(
                400,
                "Content validation failed. Please upload a cricket batting video."
            )

        # ── 3. Multi-frame pose detection (YOLO→crop→MediaPipe) ──────────────
        all_landmarks = analyze_pose_from_video_frames(frames)
        valid = [lm for lm in all_landmarks if lm is not None]

        if not valid:
            raise HTTPException(
                400,
                "No human detected in the video. Ensure the batsman is clearly "
                "visible, well-lit, and in the centre of the frame."
            )

        # ── 4. Use the most information-rich frame for primary metrics ────────
        mid_lm = valid[len(valid) // 2]
        batting_metrics = analyze_batting_landmarks(mid_lm)
        batting_metrics["shotType"] = classify_shot(valid)

        # ── 5. Score aggregation ──────────────────────────────────────────────
        scores = [
            batting_metrics.get("stanceScore", 70),
            batting_metrics.get("headPositionScore", 70),
            batting_metrics.get("timingScore", 70),
            batting_metrics.get("followThroughScore", 70),
        ]
        overall = round(sum(scores) / len(scores))
        batting_metrics["overallBattingScore"] = overall

        # ── 6. AI coaching report ─────────────────────────────────────────────
        report = await generate_report(batting_metrics, "batting")

        return {
            "success": True,
            "type": "batting",
            "upload_id": upload_id,
            "frames_analysed": len(frames),
            "frames_with_pose": len(valid),
            "batting_metrics": {
                "stanceScore":        batting_metrics.get("stanceScore", 70),
                "batSwingAngle":      batting_metrics.get("batSwingAngle", 45.0),
                "headPosition":       batting_metrics.get("headPosition", ""),
                "headPositionScore":  batting_metrics.get("headPositionScore", 70),
                "timingScore":        batting_metrics.get("timingScore", 70),
                "followThroughScore": batting_metrics.get("followThroughScore", 70),
                "shotType":           batting_metrics.get("shotType", "Cover Drive"),
                "overallBattingScore": overall,
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
