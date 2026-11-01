"""
Engine tests on synthetic pose sequences (no video or models needed).
Run: venv/Scripts/python -m pytest -q
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.analysis import FAULT_COACHING, _generate_rule_based_report, get_point, range_score
from utils.fault_detector import FAULT_RULES, FaultDetector
from utils.pipeline import AnalysisError, batting_from_poses, bowling_from_poses

FPS = 25.0


def _lerp(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _frame(points: dict, aspect: float = 1.0) -> dict:
    """Landmark dict in MediaPipe layout from aspect-correct (x, y) points."""
    lm = {k: [round(x / aspect, 4), round(y, 4), 0.0, 0.99] for k, (x, y) in points.items()}
    lm["_aspect"] = aspect
    return lm


def _body(hip=(0.5, 0.6), lean=(0.02, 0.0)):
    sh = (hip[0] + lean[0], hip[1] - 0.25)
    return {
        0: (sh[0] + 0.01, sh[1] - 0.08),                  # nose
        11: (sh[0] - 0.01, sh[1]), 12: (sh[0] + 0.01, sh[1]),
        23: (hip[0] - 0.01, hip[1]), 24: (hip[0] + 0.01, hip[1]),
    }


def make_batting(stride=True, steady_head=True, follow_through=True, plant_frame=35, impact_frame=45):
    """Right-handed batter, side-on, facing +x. Front (left) foot strides, bat swings down at impact."""
    frames = []
    for t in range(75):
        p = _body()
        p[0] = (p[0][0] + (0 if steady_head else 0.04 * math.sin(t)), p[0][1] + (0 if steady_head else 0.04 * math.cos(t)))
        back_ank, front_start, front_end = (0.42, 0.95), (0.55, 0.95), (0.75, 0.95)
        front_ank = _lerp(front_start, front_end, (t - (plant_frame - 12)) / 12) if stride else front_start
        p[28], p[27] = back_ank, front_ank
        p[26] = (back_ank[0] + 0.02, 0.78)
        p[25] = ((front_ank[0] + 0.5) / 2 + 0.03, 0.76)   # bent front knee
        # hands: rest → backlift → fast downswing into impact → follow-through
        rest, top, contact, finish = (0.46, 0.5), (0.40, 0.30), (0.66, 0.62), (0.62, 0.15)
        if t < impact_frame - 15:
            hands = _lerp(rest, top, (t - (impact_frame - 25)) / 10)
        elif t <= impact_frame:
            hands = _lerp(top, contact, ((t - (impact_frame - 15)) / 15) ** 3)
        else:
            hands = _lerp(contact, finish if follow_through else (0.70, 0.60), (t - impact_frame) / 10)
        p[15] = p[16] = hands
        p[13] = p[14] = _lerp(p[11], hands, 0.5)
        frames.append(_frame(p))
    return frames


def make_bowling(jerky=False, arm_len=0.3):
    """Right-arm bowler: bowling wrist swings in a circle about the shoulder, release at the top."""
    frames = []
    n = 60
    for t in range(n):
        p = _body(hip=(0.5, 0.6))
        sh = p[12]
        u = t / (n - 1)
        theta = 2 * math.pi * (u ** 2)                    # smooth acceleration through the arc
        if jerky:
            theta += 0.35 * math.sin(10 * math.pi * u)
        wrist = (sh[0] - arm_len * math.sin(theta), sh[1] + arm_len * math.cos(theta))
        p[16] = wrist
        p[14] = _lerp(sh, wrist, 0.5)
        p[15] = (p[11][0], p[11][1] + 0.2)
        p[13] = (p[11][0], p[11][1] + 0.1)
        p[27], p[28] = (0.62, 0.95), (0.40, 0.95)          # left foot in front
        p[25], p[26] = (0.58, 0.78), (0.44, 0.78)
        p[0] = (0.55, p[0][1])
        frames.append(_frame(p))
    return frames


# ─── geometry helpers ─────────────────────────────────────────────────────────

def test_get_point_applies_aspect():
    lm = {0: [0.5, 0.25, 0, 1], "_aspect": 0.5625}           # portrait 9:16
    assert get_point(lm, 0) == pytest.approx([0.28125, 0.25])


def test_range_score_shape():
    assert range_score(150, 140, 160, 20) == 100
    assert range_score(140, 140, 160, 20) == 86
    assert range_score(180, 140, 160, 20) == 26
    assert range_score(None, 0, 1, 1) is None


# ─── batting ──────────────────────────────────────────────────────────────────

def test_batting_good_shot_scores_well():
    metrics, faults, impact, _ = batting_from_poses(make_batting(), FPS, "right-handed")
    assert 40 <= impact <= 46
    assert metrics["plantToImpactMs"] is not None and 80 <= metrics["plantToImpactMs"] <= 500
    assert metrics["headStabilityScore"] >= 80
    assert metrics["followThroughScore"] >= 70
    assert metrics["overallBattingScore"] >= 60
    codes = {f["faultCode"] for f in faults}
    assert "HEAD_INSTABILITY" not in codes and "SHORT_FOLLOW_THROUGH" not in codes


def test_batting_head_movement_and_short_follow_through_are_faults():
    metrics, faults, _, _ = batting_from_poses(make_batting(steady_head=False, follow_through=False), FPS)
    codes = {f["faultCode"] for f in faults}
    assert "HEAD_INSTABILITY" in codes
    assert "SHORT_FOLLOW_THROUGH" in codes
    good, _, _, _ = batting_from_poses(make_batting(), FPS)
    assert metrics["overallBattingScore"] < good["overallBattingScore"]


def test_batting_late_plant_flagged():
    metrics, faults, _, _ = batting_from_poses(make_batting(plant_frame=46, impact_frame=45), FPS)
    assert metrics["plantToImpactMs"] < 50
    assert "LATE_TIMING" in {f["faultCode"] for f in faults}


def test_batting_metrics_independent_of_camera_distance():
    near = batting_from_poses(make_batting(), FPS)[0]
    far_frames = []
    for lm in make_batting():
        far_frames.append({k: ([v[0] * 0.5 + 0.25, v[1] * 0.5 + 0.25, v[2], v[3]] if isinstance(v, list) else v)
                           for k, v in lm.items()})
    far = batting_from_poses(far_frames, FPS)[0]
    for key in ("headStabilityScore", "timingScore", "followThroughScore", "strideScore"):
        assert abs(near[key] - far[key]) <= 3, key


def test_no_player_raises():
    with pytest.raises(AnalysisError, match="ERR_NO_HUMAN_DETECTED"):
        batting_from_poses([None] * 30, FPS)


# ─── bowling ──────────────────────────────────────────────────────────────────

def test_bowling_high_smooth_release():
    metrics, faults, release = bowling_from_poses(make_bowling(), 30.0, "right-arm-fast")
    assert metrics["releaseHeight"] >= 1.0
    assert metrics["releasePointScore"] >= 80
    assert metrics["armSmoothnessScore"] >= 75
    assert metrics["bowlingArm"] == "right"
    assert "LOW_RELEASE" not in {f["faultCode"] for f in faults}


def test_bowling_detects_arm_without_profile_and_jerky_action():
    smooth = bowling_from_poses(make_bowling(), 30.0, None)[0]
    jerky = bowling_from_poses(make_bowling(jerky=True), 30.0, None)[0]
    assert jerky["bowlingArm"] == "right"
    assert jerky["armSmoothnessScore"] < smooth["armSmoothnessScore"]


def test_bowling_low_arm_rejected():
    with pytest.raises(AnalysisError, match="ERR_INVALID_ACTION"):
        bowling_from_poses(make_bowling(arm_len=0.05), 30.0, "right-arm-fast")


def test_ball_speed_unavailable_is_reported_honestly():
    metrics, _, _ = bowling_from_poses(make_bowling(), 30.0, "right-arm-fast")
    assert metrics["estimatedBallSpeed"] is None
    assert metrics["speedClassification"] == "Undetected"


# ─── faults & report ──────────────────────────────────────────────────────────

def test_every_fault_code_has_coaching_text():
    for rules in FAULT_RULES.values():
        for code, *_ in rules:
            assert code in FAULT_COACHING, code


def test_missing_metrics_never_fire_faults():
    assert FaultDetector().detect({}, "batting") == []


def test_rule_based_report_uses_faults():
    metrics, faults, _, _ = batting_from_poses(make_batting(steady_head=False), FPS)
    report = _generate_rule_based_report(metrics, "batting", faults)
    assert any("Head movement" in w for w in report["weaknesses"])
    assert report["training_drills"]


def test_front_on_view_skips_footwork_instead_of_guessing():
    frames = []
    for lm in make_batting(plant_frame=46, impact_frame=45):    # would be "late" if measured
        lm = dict(lm)
        lm[11] = [lm[11][0] - 0.08, lm[11][1], 0.0, 0.99]        # shoulders appear wide: camera front-on
        lm[12] = [lm[12][0] + 0.08, lm[12][1], 0.0, 0.99]
        frames.append(lm)
    metrics, faults, _, warnings = batting_from_poses(frames, FPS)
    assert metrics["cameraView"] == "front-on"
    assert metrics["plantToImpactMs"] is None and metrics["strideScore"] is None
    assert "LATE_TIMING" not in {f["faultCode"] for f in faults}
    assert warnings


def test_face_turned_away_skips_head_stability():
    frames = []
    for lm in make_batting(steady_head=False):
        lm = dict(lm)
        lm[0] = [lm[0][0], lm[0][1], 0.5, 0.99]                   # nose behind the shoulders (z)
        frames.append(lm)
    metrics, faults, _, _ = batting_from_poses(frames, FPS)
    assert metrics["headStabilityScore"] is None
    assert "HEAD_INSTABILITY" not in {f["faultCode"] for f in faults}
