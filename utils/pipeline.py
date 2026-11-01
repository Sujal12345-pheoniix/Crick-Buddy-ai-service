"""
CrickBuddy analysis engine — batting, bowling and posture pipelines
====================================================================

Video pipeline
  1. Locate the action. Short clips are used whole; for longer clips a coarse pose
     pass finds the moment of peak hand/arm movement relative to the body.
  2. Read a short window (~3 s) around it at a steady frame rate, keeping the
     original aspect ratio.
  3. Pose per frame: YOLOv8 finds and tracks the player (re-detected every other
     frame), MediaPipe estimates 33 landmarks on the padded crop.
  4. Landmarks are converted to aspect-correct coordinates, short gaps are
     interpolated and the tracks lightly smoothed.
  5. Metrics are measured on the time series around the key event (bat-swing peak
     for batting, release for bowling), normalised by the player's torso length so
     results do not depend on how far away the camera is.

Everything here is CPU-bound and synchronous; routers run it in a worker thread.
ENGINE_LOCK serialises access because the MediaPipe graph is not thread-safe.
"""

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.analysis import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    analyze_pose_from_image,
    clamp_score,
    detect_balls,
    detect_people,
    detect_person_box,
    match_box,
    people_arm_heights,
    extract_raw_frame_metrics,
    pose_on_frame,
    range_score,
    resize_keep_aspect,
)
from utils.fault_detector import FaultDetector

ENGINE_LOCK = threading.Lock()

WINDOW_SECONDS = float(os.getenv("ANALYSIS_WINDOW_SECONDS", "3.5"))
ACTION_FPS = {"batting": 25.0, "bowling": 30.0, "posture": 10.0}
COARSE_SAMPLES = 30
BOWLER_SEARCH_RATE = 6            # frames per second scanned with YOLOv8-pose to find the bowler
BOWLER_SEARCH_MAX = 160
DETECT_EVERY = 2                  # re-run YOLO person detection every N frames
VISIBILITY_MIN = 0.05            # keep MediaPipe's estimate for briefly occluded joints (hands behind the bat)
CLIP_MAX_SIDE = int(os.getenv("ANALYSIS_MAX_SIDE", "800"))  # larger frames help distant players; lower to save RAM
SEEK_THRESHOLD = 400              # read sequentially below this many frames — seeking re-decodes from keyframes
MAX_GAP_FRAMES = 3
TIME_BUDGET_S = float(os.getenv("ANALYSIS_TIME_BUDGET_SECS", "400"))
NOSE_TO_ANKLE_M = 1.55            # typical adult nose→ankle height, for pace estimation
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

_fault_detector = FaultDetector()


class AnalysisError(Exception):
    """A user-facing problem with the uploaded media (returned as HTTP 400)."""


@dataclass
class EngineResult:
    metrics: Dict[str, Any]
    faults: List[Dict[str, Any]]
    landmarks: Optional[dict]
    raw_frame_metrics: List[dict]
    meta: Dict[str, Any] = field(default_factory=dict)


class _Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds

    def check(self):
        if time.monotonic() > self.end:
            raise AnalysisError("ERR_TIMEOUT: Analysis took too long. Upload a shorter clip (under 30 seconds) "
                                "showing a single shot or delivery.")


# ═══════════════════════════════════════════════════════════════════════════════
# Video reading
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Clip:
    frames: List[np.ndarray]
    fps: float              # effective fps of `frames`
    source_fps: float
    start_index: int        # source frame index of frames[0]
    stride: int
    duration_s: float
    seed: Optional[Tuple[int, List[float]]] = None   # (frame index, player box) to start tracking from


def _open(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise AnalysisError("ERR_INVALID_VIDEO: Could not open the video file. Use MP4 or MOV.")
    return cap


def _video_props(path: str) -> Tuple[float, int]:
    import cv2
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:  # some containers (webm) do not report a count
        count = 0
        while cap.grab():
            count += 1
    cap.release()
    if not fps or not np.isfinite(fps) or fps < 5 or fps > 480:
        fps = 30.0
    return float(fps), count


def _read_at(path: str, indices: List[int]):
    """
    Yield (index, frame) at the given source indices, decoded in one sequential pass
    (no seeking). A generator, so a long scan never holds all its frames in memory.
    """
    wanted = set(int(i) for i in indices)
    last = max(wanted)
    cap = _open(path)
    try:
        idx = 0
        while idx <= last and cap.grab():
            if idx in wanted:
                ok, frm = cap.retrieve()
                if ok and frm is not None:
                    yield idx, resize_keep_aspect(frm)
            idx += 1
    finally:
        cap.release()


def _read_range(path: str, start: int, count: int, stride: int,
                max_side: int = CLIP_MAX_SIDE) -> List[np.ndarray]:
    """Read `count` frames from `start`, keeping every `stride`-th frame."""
    import cv2
    cap = _open(path)
    if start > SEEK_THRESHOLD:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    else:
        for _ in range(start):
            if not cap.grab():
                break
    frames, i = [], 0
    while len(frames) < count:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, frm = cap.retrieve()
            if not ok or frm is None:
                break
            frames.append(resize_keep_aspect(frm, max_side))
        i += 1
    cap.release()
    return frames


def _pose_activity_center(path: str, frame_count: int, source_fps: float,
                          deadline: _Deadline) -> Optional[int]:
    """
    Coarse pass over a long video: estimate pose on evenly spaced frames and
    return the source index where wrists move fastest relative to the shoulders
    (bat swing / bowling arm). Relative motion ignores camera panning.
    """
    # ~3 samples per second of video, between 12 and COARSE_SAMPLES
    n = int(np.clip(frame_count / source_fps * 3, 12, COARSE_SAMPLES))
    indices = np.linspace(0, frame_count - 1, num=min(n, frame_count), dtype=int).tolist()
    samples = []
    for idx, frm in _read_at(path, indices):
        deadline.check()
        lm = None
        # Skip fades/black frames: they give confident-looking but meaningless poses.
        if float(frm.mean()) >= 40:
            box = detect_person_box(frm)
            lm = pose_on_frame(frm, box) if box is not None else None
            if lm and _core_visibility(lm) < 0.5:
                lm = None
        samples.append((idx, lm))

    activity, centers = [], []
    for (i0, a), (i1, b) in zip(samples, samples[1:]):
        centers.append((i0 + i1) // 2)
        if not a or not b:
            activity.append(np.nan)
            continue
        pa, pb = _frame_points(a), _frame_points(b)
        torso = np.nanmean([_torso_len_single(pa), _torso_len_single(pb)])
        if not np.isfinite(torso) or torso <= 0:
            continue
        move = 0.0
        for w in (LEFT_WRIST, RIGHT_WRIST):
            ra = pa[w] - (pa[LEFT_SHOULDER] + pa[RIGHT_SHOULDER]) / 2
            rb = pb[w] - (pb[LEFT_SHOULDER] + pb[RIGHT_SHOULDER]) / 2
            d = np.linalg.norm(rb - ra) / torso
            if np.isfinite(d):
                move = max(move, d)
        activity.append(move)
    # The first big burst is the shot/delivery; later bursts are usually the
    # player relaxing, turning or running.
    peak = _first_major_peak(np.array(activity, dtype=float))
    return centers[peak] if peak is not None else None


def _arm_height(lm: dict) -> float:
    """Highest wrist above its shoulder, in torso lengths (NaN if not measurable)."""
    pts = _frame_points(lm)
    torso = _torso_len_single(pts)
    if not np.isfinite(torso) or torso <= 0:
        return float("nan")
    h = [(pts[s][1] - pts[w][1]) / torso for s, w in ((LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST))]
    return float(np.nanmax(h)) if np.any(np.isfinite(h)) else float("nan")


def _find_bowler(samples: List[Tuple[int, np.ndarray]], deadline: _Deadline,
                 index_rate: float) -> Optional[Tuple[int, List[float], int]]:
    """
    In match footage the biggest person is often the keeper or batter, so the bowler
    is picked by behaviour — an arm above the head (YOLOv8-pose sees everyone in
    one pass). The highest arm can be the batter's swing, which comes just after the
    delivery, so the earliest overhead arm in the 2 s before the highest one (often the
    bowler's gather) is
    taken. `index_rate` is sample indices per second. Returns (index, box, frame width).
    """
    per_frame = []
    for idx, frm in samples:
        deadline.check()
        if float(frm.mean()) < 40:
            continue
        people = [p for p in people_arm_heights(frm) if np.isfinite(p[1])]
        if people:
            box, h = max(people, key=lambda p: p[1])
            per_frame.append((h, idx, box, frm.shape[1]))
    if not per_frame:
        return None
    top = max(per_frame, key=lambda b: b[0])
    if top[0] < 0.25:
        return None
    # YOLO-pose keypoints are coarse, so "overhead" is judged relative to the peak.
    earlier = [b for b in per_frame
               if b[0] >= max(0.1, 0.6 * top[0]) and top[1] - 2.0 * index_rate <= b[1] < top[1]]
    pick = min(earlier, key=lambda b: b[1]) if earlier else top
    return pick[1], pick[2], pick[3]


def read_action_clip(path: str, action: str, deadline: _Deadline) -> Clip:
    source_fps, frame_count = _video_props(path)
    if frame_count < 3:
        raise AnalysisError("ERR_INVALID_VIDEO: The video has too few frames to analyse.")

    stride = max(1, int(round(source_fps / ACTION_FPS[action])))
    eff_fps = source_fps / stride
    duration = frame_count / source_fps
    window_src = int(round(WINDOW_SECONDS * source_fps))

    bowler = None   # (source index, box, width of the frame the box was found in)
    if frame_count <= window_src * 1.5:
        start, length = 0, frame_count
    else:
        if action == "bowling":
            n = int(np.clip(frame_count / source_fps * BOWLER_SEARCH_RATE, 12, BOWLER_SEARCH_MAX))
            idx = np.linspace(0, frame_count - 1, num=n, dtype=int).tolist()
            bowler = _find_bowler(_read_at(path, idx), deadline, source_fps)
        center = bowler[0] if bowler else _pose_activity_center(path, frame_count, source_fps, deadline)
        if center is None:
            center = frame_count // 2
        # Keep more context before the event (stride, backlift, run-up) than after.
        start = int(np.clip(center - 0.55 * window_src, 0, frame_count - window_src))
        length = window_src

    frames = _read_range(path, start, int(math.ceil(length / stride)), stride, CLIP_MAX_SIDE)
    if len(frames) < 3:
        raise AnalysisError("ERR_INVALID_VIDEO: Could not decode frames from the video.")

    seed = None
    if action == "bowling":
        if bowler is None:   # short clip: search the window itself
            step = max(1, int(round(eff_fps / BOWLER_SEARCH_RATE)))
            found = _find_bowler([(i, f) for i, f in enumerate(frames[::step])], deadline,
                                 source_fps / (step * stride))
            if found:
                bowler = (start + found[0] * step * stride, found[1], found[2])
        if bowler:
            k = int(np.clip(round((bowler[0] - start) / stride), 0, len(frames) - 1))
            scale = frames[0].shape[1] / bowler[2]
            seed = (k, [v * scale for v in bowler[1]])
    return Clip(frames, eff_fps, source_fps, start, stride, duration, seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Pose tracking and landmark arrays
# ═══════════════════════════════════════════════════════════════════════════════

def track_poses(frames: List[np.ndarray], deadline: _Deadline,
                seed: Optional[Tuple[int, List[float]]] = None) -> List[Optional[dict]]:
    """
    Pose for every frame, following one player. Without a seed the largest person
    in the first frame is followed; with a seed (frame index, box) tracking runs
    forward and backward from that frame so it stays on the chosen player.
    """
    poses: List[Optional[dict]] = [None] * len(frames)

    def run(order, box):
        for n, i in enumerate(order):
            deadline.check()
            frm = frames[i]
            if box is None:
                box = detect_person_box(frm)
            elif n % DETECT_EVERY == 0:
                box = match_box(detect_people(frm), box) or box
            lm = pose_on_frame(frm, box) if box is not None else None
            if lm is None and seed is None:
                lm = pose_on_frame(frm, None)
            poses[i] = lm

    if seed is None:
        run(range(len(frames)), None)
    else:
        k, box = seed
        run(range(k, len(frames)), box)
        run(range(k - 1, -1, -1), box)
    return poses


def _frame_points(lm: dict) -> np.ndarray:
    """(33, 2) aspect-correct points for one landmark dict; NaN where low visibility."""
    pts = np.full((33, 2), np.nan)
    aspect = float(lm.get("_aspect", 1.0) or 1.0)
    for k in range(33):
        v = lm.get(k)
        if v is None:
            v = lm.get(str(k))
        if v is None or (len(v) > 3 and v[3] < VISIBILITY_MIN):
            continue
        pts[k] = (v[0] * aspect, v[1])
    return pts


def _face_visible(poses: List[Optional[dict]], max_depth: float = 0.15) -> np.ndarray:
    """
    (T,) True where the player faces the camera enough for the face to be seen.
    MediaPipe reports high visibility for a guessed nose even from behind, but its
    depth estimate puts that nose clearly behind the shoulders (larger z).
    """
    out = np.zeros(len(poses), dtype=bool)
    for t, lm in enumerate(poses):
        if lm and all(isinstance(lm.get(k), list) for k in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)):
            depth = lm[NOSE][2] - (lm[LEFT_SHOULDER][2] + lm[RIGHT_SHOULDER][2]) / 2
            out[t] = depth < max_depth
    return out


def _core_visibility(lm: dict) -> float:
    """Mean MediaPipe visibility of shoulders, hips and wrists."""
    ids = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_WRIST, RIGHT_WRIST)
    vals = [lm[k][3] for k in ids if isinstance(lm.get(k), list) and len(lm[k]) > 3]
    return float(np.mean(vals)) if vals else 0.0


def _torso_len_single(pts: np.ndarray) -> float:
    sh = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) / 2
    hp = (pts[LEFT_HIP] + pts[RIGHT_HIP]) / 2
    return float(np.linalg.norm(sh - hp))


def _fill_gaps(s: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    out = s.copy()
    valid = np.where(np.isfinite(s))[0]
    for a, b in zip(valid, valid[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [s[a], s[b]])
    return out


def _smooth(s: np.ndarray) -> np.ndarray:
    out = s.copy()
    for t in range(len(s)):
        if np.isfinite(s[t]):
            out[t] = np.nanmean(s[max(0, t - 1):t + 2])
    return out


def landmark_array(poses: List[Optional[dict]], smooth: bool = True) -> np.ndarray:
    """(T, 33, 2) array of aspect-correct landmark positions."""
    P = np.full((len(poses), 33, 2), np.nan)
    for t, lm in enumerate(poses):
        if lm:
            P[t] = _frame_points(lm)
    for k in range(33):
        for ax in range(2):
            s = _fill_gaps(P[:, k, ax])
            P[:, k, ax] = _smooth(s) if smooth else s
    return P


# ─── Vector helpers over (T, 2) series ────────────────────────────────────────

def _pt(P, idx):
    return P[:, idx, :]


def _mid(a, b):
    return (a + b) / 2


def _angle3(a, b, c):
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _lean_from_vertical(bottom, top):
    """Angle of bottom→top from straight up, degrees (0 = upright)."""
    v = top - bottom
    return np.degrees(np.arctan2(np.abs(v[..., 0]), -v[..., 1]))


def _line_tilt(a, b):
    """Tilt of the line a–b from horizontal, degrees (0–90)."""
    v = b - a
    ang = np.abs(np.degrees(np.arctan2(v[..., 1], v[..., 0])))
    return np.minimum(ang, 180 - ang)


def _speed(series, fps):
    return np.linalg.norm(np.gradient(series, axis=0), axis=-1) * fps


def _nanargmax(a) -> Optional[int]:
    a = np.asarray(a, dtype=float)
    if not np.any(np.isfinite(a)):
        return None
    return int(np.nanargmax(a))


def _val(a, i):
    if i is None or i < 0 or i >= len(a):
        return None
    v = a[i]
    return float(v) if np.isfinite(v) else None


def _r(v, nd=1):
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


def _win(T, a, b):
    return slice(max(0, int(a)), min(T, int(b)))


def _torso(P) -> float:
    sh = _mid(_pt(P, LEFT_SHOULDER), _pt(P, RIGHT_SHOULDER))
    hp = _mid(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP))
    t = np.nanmedian(np.linalg.norm(sh - hp, axis=-1))
    return float(t) if np.isfinite(t) and t > 0 else float("nan")


def _weighted(parts: List[Tuple[Optional[float], float]]) -> Optional[int]:
    got = [(v, w) for v, w in parts if v is not None]
    if not got:
        return None
    return clamp_score(sum(v * w for v, w in got) / sum(w for _, w in got))


def _require_pose(poses, min_ratio=0.4):
    found = sum(1 for p in poses if p)
    if found == 0 or found / len(poses) < min_ratio:
        raise AnalysisError(
            f"ERR_NO_HUMAN_DETECTED: A player was found in only {found}/{len(poses)} frames. Film side-on "
            "with the whole body in frame, in good light, and keep the camera steady.")


def _best_landmarks(poses, preferred: Optional[int]) -> Optional[dict]:
    if preferred is not None and 0 <= preferred < len(poses) and poses[preferred]:
        return poses[preferred]
    valid = [p for p in poses if p]
    if not valid:
        return None
    return max(valid, key=lambda lm: sum(v[3] for k, v in lm.items() if isinstance(v, list) and len(v) > 3))


# ═══════════════════════════════════════════════════════════════════════════════
# Batting
# ═══════════════════════════════════════════════════════════════════════════════

def _camera_view(P: np.ndarray, torso: float) -> str:
    """
    'side-on' when the shoulders overlap in the image (camera square of the player),
    'front-on' when they appear wide (camera in front of or behind the player).
    """
    width = np.linalg.norm(_pt(P, LEFT_SHOULDER) - _pt(P, RIGHT_SHOULDER), axis=-1) / torso
    early = width[: max(3, len(width) // 3)]
    w = np.nanmedian(early) if np.any(np.isfinite(early)) else np.nanmedian(width)
    if not np.isfinite(w):
        return "unknown"
    if w < 0.45:
        return "side-on"
    return "front-on" if w > 0.6 else "oblique"


def _first_stride(ank_speed: np.ndarray, feet_gap: np.ndarray, torso: float,
                  min_speed: float = 1.2, min_move: float = 0.25) -> Tuple[Optional[int], Optional[int]]:
    """
    (start, plant) of the first front-foot movement that changes the foot separation
    by at least `min_move` torso-lengths. Small trigger movements are skipped.
    """
    moving = np.isfinite(ank_speed) & (ank_speed > min_speed)
    t, T = 0, len(ank_speed)
    while t < T:
        if not moving[t]:
            t += 1
            continue
        a = t
        while t < T and moving[t]:
            t += 1
        before, after = feet_gap[max(a - 1, 0)], feet_gap[min(t, T - 1)]
        if np.all(np.isfinite(before)) and np.all(np.isfinite(after)) and                 np.linalg.norm(after - before) / torso >= min_move:
            return a, min(t, T - 1)
    return None, None


def _first_major_peak(s: np.ndarray, ratio: float = 0.6) -> Optional[int]:
    top = np.nanmax(s) if np.any(np.isfinite(s)) else np.nan
    if not np.isfinite(top):
        return None
    for i in range(len(s)):
        if not np.isfinite(s[i]) or s[i] < ratio * top:
            continue
        prev = s[i - 1] if i > 0 and np.isfinite(s[i - 1]) else -np.inf
        nxt = s[i + 1] if i + 1 < len(s) and np.isfinite(s[i + 1]) else -np.inf
        if s[i] >= prev and s[i] >= nxt:
            return i
    return _nanargmax(s)


def _batting_sides(batting_style: Optional[str]):
    """(front, back) landmark ids. Right-handers lead with the left side."""
    left_handed = bool(batting_style) and "left" in batting_style.lower()
    if left_handed:
        return dict(sh=RIGHT_SHOULDER, el=RIGHT_ELBOW, wr=RIGHT_WRIST, hip=RIGHT_HIP, knee=RIGHT_KNEE, ank=RIGHT_ANKLE), \
               dict(ank=LEFT_ANKLE, knee=LEFT_KNEE, hip=LEFT_HIP), True
    return dict(sh=LEFT_SHOULDER, el=LEFT_ELBOW, wr=LEFT_WRIST, hip=LEFT_HIP, knee=LEFT_KNEE, ank=LEFT_ANKLE), \
           dict(ank=RIGHT_ANKLE, knee=RIGHT_KNEE, hip=RIGHT_HIP), False


def analyze_batting(path: str, batting_style: Optional[str] = None) -> EngineResult:
    deadline = _Deadline(TIME_BUDGET_S)
    with ENGINE_LOCK:
        clip = read_action_clip(path, "batting", deadline)
        poses = track_poses(clip.frames, deadline)
    metrics, faults, impact, warnings = batting_from_poses(poses, clip.fps, batting_style)
    return EngineResult(
        metrics=metrics,
        faults=faults,
        landmarks=_best_landmarks(poses, impact),
        raw_frame_metrics=extract_raw_frame_metrics(poses, "batting"),
        meta=_meta(clip, poses, impact, warnings),
    )


def batting_from_poses(poses: List[Optional[dict]], fps: float, batting_style: Optional[str] = None):
    """Batting metrics from a pose sequence. Returns (metrics, faults, impact_frame, warnings)."""
    _require_pose(poses)
    P = landmark_array(poses)
    T = len(P)
    torso = _torso(P)
    if not np.isfinite(torso):
        raise AnalysisError("ERR_NO_HUMAN_DETECTED: Shoulders and hips are not visible — film the full body.")

    front, back, left_handed = _batting_sides(batting_style)
    warnings: List[str] = []

    hip_c = _mid(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP))
    sh_c = _mid(_pt(P, LEFT_SHOULDER), _pt(P, RIGHT_SHOULDER))
    hands = np.nanmean(np.stack([_pt(P, LEFT_WRIST), _pt(P, RIGHT_WRIST)]), axis=0)
    hand_speed = _speed(hands - hip_c, fps) / torso            # torso-lengths per second

    # Camera view decides what can be measured: from front-on/behind, the stride moves
    # toward the camera and is invisible in 2D, so footwork timing is skipped rather than guessed.
    view = _camera_view(P, torso)
    side_on = view == "side-on"
    if not side_on:
        warnings.append("Camera is not side-on — footwork timing and stride length were not measured. "
                        "Film from square of the pitch at hip height for the full analysis.")

    # Front-foot stride: the first sustained movement of the front ankle relative to the back ankle.
    feet_gap = _pt(P, front["ank"]) - _pt(P, back["ank"])
    plant = _first_stride(_speed(feet_gap, fps) / torso, feet_gap, torso)[1] if side_on else None

    # Contact: fastest hands around the front-foot landing. Without a stride (back-foot
    # shots) use the first major swing peak — later peaks are usually the batter
    # relaxing, turning or running after the shot.
    if plant is not None:
        w_imp = _win(T, plant - 0.3 * fps, plant + 0.8 * fps)
        rel = _nanargmax(hand_speed[w_imp])
        impact = w_imp.start + rel if rel is not None else None
    else:
        impact = _first_major_peak(hand_speed)
    if impact is None:
        raise AnalysisError("ERR_NO_HUMAN_DETECTED: Could not track the batter's hands.")
    peak_hand_speed = float(hand_speed[impact])
    if peak_hand_speed < 2.0:
        raise AnalysisError("ERR_NOT_CRICKET_ACTION: No bat swing detected. Upload a clip that shows the full shot.")
    plant_ms = (impact - plant) / fps * 1000 if plant is not None else None

    # Head stability: head movement during the downswing (the stride's natural drop is excluded).
    # Only frames where the face is actually seen count — from behind, the nose is a guess.
    w_down = _win(T, impact - 0.25 * fps, impact + 0.1 * fps)
    nose = _pt(P, NOSE)[w_down][_face_visible(poses)[w_down]]
    nose = nose[np.all(np.isfinite(nose), axis=1)]
    head_ratio = float(np.sqrt(np.sum(np.var(nose, axis=0))) / torso) if len(nose) >= 3 else None
    head_score = clamp_score(100 * np.clip(1 - (head_ratio - 0.03) / 0.12, 0, 1)) if head_ratio is not None else None
    if head_ratio is None:
        head_note = "Face not visible in the downswing (camera behind the batter?) — head stability not measured"
    elif head_ratio < 0.04:
        head_note = "Excellent — head very still through the downswing"
    elif head_ratio < 0.08:
        head_note = "Good — minor head movement in the downswing"
    elif head_ratio < 0.12:
        head_note = "Fair — noticeable head movement, keep the eyes level"
    else:
        head_note = "Poor — head moving a lot as the bat comes down"

    knee = _angle3(_pt(P, front["hip"]), _pt(P, front["knee"]), _pt(P, front["ank"]))
    flex = np.nanmin(knee[_win(T, impact - 0.6 * fps, impact + 0.1 * fps)]) if np.any(np.isfinite(knee)) else np.nan
    ext = np.nanmax(knee[_win(T, impact, T)]) if np.any(np.isfinite(knee[_win(T, impact, T)])) else np.nan
    flex_score = range_score(flex, 110, 150, 30)
    ext_score = range_score(ext, 155, 180, 30)
    seq_score = None
    if plant_ms is not None:
        seq_score = range_score(plant_ms, 80, 400, 300)
    timing_score = _weighted([(seq_score, 0.5), (flex_score, 0.25), (ext_score, 0.25)])
    if plant_ms is None:
        timing_note = ("Footwork timing needs a side-on camera — scored from front-knee drive only" if not side_on
                       else "No clear front-foot stride detected — timing scored from knee drive only")
    elif plant_ms < 50:
        timing_note = f"Late — front foot landed {plant_ms:.0f} ms before the swing peak (aim for 80–400 ms)"
    elif plant_ms > 600:
        timing_note = f"Early — front foot planted {plant_ms:.0f} ms before the swing peak"
    else:
        timing_note = f"Good sequencing — front foot planted {plant_ms:.0f} ms before the swing peak"

    # Balance at contact: spine lean and head over the base.
    lean = _lean_from_vertical(hip_c, sh_c)
    lean_imp = _val(lean, impact)
    feet_c = _mid(_pt(P, LEFT_ANKLE), _pt(P, RIGHT_ANKLE))
    head_off = np.abs(_pt(P, NOSE)[:, 0] - feet_c[:, 0]) / torso
    head_off_imp = _val(head_off, impact)
    balance_score = _weighted([
        (range_score(lean_imp, 0, 30, 25), 0.5),
        (clamp_score(100 * np.clip(1 - (head_off_imp - 0.5) / 1.0, 0, 1)) if head_off_imp is not None else None, 0.5),
    ])
    hip_width = np.linalg.norm(_pt(P, LEFT_HIP) - _pt(P, RIGHT_HIP), axis=-1) / torso
    hip_tilt = _line_tilt(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP))
    hip_tilt = np.where(hip_width > 0.3, hip_tilt, np.nan)  # side-on hips overlap: tilt meaningless
    avg_hip_tilt = float(np.nanmean(hip_tilt)) if np.any(np.isfinite(hip_tilt)) else None
    if balance_score is None:
        balance_note = "Could not measure balance at contact"
    elif balance_score >= 75:
        balance_note = f"Balanced — spine lean {lean_imp:.0f}°, head over the base at contact"
    elif balance_score >= 55:
        balance_note = f"Fair — spine lean {lean_imp or 0:.0f}°, head drifting from the base"
    else:
        balance_note = "Off balance at contact — head falling outside the feet"

    # Stride: widest ankle separation in the half second into contact.
    ank_gap = np.linalg.norm(_pt(P, LEFT_ANKLE) - _pt(P, RIGHT_ANKLE), axis=-1) / torso
    seg = ank_gap[_win(T, impact - 0.5 * fps, impact + 1)]
    stride_ratio = float(np.nanmax(seg)) if side_on and np.any(np.isfinite(seg)) else None
    stride_score = range_score(stride_ratio, 0.8, 1.6, 0.8)
    if stride_ratio is None:
        stride_note = ("Stride length needs a side-on camera" if not side_on
                       else "Feet not visible enough to measure the stride")
    elif stride_ratio < 0.6:
        stride_note = f"Short stride ({stride_ratio:.2f}× torso) — limited weight transfer"
    elif stride_ratio > 2.0:
        stride_note = f"Over-long stride ({stride_ratio:.2f}× torso) — hard to stay balanced"
    else:
        stride_note = f"Good stride ({stride_ratio:.2f}× torso length)"

    # Follow-through: how far the hands rise after contact.
    hand_h = (sh_c[:, 1] - hands[:, 1]) / torso     # + = above shoulders
    after = hand_h[_win(T, impact, T)]
    h_imp = _val(hand_h, impact)
    wrist_delta = float(np.nanmax(after) - h_imp) if h_imp is not None and np.any(np.isfinite(after)) else None
    follow_score = clamp_score(25 + 70 * np.clip((wrist_delta + 0.1) / 1.1, 0, 1)) if wrist_delta is not None else None
    if wrist_delta is None:
        follow_note = "Follow-through not captured — keep recording after contact"
    elif wrist_delta > 0.8:
        follow_note = "Full follow-through — hands finish high"
    elif wrist_delta > 0.3:
        follow_note = "Partial follow-through"
    else:
        follow_note = "Checked follow-through — fine for defence, short for attacking shots"

    # Stance at set-up (before the stride starts).
    setup_end = plant if plant is not None else impact - int(0.6 * fps)
    w_setup = _win(T, 0, max(3, min(setup_end, T)))
    knees_setup = np.nanmean(np.stack([
        _angle3(_pt(P, LEFT_HIP), _pt(P, LEFT_KNEE), _pt(P, LEFT_ANKLE))[w_setup],
        _angle3(_pt(P, RIGHT_HIP), _pt(P, RIGHT_KNEE), _pt(P, RIGHT_ANKLE))[w_setup],
    ]))
    width_setup = np.nanmedian(ank_gap[w_setup]) if np.any(np.isfinite(ank_gap[w_setup])) else np.nan
    stance_score = _weighted([(range_score(knees_setup, 145, 172, 25), 0.5),
                              (range_score(width_setup, 0.5, 1.0, 0.5), 0.5)])

    # Swing plane: direction of hand travel into contact (90° = vertical bat).
    v = np.gradient(hands, axis=0)[_win(T, impact - 2, impact + 1)]
    v = v[np.all(np.isfinite(v), axis=1)]
    swing_angle = float(np.degrees(np.arctan2(np.abs(v[:, 1]).mean(), np.abs(v[:, 0]).mean()))) if len(v) else None

    # Shot family from swing plane, knee bend and swing speed (2D estimate). The swing
    # plane only reads correctly from side-on, so other views are left unclassified.
    if not side_on:
        shot = "Unclassified (needs side-on view)"
    elif peak_hand_speed < 5:
        shot = "Defensive Shot"
    elif np.isfinite(flex) and flex < 100:
        shot = "Sweep"
    elif swing_angle is not None and swing_angle >= 55:
        shot = "Drive"
    elif swing_angle is not None and swing_angle >= 25:
        shot = "Cut / Flick"
    else:
        shot = "Pull / Hook"

    metrics = {
        "headStabilityScore": head_score,
        "headStabilityVariance": _r(head_ratio, 4),
        "headStabilityNote": head_note,
        "headPositionScore": head_score,
        "headPosition": head_note,
        "timingScore": timing_score,
        "plantToImpactMs": _r(plant_ms, 0),
        "peakKneeFlexion": _r(flex),
        "peakKneeExtension": _r(ext),
        "rangeOfMotion": _r(ext - flex) if np.isfinite(ext) and np.isfinite(flex) else None,
        "timingNote": timing_note,
        "balanceScore": balance_score,
        "avgHipTilt": _r(avg_hip_tilt),
        "avgSpineOffset": _r(lean_imp),
        "balanceNote": balance_note,
        "strideScore": stride_score,
        "avgStrideRatio": _r(stride_ratio, 3),
        "strideNote": stride_note,
        "followThroughScore": follow_score,
        "wristYDelta": _r(wrist_delta, 3),
        "followThroughNote": follow_note,
        "stanceScore": stance_score,
        "batSwingAngle": _r(swing_angle),
        "peakHandSpeed": _r(peak_hand_speed, 2),
        "shotType": shot,
        "battingSide": "left-handed" if left_handed else "right-handed",
        "cameraView": view,
    }
    metrics["overallBattingScore"] = _weighted([
        (timing_score, 0.25), (balance_score, 0.20), (head_score, 0.20),
        (stance_score, 0.15), (follow_score, 0.10), (stride_score, 0.10),
    ]) or 0

    # Soft sanity check: a high overhead arm arc looks more like bowling.
    arm_top = np.nanmax(np.stack([
        (_pt(P, LEFT_SHOULDER)[:, 1] - _pt(P, LEFT_WRIST)[:, 1]),
        (_pt(P, RIGHT_SHOULDER)[:, 1] - _pt(P, RIGHT_WRIST)[:, 1]),
    ])) / torso
    if np.isfinite(arm_top) and arm_top > 1.2 and plant is None:
        warnings.append("This clip looks more like a bowling action than a batting shot.")

    return metrics, _fault_detector.detect(metrics, "batting"), impact, warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Bowling
# ═══════════════════════════════════════════════════════════════════════════════

def _arm(side: str):
    if side == "left":
        return dict(sh=LEFT_SHOULDER, el=LEFT_ELBOW, wr=LEFT_WRIST, front_ank=RIGHT_ANKLE)
    return dict(sh=RIGHT_SHOULDER, el=RIGHT_ELBOW, wr=RIGHT_WRIST, front_ank=LEFT_ANKLE)


def _count_speed_peaks(s: np.ndarray) -> int:
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return 0
    # Pad with zeros so a surge that peaks at either edge of the window still counts.
    s = np.concatenate([[0.0], s, [0.0]])
    top = s.max()
    peaks = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.25 * top]
    distinct = peaks[:1]
    for p in peaks[1:]:
        valley = s[distinct[-1]:p + 1].min()
        if valley < 0.75 * min(s[distinct[-1]], s[p]):
            distinct.append(p)
        elif s[p] > s[distinct[-1]]:
            distinct[-1] = p
    return len(distinct)


def _estimate_ball_speed(path: str, clip: Clip, release: int, P: np.ndarray,
                         deadline: _Deadline) -> Tuple[Optional[float], str]:
    """
    Track the ball (YOLO 'sports ball') over ~0.35 s after release at the source
    frame rate, fit a straight path and scale pixels to metres using the bowler's
    height. Only valid for a roughly side-on camera; returns None when the ball is
    not tracked reliably.
    """
    src = clip.start_index + release * clip.stride
    n = max(4, int(round(0.35 * clip.source_fps)))
    frames = _read_range(path, src, n, 1)
    if len(frames) < 4:
        return None, "Not enough frames after release to track the ball"
    h_px = frames[0].shape[0]

    pts = []
    prev = None
    for i, frm in enumerate(frames):
        deadline.check()
        dets = [d for d in detect_balls(frm) if d[2] < 0.08 * h_px]
        if not dets:
            continue
        d = min(dets, key=lambda d: np.hypot(d[0] - prev[0], d[1] - prev[1])) if prev else min(dets, key=lambda d: d[2])
        pts.append((i / clip.source_fps, d[0] / h_px, d[1] / h_px))   # frame-height units
        prev = d
    if len(pts) < 3:
        return None, "Ball not detected in enough frames after release"

    t = np.array([p[0] for p in pts])
    xy = np.array([[p[1], p[2]] for p in pts])
    vx, cx = np.polyfit(t, xy[:, 0], 1)
    vy, cy = np.polyfit(t, xy[:, 1], 1)
    resid = np.sqrt(np.mean((xy[:, 0] - (vx * t + cx)) ** 2 + (xy[:, 1] - (vy * t + cy)) ** 2))
    if resid > 0.03:
        return None, "Ball track was inconsistent — could not estimate pace"

    body = np.linalg.norm(_pt(P, NOSE) - _mid(_pt(P, LEFT_ANKLE), _pt(P, RIGHT_ANKLE)), axis=-1)
    body_h = float(np.nanmedian(body))
    if not np.isfinite(body_h) or body_h <= 0:
        return None, "Bowler height not measurable for pace scaling"
    kmh = float(np.hypot(vx, vy)) * (NOSE_TO_ANKLE_M / body_h) * 3.6
    if not 40 <= kmh <= 170:
        return None, "Ball track gave an implausible speed — camera angle probably not side-on"
    return round(kmh, 1), f"Estimated from {len(pts)} ball detections after release (side-on camera assumed)"


def classify_ball_speed(speed_kmh: Optional[float]) -> str:
    if speed_kmh is None:
        return "Undetected"
    if speed_kmh >= 135:
        return "Fast"
    if speed_kmh >= 115:
        return "Medium-Fast"
    if speed_kmh >= 95:
        return "Medium"
    return "Spin/Slow"


def analyze_bowling(path: str, bowling_style: Optional[str] = None) -> EngineResult:
    deadline = _Deadline(TIME_BUDGET_S)
    with ENGINE_LOCK:
        clip = read_action_clip(path, "bowling", deadline)
        poses = track_poses(clip.frames, deadline, clip.seed)
        metrics, faults, release = bowling_from_poses(
            poses, clip.fps, bowling_style,
            ball_speed_fn=lambda rel, P: _estimate_ball_speed(path, clip, rel, P, deadline),
            around=clip.seed[0] if clip.seed else None,
        )
    return EngineResult(
        metrics=metrics,
        faults=faults,
        landmarks=_best_landmarks(poses, release),
        raw_frame_metrics=extract_raw_frame_metrics(poses, "bowling"),
        meta=_meta(clip, poses, release, []),
    )


def bowling_from_poses(poses: List[Optional[dict]], fps: float, bowling_style: Optional[str] = None,
                       ball_speed_fn=None, around: Optional[int] = None):
    """
    Bowling metrics from a pose sequence. Returns (metrics, faults, release_frame).
    `around` is the frame where the bowler was identified; release is searched near
    it so a later overhead movement (batter's swing after a hand-over) is ignored.
    """
    _require_pose(poses)
    P = landmark_array(poses)
    T = len(P)
    torso = _torso(P)
    if not np.isfinite(torso):
        raise AnalysisError("ERR_NO_HUMAN_DETECTED: Shoulders and hips are not visible — film the full body.")

    # Bowling arm: from the profile, otherwise the arm that whirls fastest around its
    # shoulder (the front arm can go just as high in the gather, but moves far slower).
    style = (bowling_style or "").lower()
    heights, whirl = {}, {}
    for side in ("left", "right"):
        a = _arm(side)
        heights[side] = (_pt(P, a["sh"])[:, 1] - _pt(P, a["wr"])[:, 1]) / torso
        sp = _speed(_pt(P, a["wr"]) - _pt(P, a["sh"]), fps) / torso
        whirl[side] = float(np.nanmax(sp)) if np.any(np.isfinite(sp)) else -1.0
    if "left-arm" in style:
        side = "left"
    elif "right-arm" in style:
        side = "right"
    else:
        side = max(whirl, key=whirl.get)
    arm = _arm(side)
    h = heights[side]

    w_rel = _win(T, around - 0.6 * fps, around + 1.0 * fps) if around is not None else slice(0, T)
    rel = _nanargmax(h[w_rel])
    release = w_rel.start + rel if rel is not None else None
    if release is None:
        raise AnalysisError("ERR_NO_HUMAN_DETECTED: Could not track the bowling arm.")
    release_h = float(h[release])
    if release_h < 0.3:
        raise AnalysisError(
            "ERR_INVALID_ACTION: The bowling arm never rises well above the shoulder. Make sure the whole "
            "delivery stride and release are in the clip.")

    ball_speed, speed_note = ball_speed_fn(release, P) if ball_speed_fn else (None, "Pace not estimated")

    sh, el, wr = _pt(P, arm["sh"]), _pt(P, arm["el"]), _pt(P, arm["wr"])

    # Release point: height above the shoulder and elbow extension at release.
    elbow = _angle3(sh, el, wr)
    elbow_rel = _val(elbow, release)
    height_score = clamp_score(35 + 60 * np.clip(release_h / 1.0, 0, 1))
    ext_score = range_score(elbow_rel, 160, 180, 30)
    release_score = _weighted([(height_score, 0.6), (ext_score, 0.4)])
    if release_h >= 0.9:
        release_note = f"High release — wrist {release_h:.2f} torso-lengths above the shoulder"
    elif release_h >= 0.6:
        release_note = f"Good release height ({release_h:.2f} torso-lengths above the shoulder)"
    else:
        release_note = f"Low release — wrist only {release_h:.2f} torso-lengths above the shoulder"
    arm_slot = _val(_lean_from_vertical(sh, wr), release)

    # Wrist position: wrist above (behind) the elbow at release.
    wrist_delta = _val((el[:, 1] - wr[:, 1]) / torso, release)
    wrist_score = clamp_score(30 + 65 * np.clip(wrist_delta / 0.6, 0, 1)) if wrist_delta is not None else None
    if wrist_delta is None:
        wrist_note = "Wrist/elbow not visible at release"
    elif wrist_delta >= 0.3:
        wrist_note = "Good — wrist high and behind the ball at release"
    else:
        wrist_note = "Wrist dropping toward elbow height at release — firm it up"

    # Smoothness: arm speed (relative to the shoulder) through the delivery arc.
    rel = wr - sh
    arm_speed = _speed(rel, fps) / torso
    phase = _win(T, release - 0.5 * fps, release + 0.25 * fps)
    speed_seg = arm_speed[phase]
    jerk = np.gradient(np.gradient(np.gradient(rel[phase], axis=0), axis=0), axis=0) * fps ** 3 / torso
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    avg_jerk = float(np.nanmean(jerk_mag)) if np.any(np.isfinite(jerk_mag)) else None
    if np.sum(np.isfinite(speed_seg)) >= 6:
        extra = max(0, _count_speed_peaks(speed_seg) - 1)
        smooth_score = clamp_score(98 - 22 * extra)
        smooth_note = ("Smooth — one continuous acceleration through the arc" if extra == 0
                       else f"Arm speed surges {extra + 1} times through the arc — rhythm is uneven")
    else:
        smooth_score, smooth_note = None, "Arm not tracked through enough of the delivery arc"

    # Balance at release: spine lean and head over the front foot.
    hip_c = _mid(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP))
    sh_c = _mid(_pt(P, LEFT_SHOULDER), _pt(P, RIGHT_SHOULDER))
    lean_rel = _val(_lean_from_vertical(hip_c, sh_c), release)
    head_off = _val(np.abs(_pt(P, NOSE)[:, 0] - _pt(P, arm["front_ank"])[:, 0]) / torso, release)
    balance_score = _weighted([
        (range_score(lean_rel, 0, 25, 25), 0.5),
        (clamp_score(100 * np.clip(1 - (head_off - 0.6) / 1.0, 0, 1)) if head_off is not None else None, 0.5),
    ])
    hip_width = np.linalg.norm(_pt(P, LEFT_HIP) - _pt(P, RIGHT_HIP), axis=-1) / torso
    hip_tilt = np.where(hip_width > 0.3, _line_tilt(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP)), np.nan)
    hip_tilt_rel = _val(hip_tilt, release)
    balance_note = (f"Spine lean {lean_rel:.0f}° at release, head "
                    f"{'over' if (head_off or 9) <= 0.6 else 'away from'} the front foot") if lean_rel is not None \
        else "Could not measure balance at release"

    speed_class = classify_ball_speed(ball_speed)
    pace = {"Fast": "fast", "Medium-Fast": "medium-fast", "Medium": "medium pace", "Spin/Slow": "spin/slow"}
    bowling_type = f"{side.capitalize()}-arm " + (pace[speed_class] if ball_speed else "bowler")

    metrics = {
        "armSmoothnessScore": smooth_score,
        "avgJerk": _r(avg_jerk, 2),
        "armSmoothnessNote": smooth_note,
        "releasePointScore": release_score,
        "releaseHeight": _r(release_h, 3),
        "peakWristY": _r(_val(_pt(P, arm["wr"])[:, 1], release), 4),
        "releaseArmAngle": _r(elbow_rel),
        "releasePointNote": release_note,
        "armSlotAngle": _r(arm_slot),
        "armRotationAngle": _r(elbow_rel),
        "armRotationScore": ext_score,
        "wristPositionScore": wrist_score,
        "wristElbowDelta": _r(wrist_delta, 3),
        "wristPositionNote": wrist_note,
        "balanceScore": balance_score,
        "avgHipTilt": _r(hip_tilt_rel),
        "balanceNote": balance_note,
        "estimatedBallSpeed": ball_speed,
        "speedClassification": speed_class,
        "speedNote": speed_note,
        "bowlingArm": side,
        "bowlingStyle": bowling_type,
    }
    metrics["overallBowlingScore"] = _weighted([
        (smooth_score, 0.30), (release_score, 0.30), (wrist_score, 0.20), (balance_score, 0.20),
    ]) or 0

    return metrics, _fault_detector.detect(metrics, "bowling"), release


# ═══════════════════════════════════════════════════════════════════════════════
# Posture
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_posture(path: str) -> EngineResult:
    deadline = _Deadline(TIME_BUDGET_S)
    is_video = os.path.splitext(path)[1].lower() in VIDEO_EXTS
    clip = None
    with ENGINE_LOCK:
        if is_video:
            clip = read_action_clip(path, "posture", deadline)
            poses = track_poses(clip.frames, deadline)
            _require_pose(poses)
        else:
            lm = analyze_pose_from_image(path)
            if lm is None:
                raise AnalysisError("ERR_NO_HUMAN_DETECTED: No player found. Make sure the whole body is visible "
                                    "and well lit.")
            poses = [lm]

    P = landmark_array(poses, smooth=False)
    torso = _torso(P)
    if not np.isfinite(torso):
        raise AnalysisError("ERR_NO_HUMAN_DETECTED: Shoulders and hips must both be visible.")

    def med(a):
        return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else None

    sh_w = np.linalg.norm(_pt(P, LEFT_SHOULDER) - _pt(P, RIGHT_SHOULDER), axis=-1) / torso
    hip_w = np.linalg.norm(_pt(P, LEFT_HIP) - _pt(P, RIGHT_HIP), axis=-1) / torso
    sh_tilt = med(np.where(sh_w > 0.25, _line_tilt(_pt(P, LEFT_SHOULDER), _pt(P, RIGHT_SHOULDER)), np.nan))
    hip_tilt = med(np.where(hip_w > 0.2, _line_tilt(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP)), np.nan))
    knee = med(np.nanmean(np.stack([
        _angle3(_pt(P, LEFT_HIP), _pt(P, LEFT_KNEE), _pt(P, LEFT_ANKLE)),
        _angle3(_pt(P, RIGHT_HIP), _pt(P, RIGHT_KNEE), _pt(P, RIGHT_ANKLE)),
    ]), axis=0))
    lean = med(_lean_from_vertical(_mid(_pt(P, LEFT_HIP), _pt(P, RIGHT_HIP)),
                                   _mid(_pt(P, LEFT_SHOULDER), _pt(P, RIGHT_SHOULDER))))

    sh_score = range_score(sh_tilt, 0, 4, 10)
    knee_score = range_score(knee, 140, 170, 25)
    hip_score = range_score(hip_tilt, 0, 4, 10)
    spine_score = range_score(lean, 0, 12, 20)

    def note(v, good, ok, unit, labels):
        if v is None:
            return "Not visible enough to measure (side-on view)"
        return f"{labels[0] if v <= good else labels[1] if v <= ok else labels[2]} ({v:.0f}{unit})"

    knee_note = ("Not visible enough to measure" if knee is None else
                 f"Good athletic knee bend ({knee:.0f}°)" if 140 <= knee <= 170 else
                 f"Too crouched ({knee:.0f}°)" if knee < 140 else f"Too upright ({knee:.0f}°)")

    metrics = {
        "shoulderAlignmentScore": sh_score,
        "shoulderTiltDeg": _r(sh_tilt),
        "shoulderAlignmentNote": note(sh_tilt, 4, 8, "°", ["Level shoulders", "Slight shoulder tilt", "Shoulder tilt"]),
        "kneeBendAngle": _r(knee),
        "kneeBendScore": knee_score,
        "kneeBendNote": knee_note,
        "balanceScore": hip_score,
        "hipTiltDeg": _r(hip_tilt),
        "balanceNote": note(hip_tilt, 4, 8, "°", ["Level hips", "Slight hip tilt", "Hip tilt"]),
        "spinePosScore": spine_score,
        "spineLeanDeg": _r(lean),
        "spinePosNote": note(lean, 12, 25, "° from vertical", ["Upright spine", "Some lean", "Leaning spine"]),
    }
    metrics["overallPostureScore"] = _weighted([
        (sh_score, 0.30), (knee_score, 0.25), (hip_score, 0.25), (spine_score, 0.20),
    ]) or 0

    faults = _fault_detector.detect(metrics, "posture")
    return EngineResult(
        metrics=metrics,
        faults=faults,
        landmarks=_best_landmarks(poses, None),
        raw_frame_metrics=extract_raw_frame_metrics(poses, "posture"),
        meta=_meta(clip, poses, None, []),
    )


def _meta(clip: Optional[Clip], poses, key_frame: Optional[int], warnings: List[str]) -> Dict[str, Any]:
    meta = {
        "frames_analysed": len(poses),
        "frames_with_pose": sum(1 for p in poses if p),
        "key_frame": key_frame,
        "warnings": warnings,
    }
    if clip is not None:
        meta.update({
            "analysis_fps": round(clip.fps, 2),
            "source_fps": round(clip.source_fps, 2),
            "video_seconds": round(clip.duration_s, 2),
            "window_start_seconds": round(clip.start_index / clip.source_fps, 2),
            "key_moment_seconds": round((clip.start_index + key_frame * clip.stride) / clip.source_fps, 2)
            if key_frame is not None else None,
        })
    return meta
