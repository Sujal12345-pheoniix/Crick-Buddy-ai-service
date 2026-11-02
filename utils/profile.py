"""
Analysis profile: how much work the engine does per clip.

"accurate" suits a laptop or a host with ≥ 2 CPUs. "fast" is for CPU-starved hosts
such as Render's free tier (0.1 CPU), where the accurate profile takes ~4–5 minutes
for a 5-second clip. It analyses fewer frames, uses MediaPipe's lite pose model and
re-detects the player less often — roughly 4–5× less compute.

ANALYSIS_PROFILE=fast | accurate | auto (default: auto picks from the CPU quota).
"""

import os


def available_cpus() -> float:
    """CPUs this process may use, honouring a container CPU quota (cgroup v2 or v1)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:                       # cgroup v2: "quota period"
            quota, period = f.read().split()[:2]
            if quota != "max":
                return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota, period = int(fq.read()), int(fp.read())
            if quota > 0:
                return quota / period
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


def _resolve() -> str:
    choice = os.getenv("ANALYSIS_PROFILE", "auto").strip().lower()
    if choice in ("fast", "accurate"):
        return choice
    return "fast" if available_cpus() < 1.5 else "accurate"


PROFILE = _resolve()
FAST = PROFILE == "fast"

# Per-profile settings
SETTINGS = {
    "accurate": dict(batting_fps=25.0, bowling_fps=30.0, window_s=3.5, detect_every=2, pose_complexity=1,
                     bowling_pose_complexity=1,
                     max_side=800, coarse_samples=30, bowler_rate=6, bowler_imgsz=640),
    "fast": dict(batting_fps=15.0, bowling_fps=20.0, window_s=3.5, detect_every=4, pose_complexity=1,
                 # the lite model mixes up left/right arms on bowlers seen from behind
                 bowling_pose_complexity=1,
                 max_side=640, coarse_samples=24, bowler_rate=4, bowler_imgsz=640),
}[PROFILE]

print(f"[Info] Analysis profile: {PROFILE} (CPUs available: {available_cpus():.2f})")
