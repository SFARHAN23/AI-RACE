"""Outlap track-memory features for AIRACE.

This is the practical version of "learn the track on the outlap, use it on the push lap".
In sim we can key memory by virtual sector index. On real hardware we would need
odometry/localization/markers to decide which sector the car is currently in.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TrackMemoryConfig:
    enabled: bool = True
    sector_count: int = 12
    ema_alpha: float = 0.18
    safe_clearance_m: float = 1.0
    max_speed_mps: float = 2.0
    max_yaw_rate_rad_s: float = 4.0


class SectorTrackMemory:
    """Compact per-sector memory table updated during the first/outlap."""

    def __init__(self, config: TrackMemoryConfig | None = None):
        self.config = config or TrackMemoryConfig()
        self.reset()

    def reset(self):
        n = max(1, int(self.config.sector_count))
        self.visits = np.zeros(n, dtype=np.int32)
        self.min_clearance = np.full(n, np.inf, dtype=np.float64)
        self.avg_speed = np.zeros(n, dtype=np.float64)
        self.avg_abs_yaw = np.zeros(n, dtype=np.float64)
        self.safety_interventions = np.zeros(n, dtype=np.float64)
        self.corner_sign = np.zeros(n, dtype=np.float64)  # -left, +right-ish depending model sign
        self.lap_count = 0
        self.last_sector = 0

    def update(self, sector_index: int, speed_mps: float, yaw_rate_rad_s: float,
               front_clearance_m: float, safety_count: int = 0, finished_lap: bool = False):
        cfg = self.config
        n = len(self.visits)
        s = int(sector_index) % n
        if finished_lap:
            self.lap_count += 1
        self.last_sector = s
        self.visits[s] += 1
        a = cfg.ema_alpha if self.visits[s] > 1 else 1.0
        self.avg_speed[s] = (1 - a) * self.avg_speed[s] + a * float(speed_mps)
        self.avg_abs_yaw[s] = (1 - a) * self.avg_abs_yaw[s] + a * abs(float(yaw_rate_rad_s))
        self.corner_sign[s] = (1 - a) * self.corner_sign[s] + a * float(np.sign(yaw_rate_rad_s))
        self.safety_interventions[s] = (1 - a) * self.safety_interventions[s] + a * float(safety_count)
        if np.isfinite(front_clearance_m):
            self.min_clearance[s] = min(self.min_clearance[s], float(front_clearance_m))

    def features(self, sector_index: int) -> np.ndarray:
        """Return compact normalized memory for current and next sector.

        Features:
        [lap_progress_phase, current_visited, current_clearance_risk,
         current_speed_memory, current_corner_strength, current_safety_risk,
         next_visited, next_clearance_risk, next_corner_strength, next_safety_risk]
        """
        cfg = self.config
        n = len(self.visits)
        s = int(sector_index) % n
        nxt = (s + 1) % n

        def f(idx: int):
            visited = 1.0 if self.visits[idx] > 0 else 0.0
            clearance = self.min_clearance[idx]
            if not np.isfinite(clearance):
                clearance_risk = 0.0
            else:
                clearance_risk = np.clip((cfg.safe_clearance_m - clearance) / max(cfg.safe_clearance_m, 1e-6), 0.0, 1.0)
            speed_mem = np.clip(self.avg_speed[idx] / max(cfg.max_speed_mps, 1e-6), 0.0, 1.0)
            corner_strength = np.clip(self.avg_abs_yaw[idx] / max(cfg.max_yaw_rate_rad_s, 1e-6), 0.0, 1.0)
            safety_risk = np.clip(self.safety_interventions[idx] / 3.0, 0.0, 1.0)
            return visited, clearance_risk, speed_mem, corner_strength, safety_risk

        cv, cc, cs, cy, ci = f(s)
        nv, nc, ns, ny, ni = f(nxt)
        lap_phase = np.clip(self.lap_count / 2.0, 0.0, 1.0)  # 0=outlap, 0.5/1=push-ish
        return np.array([lap_phase, cv, cc, cs, cy, ci, nv, nc, ny, ni], dtype=np.float32)

    def summary(self) -> dict:
        return {
            "lap_count": int(self.lap_count),
            "visited_sectors": int(np.count_nonzero(self.visits)),
            "sector_count": int(len(self.visits)),
            "visits": self.visits.astype(int).tolist(),
            "min_clearance_m": [None if not np.isfinite(x) else float(x) for x in self.min_clearance],
            "avg_speed_mps": self.avg_speed.astype(float).round(3).tolist(),
            "avg_abs_yaw_rate_rad_s": self.avg_abs_yaw.astype(float).round(3).tolist(),
        }
