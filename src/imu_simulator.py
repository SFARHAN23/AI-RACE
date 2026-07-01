"""Simulated IMU and real-RC failure modes for AIRACE.

This does not replace LiDAR. It adds the signals a real RC car IMU would give us:
- yaw rate gyro z
- longitudinal acceleration
- lateral acceleration
- simple impact/spin/slip/stuck diagnostics

The point is to train/debug policies against problems we expect on hardware:
noise, bias drift, saturation, occasional spikes, and short dropouts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from rc_car_model import DT


@dataclass
class IMUConfig:
    enabled: bool = True
    yaw_rate_noise_std: float = 0.025       # rad/s
    accel_noise_std: float = 0.12           # m/s^2
    yaw_bias_walk_std: float = 0.0008       # rad/s per step
    accel_bias_walk_std: float = 0.004      # m/s^2 per step
    yaw_rate_clip: float = 8.0              # rad/s, cheap IMU saturation
    accel_clip: float = 25.0                # m/s^2
    dropout_prob: float = 0.001             # occasional stale/invalid IMU packet
    spike_prob: float = 0.001               # occasional hard bump/electrical spike
    spike_scale_yaw: float = 1.8
    spike_scale_accel: float = 5.0
    stale_hold_steps: int = 3

    # Diagnostics thresholds for safety/reward/logging.
    spin_yaw_rate_rad_s: float = 2.4
    slip_lat_acc_m_s2: float = 4.5
    impact_acc_m_s2: float = 10.0
    stuck_speed_mps: float = 0.10
    stuck_accel_m_s2: float = 0.40


class SimulatedIMU:
    """Stateful noisy IMU model."""

    def __init__(self, config: IMUConfig | None = None, rng=None):
        self.config = config or IMUConfig()
        self.rng = rng if rng is not None else np.random
        self.reset()

    def reset(self):
        self.yaw_bias = 0.0
        self.long_bias = 0.0
        self.lat_bias = 0.0
        self._last = np.zeros(3, dtype=np.float64)
        self._stale_left = 0
        self.last_meta: dict[str, Any] = {"valid": True, "events": []}

    def read(self, car, applied_action=None) -> tuple[np.ndarray, dict[str, Any]]:
        """Return [yaw_rate, long_acc, lat_acc] with noise/faults and metadata."""
        cfg = self.config
        events: list[str] = []

        true_yaw = float(car.psi_dot)
        true_long = float(getattr(car, "long_acc", 0.0))
        true_lat = float(getattr(car, "lat_acc", 0.0))

        self.yaw_bias += float(self.rng.normal(0.0, cfg.yaw_bias_walk_std))
        self.long_bias += float(self.rng.normal(0.0, cfg.accel_bias_walk_std))
        self.lat_bias += float(self.rng.normal(0.0, cfg.accel_bias_walk_std))

        sample = np.array([
            true_yaw + self.yaw_bias + self.rng.normal(0.0, cfg.yaw_rate_noise_std),
            true_long + self.long_bias + self.rng.normal(0.0, cfg.accel_noise_std),
            true_lat + self.lat_bias + self.rng.normal(0.0, cfg.accel_noise_std),
        ], dtype=np.float64)

        if self.rng.random() < cfg.spike_prob:
            sample[0] += float(self.rng.normal(0.0, cfg.spike_scale_yaw))
            sample[1:] += self.rng.normal(0.0, cfg.spike_scale_accel, size=2)
            events.append("imu_spike")

        valid = True
        if self._stale_left > 0:
            sample = self._last.copy()
            self._stale_left -= 1
            valid = False
            events.append("imu_stale")
        elif self.rng.random() < cfg.dropout_prob:
            self._stale_left = cfg.stale_hold_steps
            sample = self._last.copy()
            valid = False
            events.append("imu_dropout")

        sample[0] = np.clip(sample[0], -cfg.yaw_rate_clip, cfg.yaw_rate_clip)
        sample[1:] = np.clip(sample[1:], -cfg.accel_clip, cfg.accel_clip)
        self._last = sample.copy()

        speed = float(getattr(car, "spd", 0.0))
        yaw_abs = abs(float(sample[0]))
        acc_norm = float(np.hypot(sample[1], sample[2]))
        lat_abs = abs(float(sample[2]))
        if yaw_abs > cfg.spin_yaw_rate_rad_s:
            events.append("spin_risk")
        if lat_abs > cfg.slip_lat_acc_m_s2:
            events.append("lateral_slip_risk")
        if acc_norm > cfg.impact_acc_m_s2:
            events.append("impact_or_bump")
        if speed < cfg.stuck_speed_mps and abs(float(sample[1])) < cfg.stuck_accel_m_s2:
            events.append("possibly_stuck")

        meta = {
            "valid": bool(valid),
            "events": events,
            "yaw_rate_rad_s": float(sample[0]),
            "long_acc_m_s2": float(sample[1]),
            "lat_acc_m_s2": float(sample[2]),
            "acc_norm_m_s2": acc_norm,
            "config": asdict(cfg),
        }
        self.last_meta = meta
        return sample, meta


def normalize_imu(sample: np.ndarray, cfg: IMUConfig) -> np.ndarray:
    """Normalize IMU sample to roughly [-1, 1] for neural policy input."""
    sample = np.asarray(sample, dtype=np.float64).flatten()[:3]
    return np.array([
        np.clip(sample[0] / cfg.yaw_rate_clip, -1.0, 1.0),
        np.clip(sample[1] / cfg.accel_clip, -1.0, 1.0),
        np.clip(sample[2] / cfg.accel_clip, -1.0, 1.0),
    ], dtype=np.float32)
