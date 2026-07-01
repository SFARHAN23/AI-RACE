"""
safety_wrapper.py — conservative action filter for sim-to-real RC car tests.

The trained TD3 policy can output aggressive throttle/steer commands.  This
module keeps the neural policy as the planner, but clamps its command before it
is allowed to move the car or reach hardware.

The wrapper is deliberately simple and deterministic so it can be reused in:
- laptop rollout visualization
- future ROS2/laptop controller
- future serial command bridge to Arduino/ESC/servo

Action convention:
    raw_action = [throttle/brake, steer] in [-1, 1]
    safe_action = [throttle/brake, steer] in [-1, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from rc_car_model import DT, SPD_MAX
from domain_randomization import LIDAR_MAX_RANGE


@dataclass
class SafetyConfig:
    """Conservative defaults for first real-car style testing."""

    enabled: bool = True

    # First hardware target should be around 0.5-1.0 m/s.  The simulator can
    # run much faster, but this cap keeps visualization honest for hardware.
    max_speed_mps: float = 1.0

    # Normalized command caps.  Steering is still fairly wide, throttle is small.
    max_throttle_cmd: float = 0.28
    max_brake_cmd: float = 0.85
    max_steer_cmd: float = 0.70

    # Per-control-step command slew limits at 50 Hz.  These avoid sudden servo
    # snaps and ESC command jumps.
    max_throttle_delta_per_step: float = 0.04
    max_steer_delta_per_step: float = 0.06

    # LiDAR safety sectors.  Beams are selected from angle ranges relative to
    # car heading.  If front clearance is very low, brake immediately.
    front_sector_deg: float = 30.0
    caution_distance_m: float = 1.10
    stop_distance_m: float = 0.55

    # When near a wall/obstacle, allow only this much forward throttle.  If the
    # model asks for more, it is reduced smoothly or converted to braking.
    caution_throttle_cmd: float = 0.08
    emergency_brake_cmd: float = -0.85

    # If model observations are invalid, fail safe by braking.
    brake_on_invalid_lidar: bool = True

    # Optional IMU-derived hardware-safety rules. These catch real RC issues
    # LiDAR alone may miss: spinout, lateral slide, hard impact, stale IMU.
    use_imu_safety: bool = False
    max_yaw_rate_rad_s: float = 2.4
    max_lat_acc_m_s2: float = 4.5
    impact_acc_m_s2: float = 10.0
    brake_on_imu_dropout: bool = False


class SafetyActionWrapper:
    """Stateful action safety filter with speed, LiDAR, and slew-rate limits."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config if config is not None else SafetyConfig()
        self.prev_safe_action = np.zeros(2, dtype=np.float64)
        self.last_front_min_m = LIDAR_MAX_RANGE
        self.last_interventions: list[str] = []

    def reset(self):
        self.prev_safe_action[:] = 0.0
        self.last_front_min_m = LIDAR_MAX_RANGE
        self.last_interventions = []

    def filter_action(self, raw_action, obs, beam_angles_rad=None, imu_meta=None) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a safe action and metadata.

        Parameters
        ----------
        raw_action:
            Model action [throttle, steer] in [-1, 1].
        obs:
            Observation [speed_norm, yaw_rate_norm, steer_norm, lidar_norm...].
        beam_angles_rad:
            Optional array matching lidar beams.  If omitted, the front sector
            uses the middle beams.
        """
        cfg = self.config
        raw = np.array(raw_action, dtype=np.float64).flatten()[:2]
        safe = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        safe = np.clip(safe, -1.0, 1.0)
        interventions: list[str] = []

        if not cfg.enabled:
            safe = np.clip(safe, -1.0, 1.0)
            meta = self._meta(raw, safe, obs, interventions, front_min_m=LIDAR_MAX_RANGE)
            self.prev_safe_action = safe.copy()
            self.last_interventions = interventions
            return safe, meta

        obs_arr = np.array(obs, dtype=np.float64).flatten()
        speed_mps = float(np.clip(obs_arr[0], 0.0, 1.0) * SPD_MAX) if obs_arr.size else 0.0
        # Observation can be V2 base (23) or V3 extended (36). LiDAR is always
        # the first 20 values after speed/yaw/steer; do not let IMU/memory tail
        # contaminate front-distance safety checks.
        lidar_norm = obs_arr[3:23] if obs_arr.size > 3 else np.array([], dtype=np.float64)
        lidar_m = np.nan_to_num(lidar_norm, nan=0.0, posinf=1.0, neginf=0.0) * LIDAR_MAX_RANGE
        invalid_lidar = bool(lidar_m.size == 0 or np.any(~np.isfinite(lidar_norm)) or np.any(lidar_norm < 0.0))

        front_min_m = self._front_min(lidar_m, beam_angles_rad, cfg.front_sector_deg)
        self.last_front_min_m = front_min_m

        # Absolute command envelope.
        before = safe.copy()
        safe[0] = np.clip(safe[0], -cfg.max_brake_cmd, cfg.max_throttle_cmd)
        safe[1] = np.clip(safe[1], -cfg.max_steer_cmd, cfg.max_steer_cmd)
        if not np.allclose(before, safe):
            interventions.append("command_cap")

        # Speed governor: no positive throttle above cap; brake harder if well above.
        if speed_mps >= cfg.max_speed_mps and safe[0] > 0.0:
            safe[0] = 0.0
            interventions.append("speed_cap_cut_throttle")
        if speed_mps >= cfg.max_speed_mps * 1.20:
            target_brake = -min(cfg.max_brake_cmd, 0.20 + 0.25 * (speed_mps - cfg.max_speed_mps))
            if safe[0] > target_brake:
                safe[0] = target_brake
                interventions.append("speed_cap_brake")

        # LiDAR stop/caution.  This is a forward-sector rule; side wall hugging
        # is still visible in telemetry but not over-controlled here.
        if invalid_lidar and cfg.brake_on_invalid_lidar:
            safe[0] = cfg.emergency_brake_cmd
            safe[1] = 0.0
            interventions.append("invalid_lidar_brake")
        elif front_min_m <= cfg.stop_distance_m:
            safe[0] = cfg.emergency_brake_cmd
            safe[1] = 0.0
            interventions.append("front_emergency_stop")
        elif front_min_m <= cfg.caution_distance_m:
            if safe[0] > cfg.caution_throttle_cmd:
                safe[0] = cfg.caution_throttle_cmd
                interventions.append("front_caution_throttle")
            # Damp steering near close obstacles to reduce snap/spin.
            safe[1] *= 0.75
            interventions.append("front_caution_steer_damp")

        if cfg.use_imu_safety and imu_meta:
            events = set(imu_meta.get("events", []))
            yaw_abs = abs(float(imu_meta.get("yaw_rate_rad_s", 0.0)))
            lat_abs = abs(float(imu_meta.get("lat_acc_m_s2", 0.0)))
            acc_norm = abs(float(imu_meta.get("acc_norm_m_s2", 0.0)))
            imu_valid = bool(imu_meta.get("valid", True))
            if (not imu_valid) and cfg.brake_on_imu_dropout:
                safe[0] = min(safe[0], -0.30)
                safe[1] *= 0.50
                interventions.append("imu_dropout_caution")
            if yaw_abs > cfg.max_yaw_rate_rad_s or "spin_risk" in events:
                safe[0] = min(safe[0], -0.20)
                safe[1] *= 0.45
                interventions.append("imu_spin_damp")
            if lat_abs > cfg.max_lat_acc_m_s2 or "lateral_slip_risk" in events:
                safe[0] = min(safe[0], 0.0)
                safe[1] *= 0.65
                interventions.append("imu_slip_damp")
            if acc_norm > cfg.impact_acc_m_s2 or "impact_or_bump" in events:
                safe[0] = cfg.emergency_brake_cmd
                safe[1] = 0.0
                interventions.append("imu_impact_brake")

        # Slew-rate limits.  Positive throttle changes are slow, but safety
        # braking must be allowed to come in immediately; otherwise an already
        # accelerating car could keep receiving positive throttle for several
        # frames after an overspeed or obstacle trigger.
        prev = self.prev_safe_action
        throttle_delta = safe[0] - prev[0]
        fast_brake_reasons = {
            "front_emergency_stop",
            "invalid_lidar_brake",
            "speed_cap_brake",
        }
        allow_fast_brake = any(reason in interventions for reason in fast_brake_reasons) and safe[0] < prev[0]
        if not allow_fast_brake:
            throttle_delta = float(np.clip(throttle_delta, -cfg.max_throttle_delta_per_step, cfg.max_throttle_delta_per_step))
            slewed_throttle = prev[0] + throttle_delta
            if not np.isclose(slewed_throttle, safe[0]):
                interventions.append("throttle_slew")
            safe[0] = slewed_throttle

        steer_delta = float(np.clip(safe[1] - prev[1], -cfg.max_steer_delta_per_step, cfg.max_steer_delta_per_step))
        slewed_steer = prev[1] + steer_delta
        if not np.isclose(slewed_steer, safe[1]):
            interventions.append("steer_slew")
        safe[1] = slewed_steer

        safe[0] = np.clip(safe[0], -cfg.max_brake_cmd, cfg.max_throttle_cmd)
        safe[1] = np.clip(safe[1], -cfg.max_steer_cmd, cfg.max_steer_cmd)

        meta = self._meta(raw, safe, obs_arr, interventions, front_min_m=front_min_m)
        self.prev_safe_action = safe.copy()
        self.last_interventions = interventions
        return safe, meta

    def _front_min(self, lidar_m: np.ndarray, beam_angles_rad, front_sector_deg: float) -> float:
        if lidar_m.size == 0:
            return 0.0
        if beam_angles_rad is None:
            center = lidar_m.size // 2
            half = max(1, lidar_m.size // 10)
            sector = lidar_m[max(0, center - half): min(lidar_m.size, center + half + 1)]
        else:
            angles = np.array(beam_angles_rad, dtype=np.float64).flatten()
            mask = np.abs(np.degrees(angles)) <= front_sector_deg
            sector = lidar_m[mask] if np.any(mask) else lidar_m
        return float(np.min(sector))

    def _meta(self, raw, safe, obs, interventions, front_min_m: float) -> dict[str, Any]:
        speed_mps = float(np.clip(obs[0], 0.0, 1.0) * SPD_MAX) if len(obs) else 0.0
        return {
            "raw_throttle_cmd": float(raw[0]),
            "raw_steer_cmd": float(raw[1]),
            "safe_throttle_cmd": float(safe[0]),
            "safe_steer_cmd": float(safe[1]),
            "speed_mps": speed_mps,
            "front_min_lidar_m": float(front_min_m),
            "interventions": interventions,
            "intervention_count": len(interventions),
        }

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
