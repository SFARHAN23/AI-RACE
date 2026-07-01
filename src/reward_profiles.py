"""Tunable reward profiles for AIRACE TD3 LiDAR training.

Two-stage plan:
1. slow_completion: make the car finish laps slowly and safely.
2. lap_improvement: once completion is reliable, shift some weight toward speed/lap time.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RewardProfile:
    name: str
    target_speed_mps: float = 0.60
    progress_weight: float = 1.25
    progress_clip: float = 1.50
    reverse_weight: float = 2.50
    reverse_bias: float = 0.25
    alive_forward_bonus: float = 0.08
    stable_speed_bonus: float = 0.06
    low_speed_penalty: float = 0.45
    low_speed_reverse_penalty: float = 0.55
    brake_penalty: float = 0.12
    high_speed_penalty: float = 0.10
    high_speed_threshold_mps: float = 1.25
    wall_penalty: float = 0.45
    wall_start_frac: float = 0.55
    steer_penalty: float = 0.025
    action_filter_penalty: float = 0.35
    intervention_penalty: float = 0.50
    front_penalty: float = 0.65
    collision_terminal: float = -18.0
    wrong_dir_terminal: float = -18.0
    stopped_terminal: float = -25.0
    finished_terminal: float = 150.0
    time_limit_penalty: float = -20.0
    sector_bonus: float = 0.35
    sector_repeat_penalty: float = 0.02
    steer_delta_penalty: float = 0.18
    throttle_delta_penalty: float = 0.10
    yaw_rate_penalty: float = 0.035
    lap2_lane_error_penalty: float = 0.22
    lap2_center_pull_penalty: float = 0.18
    intermediate_lap_bonus: float = 55.0


PROFILES: dict[str, RewardProfile] = {
    "slow_completion": RewardProfile(
        name="slow_completion",
        target_speed_mps=0.60,
        progress_weight=1.45,
        alive_forward_bonus=0.10,
        stable_speed_bonus=0.08,
        low_speed_penalty=0.35,
        stopped_terminal=-30.0,
        finished_terminal=180.0,
        sector_bonus=0.45,
        steer_delta_penalty=0.22,
        throttle_delta_penalty=0.12,
        yaw_rate_penalty=0.045,
        lap2_lane_error_penalty=0.28,
        lap2_center_pull_penalty=0.22,
        intermediate_lap_bonus=65.0,
    ),
    "lap_improvement": RewardProfile(
        name="lap_improvement",
        target_speed_mps=0.90,
        progress_weight=1.15,
        alive_forward_bonus=0.06,
        stable_speed_bonus=0.03,
        low_speed_penalty=0.55,
        high_speed_threshold_mps=1.60,
        high_speed_penalty=0.06,
        stopped_terminal=-28.0,
        finished_terminal=220.0,
        sector_bonus=0.25,
        steer_delta_penalty=0.16,
        throttle_delta_penalty=0.08,
        yaw_rate_penalty=0.030,
        lap2_lane_error_penalty=0.18,
        lap2_center_pull_penalty=0.14,
        intermediate_lap_bonus=45.0,
    ),
}


def get_reward_profile(name: str | None = None, **overrides) -> RewardProfile:
    key = name or "slow_completion"
    if key not in PROFILES:
        raise ValueError(f"Unknown reward profile {key!r}; options: {sorted(PROFILES)}")
    profile = PROFILES[key]
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(profile, **clean) if clean else profile
