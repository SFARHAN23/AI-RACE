"""
lidar_env.py — Gym-style LiDAR racing environment for 1/10 RC cars.

Observation space  (STATE_DIM = 23):
    [speed_norm, yaw_rate_norm, steer_norm, lidar_0, ..., lidar_19]

Action space  (ACTION_DIM = 2):
    [throttle/brake ∈ [-1,1],  steer ∈ [-1,1]]

The track has HARD border walls.  If ANY corner of the car body exits
the track boundaries the episode terminates with a crash penalty.
"""

import numpy as np

from rc_track import RCTrackClass
from rc_car_model import (
    RCCarModelClass,
    CarLength,
    CarWidth,
    SPD_MAX,
    MaxSteer,
    DT,
    K_THROTTLE,
    K_BRAKE,
)
from lidar_sim import LidarSimulator
from domain_randomization import (
    DRConfig,
    apply_lidar_noise,
    sample_friction_multiplier,
    sample_throttle_multiplier,
    sample_latency,
    sample_realism_params,
    perturb_walls,
    LIDAR_MAX_RANGE,
    LIDAR_MIN_RANGE,
)
from safety_wrapper import SafetyActionWrapper, SafetyConfig
from reward_profiles import RewardProfile, get_reward_profile
from imu_simulator import IMUConfig, SimulatedIMU, normalize_imu
from track_memory import SectorTrackMemory, TrackMemoryConfig


# ── Constants ─────────────────────────────────────────────────────────────────
N_BEAMS = 20
BASE_STATE_DIM = 3 + N_BEAMS   # 23: speed/yaw/steer + LiDAR
IMU_STATE_DIM = 3              # yaw_rate, long accel, lateral accel
MEMORY_STATE_DIM = 10          # compact outlap sector memory
STATE_DIM = BASE_STATE_DIM
EXTENDED_STATE_DIM = BASE_STATE_DIM + IMU_STATE_DIM + MEMORY_STATE_DIM
ACTION_DIM = 2

LIDAR_ANGLE_MIN = -2.3562  # -135 deg
LIDAR_ANGLE_MAX = 2.3562   # +135 deg

# Collision uses the full car body rectangle
HALF_LENGTH = CarLength / 2.0
HALF_WIDTH = CarWidth / 2.0

# Episode limits
MAX_STEPS = 2_000           # 40 seconds at 50 Hz
LAP_WRAP_THRESHOLD = 0.5    # fraction of total_trip to detect wrap


class LidarRacingEnv:
    """LiDAR-observation racing environment with hard wall boundaries."""

    def __init__(
        self,
        dr_config: DRConfig | None = None,
        track_name: str | None = None,
        safe_training: bool = False,
        safety_config: SafetyConfig | None = None,
        safety_action_penalty: float = 0.35,
        safety_intervention_penalty: float = 0.04,
        safety_front_margin_m: float = 0.80,
        safety_front_penalty: float = 0.25,
        random_start_speed_min: float = 0.5,
        random_start_speed_max: float = 2.0,
        reward_profile: str | RewardProfile | None = "slow_completion",
        sector_count: int = 12,
        use_imu_memory: bool = False,
        imu_config: IMUConfig | None = None,
        target_laps: int = 1,
        routelet_length_m: float | None = None,
        adaptive_spawn: bool = True,
        spawn_lateral_frac: float = 0.42,
        spawn_heading_jitter_rad: float = 0.12,
        spawn_allow_curves: bool = True,
        push_lap_start: int = 1,
        push_target_speed_mps: float | None = None,
        max_steps: int = MAX_STEPS,
    ):
        # Track
        self.track_name = track_name
        self.track = RCTrackClass(track_name=track_name)

        # LiDAR
        self.lidar = LidarSimulator(
            n_beams=N_BEAMS,
            angle_min=LIDAR_ANGLE_MIN,
            angle_max=LIDAR_ANGLE_MAX,
            max_range=LIDAR_MAX_RANGE,
            min_range=LIDAR_MIN_RANGE,
        )
        self._base_walls = self.track.get_wall_segments().copy()
        self.lidar.set_walls(self._base_walls)

        # Domain randomization
        self.dr_config = dr_config if dr_config is not None else DRConfig()

        # Optional real-car-envelope training. When enabled, the policy's raw
        # action is safety-filtered before physics so TD3 learns the commands it
        # will actually be allowed to send on hardware, instead of learning an
        # aggressive policy that only works when unclamped.
        self.safe_training = bool(safe_training)
        self.safety = SafetyActionWrapper(safety_config) if self.safe_training else None
        self.safety_action_penalty = float(safety_action_penalty)
        self.safety_intervention_penalty = float(safety_intervention_penalty)
        self.safety_front_margin_m = float(safety_front_margin_m)
        self.safety_front_penalty = float(safety_front_penalty)
        self._last_raw_action = np.zeros(2, dtype=np.float64)
        self._last_applied_action = np.zeros(2, dtype=np.float64)
        self._last_safety_meta: dict = {}
        self.random_start_speed_min = float(random_start_speed_min)
        self.random_start_speed_max = float(random_start_speed_max)
        self.reward_profile = reward_profile if isinstance(reward_profile, RewardProfile) else get_reward_profile(reward_profile)
        self.sector_count = max(1, int(sector_count))
        self._sector_len = self.track.total_trip / self.sector_count
        self._current_sector = 0
        self._best_sector = 0
        self._sector_advances = 0
        self._sector_advances_this_step = 0
        self.use_imu_memory = bool(use_imu_memory)
        self.imu_config = imu_config or IMUConfig(enabled=self.use_imu_memory)
        self.imu = SimulatedIMU(self.imu_config) if self.use_imu_memory else None
        self.track_memory = SectorTrackMemory(TrackMemoryConfig(enabled=self.use_imu_memory, sector_count=self.sector_count)) if self.use_imu_memory else None
        self.observation_dim = EXTENDED_STATE_DIM if self.use_imu_memory else BASE_STATE_DIM
        self._last_imu_sample = np.zeros(3, dtype=np.float64)
        self._last_imu_meta: dict = {"valid": True, "events": []}

        # Adaptive-spawn / multi-lap settings.  target_laps > 1 lets training
        # expose the second-lap centerline-pull problem instead of terminating
        # as soon as lap 1 looks good.
        self.target_laps = max(1, int(target_laps))
        self.routelet_length_m = float(routelet_length_m) if routelet_length_m is not None and routelet_length_m > 0 else None
        self.adaptive_spawn = bool(adaptive_spawn)
        self.spawn_lateral_frac = float(spawn_lateral_frac)
        self.spawn_heading_jitter_rad = float(spawn_heading_jitter_rad)
        self.spawn_allow_curves = bool(spawn_allow_curves)
        # Outlap/push-lap curriculum: train one policy to drive the early lap(s)
        # conservatively, then use the remembered sector signals to push harder
        # after lap_count reaches push_lap_start. This keeps the observation shape
        # unchanged, so existing IMU+memory checkpoints can be warm-started.
        self.push_lap_start = max(0, int(push_lap_start))
        self.push_target_speed_mps = None if push_target_speed_mps is None else float(push_target_speed_mps)
        self.max_steps = max(1, int(max_steps))
        self.lap_count = 0
        self._last_lap_count = 0
        self._target_lane_offset = 0.0
        self._prev_abs_lane_error = 0.0
        self._last_raw_action_prev = np.zeros(2, dtype=np.float64)
        self._last_applied_action_prev = np.zeros(2, dtype=np.float64)
        self._raw_action_delta = np.zeros(2, dtype=np.float64)
        self._applied_action_delta = np.zeros(2, dtype=np.float64)
        self._intermediate_lap_event = False

        # Episode-level DR multipliers (set at reset)
        self._friction_mult = 1.0
        self._throttle_mult = 1.0
        self._latency_steps = 0
        self._obs_buffer: list[np.ndarray] = []
        self._realism = sample_realism_params(DRConfig())
        self._control_buffer: list[np.ndarray] = []
        self._lagged_action = np.zeros(2, dtype=np.float64)

        # Car (created at reset)
        self.car: RCCarModelClass | None = None

        # Episode state flags
        self.DONE = False
        self.COLLISION = False
        self.WRONG_DIR = False
        self.STOPPED = False
        self.FINISHED = False
        self.TIME_LIMIT = False

        # Tracking
        self.step_count = 0
        self._prev_trip = 0.0
        self._total_progress = 0.0
        self._step_delta = 0.0   # per-step progress delta [m]
        self._lap_started = False

    # ── resets ─────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Random-start reset with domain randomization."""
        if self.adaptive_spawn:
            pose = self.track.random_car_pose(
                lateral_frac=self.spawn_lateral_frac,
                heading_jitter_rad=self.spawn_heading_jitter_rad,
                allow_curves=self.spawn_allow_curves,
            )
        else:
            pose = self.track.random_car_pose(lateral_frac=0.25, heading_jitter_rad=0.05, allow_curves=False)
        low = min(self.random_start_speed_min, self.random_start_speed_max)
        high = max(self.random_start_speed_min, self.random_start_speed_max)
        spd = float(np.random.uniform(low, high))
        return self._do_reset(pose, spd, apply_dr=True)

    def test_reset(self, spd: float = 1.0, lateral_offset_m: float = 0.0, trip_m: float = 0.0) -> np.ndarray:
        """Fixed reset (configurable progress/lateral offset/speed, no DR)."""
        pose = self.track.test_reset_pose(lateral_offset_m=lateral_offset_m, trip_m=trip_m)
        return self._do_reset(pose, spd=float(spd), apply_dr=False)

    def _do_reset(self, pose, spd, apply_dr: bool) -> np.ndarray:
        """Internal reset helper."""
        # Create / reset car
        if self.car is None:
            self.car = RCCarModelClass(pose, spd)
        else:
            self.car.reset(pose, spd)

        # Episode flags
        self.DONE = False
        self.COLLISION = False
        self.WRONG_DIR = False
        self.STOPPED = False
        self.FINISHED = False
        self.TIME_LIMIT = False
        self.step_count = 0
        self._total_progress = 0.0
        self._lap_started = False
        self.lap_count = 0
        self._last_lap_count = 0
        self._intermediate_lap_event = False

        # Initial trip / target lane.  The target lane offset is intentionally
        # taken from the spawn pose so non-center starts do not get rewarded for
        # snapping back to the centerline, especially on lap 2.
        self.track.findcar(self.car.pose[:2])
        self._prev_trip = self.track.car_trip
        self._target_lane_offset = float(self.track.centerlinedist)
        self._prev_abs_lane_error = 0.0
        self._current_sector = self._sector_index(self._prev_trip)
        self._best_sector = self._current_sector
        self._sector_advances = 0
        self._sector_advances_this_step = 0

        # Domain randomization (episode-level)
        if apply_dr:
            self._friction_mult = sample_friction_multiplier(self.dr_config)
            self._throttle_mult = sample_throttle_multiplier(self.dr_config)
            self._latency_steps = sample_latency(self.dr_config)
            self._realism = sample_realism_params(self.dr_config)

            # Wall perturbation
            if self.dr_config.wall_noise_std > 0:
                walls = perturb_walls(self._base_walls, self.dr_config)
                self.lidar.set_walls(walls)
            else:
                self.lidar.set_walls(self._base_walls)
        else:
            self._friction_mult = 1.0
            self._throttle_mult = 1.0
            self._latency_steps = 0
            self._realism = sample_realism_params(DRConfig())
            self.lidar.set_walls(self._base_walls)

        # Observation buffer (for latency injection)
        self._obs_buffer = []
        if self.safety is not None:
            self.safety.reset()
        if self.imu is not None:
            self.imu.reset()
        if self.track_memory is not None:
            self.track_memory.reset()
        self._last_imu_sample[:] = 0.0
        self._last_imu_meta = {"valid": True, "events": []}
        self._last_raw_action[:] = 0.0
        self._last_applied_action[:] = 0.0
        self._last_raw_action_prev[:] = 0.0
        self._last_applied_action_prev[:] = 0.0
        self._raw_action_delta[:] = 0.0
        self._applied_action_delta[:] = 0.0
        self._last_safety_meta = {}
        self._control_buffer = []
        self._lagged_action[:] = 0.0

        return self.observe()

    # ── step ───────────────────────────────────────────────────────────────

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the environment by one time-step.

        Parameters
        ----------
        action : array-like [ux, uy] each in [-1, 1]

        Returns
        -------
        (obs, reward, done, info)
        """
        action = np.array(action, dtype=np.float64).flatten()[:2]
        action = np.clip(action, -1.0, 1.0)
        raw_action = action.copy()

        # Optional safety-in-the-loop filtering, based on the pre-step
        # observation. This mirrors the laptop/Arduino safety wrapper used for
        # rollouts and dry-run hardware staging.
        safety_meta = {}
        if self.safety is not None:
            pre_obs = self.observe()
            action, safety_meta = self.safety.filter_action(raw_action, pre_obs, imu_meta=self._last_imu_meta)

        self._last_raw_action_prev = self._last_raw_action.copy()
        self._last_applied_action_prev = self._last_applied_action.copy()
        self._last_raw_action = raw_action.copy()
        self._raw_action_delta = self._last_raw_action - self._last_raw_action_prev

        # Apply actuator/ESC/servo real-car imperfections after the safety
        # wrapper.  This lets training experience the exact problems hardware
        # introduces: command delay, low-voltage weak throttle, neutral deadband,
        # servo trim error, and actuator lag.
        action = self._apply_actuator_realism(action)
        self._last_applied_action = action.copy()
        self._applied_action_delta = self._last_applied_action - self._last_applied_action_prev
        self._last_safety_meta = safety_meta

        # Apply DR multipliers
        modified_action = action.copy()
        modified_action[0] *= self._throttle_mult

        # Scale car physics by friction multiplier
        # (Temporarily modify module-level constants is ugly, so we
        #  scale the force directly in the action instead.)
        # For friction: scale both throttle and brake capability
        modified_action[0] *= self._friction_mult

        # Step car physics
        self.car.step(modified_action)
        if self._realism.get("yaw_jitter_std", 0.0) > 0.0:
            # Small heading jitter represents bumps, tyre scrub, and slick/drift
            # surface micro-slips not captured by the kinematic bicycle model.
            jitter_scale = 1.0 + min(abs(float(self.car.lat_acc)) / 5.0, 2.0)
            self.car.pose[2] += float(np.random.normal(0.0, self._realism["yaw_jitter_std"] * jitter_scale))
            self.car.pose[2] = (self.car.pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.step_count += 1

        # Locate car on track (sets centerlinedist, track_dir, car_trip)
        on_track = self.track.findcar(self.car.pose[:2])

        # Track progress
        self._update_progress()

        # IMU + outlap memory update. These simulate real RC failure signals
        # (spin/slip/impact/stale sensor) and remember sector-level hazards for
        # later push laps. The sector key is sim-only unless hardware localization
        # is added.
        if self.imu is not None:
            self._last_imu_sample, self._last_imu_meta = self.imu.read(self.car, applied_action=action)
        if self.track_memory is not None:
            front_scan = self._front_lidar_min_m()
            safety_count = int(safety_meta.get("intervention_count", 0))
            self.track_memory.update(
                self._current_sector,
                speed_mps=self.car.spd,
                yaw_rate_rad_s=float(self._last_imu_sample[0]) if self.imu is not None else self.car.psi_dot,
                front_clearance_m=front_scan,
                safety_count=safety_count,
                finished_lap=False,
            )

        # Check termination (MUST be before reward)
        self._check_done(on_track)

        # Compute reward
        r = self._reward()

        # Build observation
        obs = self.observe()

        # Apply latency
        if self._latency_steps > 0:
            self._obs_buffer.append(obs)
            if len(self._obs_buffer) > self._latency_steps:
                obs = self._obs_buffer.pop(0)
            else:
                obs = self._obs_buffer[0]  # return oldest available

        info = {
            "speed": self.car.spd,
            "trip": self.track.car_trip,
            "centerline_dist": self.track.centerlinedist,
            "progress": self._total_progress,
            "reason": self.query_fail_reason() if self.DONE else "",
            "raw_action": self._last_raw_action.copy(),
            "applied_action": self._last_applied_action.copy(),
            "safety": safety_meta,
            "lap_count": self.lap_count,
            "target_laps": self.target_laps,
            "routelet_length_m": self.routelet_length_m,
            "lap_progress_pct": 100.0 * ((self._total_progress % self.track.total_trip) / max(self.track.total_trip, 1e-6)),
            "target_lane_offset_m": self._target_lane_offset,
            "raw_action_delta": self._raw_action_delta.copy(),
            "applied_action_delta": self._applied_action_delta.copy(),
            "sector_index": self._current_sector,
            "best_sector": self._best_sector,
            "sector_advances": self._sector_advances,
            "sector_advance_this_step": self._sector_advances_this_step,
            "reward_profile": self.reward_profile.name,
            "push_lap_start": self.push_lap_start,
            "active_target_speed_mps": self._active_target_speed_mps(),
            "imu": self._last_imu_meta,
            "track_memory": self.track_memory.summary() if self.track_memory is not None else None,
            "realism": dict(self._realism),
        }

        return obs, r, self.DONE, info

    def _active_target_speed_mps(self) -> float:
        """Lap-aware speed target: conservative outlap(s), faster push lap."""
        p = self.reward_profile
        if self.push_target_speed_mps is not None and self.lap_count >= self.push_lap_start:
            return max(0.05, float(self.push_target_speed_mps))
        return max(0.05, float(p.target_speed_mps))

    # ── real-car imperfection model ─────────────────────────────────────────

    def _apply_actuator_realism(self, action: np.ndarray) -> np.ndarray:
        """Apply command-side sim-to-real effects before car physics.

        This intentionally lives outside the safety wrapper: the policy may ask
        for a safe command, but real hardware can still lag, have ESC deadband,
        servo trim, calibration error, or delayed packets.
        """
        out = np.array(action, dtype=np.float64).flatten()[:2].copy()
        r = self._realism

        # Command latency: execute a previous safe command for N control ticks.
        delay = int(r.get("control_latency_steps", 0))
        if delay > 0:
            self._control_buffer.append(out.copy())
            if len(self._control_buffer) > delay:
                out = self._control_buffer.pop(0)
            else:
                out = self._control_buffer[0].copy()

        # ESC neutral deadband.  Tiny throttle commands often do nothing.
        db = float(r.get("throttle_deadband", 0.0))
        if db > 0.0 and abs(out[0]) < db:
            out[0] = 0.0

        # Battery sag affects positive throttle more than braking.
        if out[0] > 0.0:
            out[0] *= float(r.get("battery_mult", 1.0))

        # Servo trim/calibration; clipped because hardware has hard travel limits.
        out[1] = out[1] * float(r.get("steering_response", 1.0)) + float(r.get("steer_bias", 0.0))
        out = np.clip(out, -1.0, 1.0)

        # First-order actuator lag after all calibration effects.
        alpha = float(np.clip(r.get("actuator_lag_alpha", 1.0), 0.0, 1.0))
        self._lagged_action = alpha * out + (1.0 - alpha) * self._lagged_action
        return np.clip(self._lagged_action.copy(), -1.0, 1.0)

    # ── observe ────────────────────────────────────────────────────────────

    def observe(self) -> np.ndarray:
        """Build observation vector.

        Base V2 shape is 23. V3 optional IMU+memory shape is 36:
        [base LiDAR policy input, normalized IMU, sector memory features].
        """
        x, y, heading = self.car.pose

        # LiDAR scan
        raw_scan = self.lidar.scan(x, y, heading)
        scan = apply_lidar_noise(raw_scan, self.dr_config)

        # Normalize scan to [0, 1]
        scan_norm = scan / LIDAR_MAX_RANGE

        # Car state (normalised)
        speed_norm = np.clip(self.car.spd / SPD_MAX, 0.0, 1.0)
        yaw_rate_norm = np.clip(self.car.psi_dot / 3.0, -1.0, 1.0)
        steer_norm = np.clip(self.car.steer / MaxSteer, -1.0, 1.0)

        base = np.concatenate([
            [speed_norm, yaw_rate_norm, steer_norm],
            scan_norm,
        ]).astype(np.float32)

        if not self.use_imu_memory:
            return base

        imu_features = normalize_imu(self._last_imu_sample, self.imu_config)
        if self.track_memory is not None:
            memory_features = self.track_memory.features(self._current_sector)
        else:
            memory_features = np.zeros(MEMORY_STATE_DIM, dtype=np.float32)
        obs = np.concatenate([base, imu_features, memory_features]).astype(np.float32)
        return obs

    def _front_lidar_min_m(self) -> float:
        """Return current front-sector LiDAR distance for memory/safety diagnostics."""
        x, y, heading = self.car.pose
        scan = self.lidar.scan(x, y, heading)
        center = scan.size // 2
        half = max(1, scan.size // 10)
        return float(np.min(scan[max(0, center - half): min(scan.size, center + half + 1)]))

    # ── reward ─────────────────────────────────────────────────────────────

    def _reward(self) -> float:
        """Compute per-step reward for slow, safe forward driving.

        The 9-hour conservative run showed a classic failure mode: the policy
        learned that stopping/braking was safer than making progress. For the
        physical-car safety goal we do *not* want a fast policy, but we do need steady
        forward motion. This reward therefore scales progress around a low-speed
        target (not simulator max speed), penalizes reverse/brake-to-stop
        behaviour, and keeps wall/safety penalties as guardrails rather than the
        dominant signal.
        """
        p = self.reward_profile

        # Terminal rewards. STOPPED is handled in _check_done but appears as a
        # generic non-finished terminal state here, so explicitly penalize it.
        if self.COLLISION:
            return p.collision_terminal
        if self.WRONG_DIR:
            return p.wrong_dir_terminal
        if self.STOPPED:
            return p.stopped_terminal
        if self.FINISHED:
            return p.finished_terminal
        if self.TIME_LIMIT:
            # Time-limit with useful progress is not as bad as stopping; tiny
            # progress should still be clearly bad.
            target_progress = self.routelet_length_m if self.routelet_length_m is not None else self.track.total_trip
            progress_frac = self._total_progress / max(target_progress, 1e-6)
            return p.time_limit_penalty * float(np.clip(1.0 - progress_frac, 0.0, 1.0))

        # 1. Dominant signal: forward progress at a slow hardware-relevant pace.
        # At 0.6 m/s and DT=0.02, expected delta is 0.012 m. Scaling by a
        # 1.0 m/s target makes slow forward motion worth learning; the old
        # SPD_MAX scaling made slow safe progress too weak.
        active_target_speed = self._active_target_speed_mps()
        target_delta_per_step = active_target_speed * DT
        progress_rate = self._step_delta / max(target_delta_per_step, 1e-9)
        if progress_rate >= 0.0:
            r = p.progress_weight * float(np.clip(progress_rate, 0.0, p.progress_clip))
        else:
            # Reversing/backwards progress must be worse than simply going slow.
            r = p.reverse_weight * float(np.clip(progress_rate, -2.0, 0.0)) - p.reverse_bias

        # 2. Small alive bonus only when actually moving forward. This prevents
        # stationary policies from collecting harmless-looking rewards.
        if self._step_delta > 0.002 and self.car.spd >= 0.15:
            r += p.alive_forward_bonus

        if self._sector_advances_this_step > 0:
            r += p.sector_bonus * self._sector_advances_this_step
        elif self.step_count > 50:
            r -= p.sector_repeat_penalty

        # 3. Stop/reverse shaping. Braking is fine near danger through the safety
        # wrapper, but repeatedly asking for negative throttle at very low speed
        # is exactly the failure we saw in the completed run.
        raw_throttle = float(self._last_raw_action[0])
        applied_throttle = float(self._last_applied_action[0])
        if self.car.spd < 0.18:
            r -= p.low_speed_penalty
            if raw_throttle < 0.0:
                r -= p.low_speed_reverse_penalty * abs(raw_throttle)
        elif raw_throttle < -0.15 and applied_throttle < 0.05:
            r -= p.brake_penalty * abs(raw_throttle)

        # 4. Encourage stable low-speed band rather than aggression.
        active_high_speed_threshold = max(p.high_speed_threshold_mps, active_target_speed * 1.35)
        if 0.25 <= self.car.spd <= max(active_high_speed_threshold, active_target_speed) and self._step_delta > 0.0:
            r += p.stable_speed_bonus
        elif self.car.spd > active_high_speed_threshold:
            r -= p.high_speed_penalty * min((self.car.spd - active_high_speed_threshold) / max(active_high_speed_threshold, 1e-6), 1.0)

        # 5. Wall proximity: guardrail, not stronger than progress.
        wall_closeness = abs(self.track.centerlinedist) / max(self.track.width / 2, 1e-6)
        if wall_closeness > p.wall_start_frac:
            r -= p.wall_penalty * float(np.clip((wall_closeness - p.wall_start_frac) / max(1.0 - p.wall_start_frac, 1e-6), 0.0, 1.5))

        # 6. Smooth commands help real hardware and reduce oscillation.
        r -= p.steer_penalty * abs(float(self._last_applied_action[1]))
        r -= p.steer_delta_penalty * abs(float(self._applied_action_delta[1]))
        r -= p.throttle_delta_penalty * abs(float(self._applied_action_delta[0]))
        r -= p.yaw_rate_penalty * min(abs(float(self.car.psi_dot)), 3.0) / 3.0
        r -= 0.015 * abs(raw_throttle - applied_throttle)

        # 6b. Multi-lap lane consistency: once the car starts off-center, lap 2
        # should continue safely in that lane instead of reflexively pulling back
        # to the centerline.  The penalty is mild on lap 1 and stronger after a
        # completed wrap, so it trains the exact failure Farhan observed.
        lane_error = abs(float(self.track.centerlinedist) - self._target_lane_offset)
        if self.lap_count >= 1:
            r -= p.lap2_lane_error_penalty * min(lane_error / max(self.track.width * 0.5, 1e-6), 1.5)
            if abs(self._target_lane_offset) > 0.08:
                center_pull = max(0.0, abs(self._target_lane_offset) - abs(float(self.track.centerlinedist)))
                r -= p.lap2_center_pull_penalty * min(center_pull / max(abs(self._target_lane_offset), 1e-6), 1.0)
        if self._intermediate_lap_event:
            r += p.intermediate_lap_bonus
            self._intermediate_lap_event = False

        # 7. Safety-in-the-loop corrections remain penalties, but lower than
        # before so the policy still learns to move forward while becoming safer.
        if self.safe_training:
            action_delta = np.abs(self._last_raw_action - self._last_applied_action)
            r -= p.action_filter_penalty * self.safety_action_penalty * float(np.mean(action_delta))

            intervention_count = int(self._last_safety_meta.get("intervention_count", 0))
            r -= p.intervention_penalty * self.safety_intervention_penalty * intervention_count

            front_min = self._last_safety_meta.get("front_min_lidar_m")
            if front_min is not None and front_min < self.safety_front_margin_m:
                closeness = (self.safety_front_margin_m - float(front_min)) / max(self.safety_front_margin_m, 1e-6)
                r -= p.front_penalty * self.safety_front_penalty * float(np.clip(closeness, 0.0, 1.0))

        return float(r)

    # ── termination checks ─────────────────────────────────────────────────

    def _check_done(self, on_track_center: bool):
        """Check all termination conditions."""
        # 1. Wall collision — HARD WALLS: even a wheel can't exceed!
        #    Check all 4 body corners
        corners = self.car.get_body_corners()
        for corner in corners:
            if not self.track.findcar(corner.tolist()):
                self.COLLISION = True
                self.DONE = True
                # Restore findcar state for the car center
                self.track.findcar(self.car.pose[:2])
                return

        # Also fail if the center itself is off-track
        if not on_track_center:
            self.COLLISION = True
            self.DONE = True
            return

        # 2. Wrong direction
        angle_error = abs(
            self._angle_diff(self.track.track_dir, self.car.pose[2])
        )
        if angle_error > np.pi / 2:
            self.WRONG_DIR = True
            self.DONE = True
            return

        # 3. No-progress check. The previous 200-step / 0.5 m rule made slow
        # safe policies terminate after only ~4 s and encouraged brake-to-stop
        # local minima. Give slow hardware-style policies enough time to begin
        # moving, but still end truly stationary episodes.
        if self.step_count > 500 and self._total_progress < 0.35:
            self.STOPPED = True
            self.DONE = True
            return
        if self.step_count > 300 and self.car.spd < 0.08 and self._total_progress < 0.20:
            self.STOPPED = True
            self.DONE = True
            return

        # 4. Routelet/local-section completion.  This is used for long F1-style
        # tracks where full laps are too sparse for early RL.  It lets an
        # episode succeed after making N metres of forward progress from a
        # random/adaptive spawn, while still using the same collision, stop,
        # wrong-direction, safety, and reward logic.
        if self.routelet_length_m is not None and self._total_progress >= self.routelet_length_m:
            self.FINISHED = True
            self.DONE = True
            return

        # 5. Target multi-lap completion.  Do not terminate at lap 1 when the
        # training/eval target is 2+ laps; that is how the lap-2 instability is
        # exposed and fixed.
        completed_now = self._completed_laps()
        if completed_now > self.lap_count:
            self.lap_count = completed_now
            if self.track_memory is not None:
                self.track_memory.update(
                    self._current_sector,
                    speed_mps=self.car.spd,
                    yaw_rate_rad_s=float(self._last_imu_sample[0]) if self.imu is not None else self.car.psi_dot,
                    front_clearance_m=self._front_lidar_min_m(),
                    safety_count=int(self._last_safety_meta.get("intervention_count", 0)),
                    finished_lap=True,
                )
            if self.lap_count >= self.target_laps:
                self.FINISHED = True
                self.DONE = True
                return
            self._intermediate_lap_event = True

        # 6. Time limit
        if self.step_count >= self.max_steps:
            self.TIME_LIMIT = True
            self.DONE = True

    def _update_progress(self):
        """Track cumulative forward progress along the centerline."""
        current_trip = self.track.car_trip
        total = self.track.total_trip

        # Delta trip (handle wrap-around)
        delta = current_trip - self._prev_trip
        if delta < -total * 0.5:
            delta += total  # forward wrap
        elif delta > total * 0.5:
            delta -= total  # backward wrap

        self._step_delta = delta   # store for reward function
        self._total_progress += delta
        self._prev_trip = current_trip

        # Virtual sector tracking for simulator diagnostics/reward shaping.
        # This relies on simulator centerline progress; on real hardware this
        # would need localization/timing gates/markers, not LiDAR-only ranges.
        old_sector = self._current_sector
        new_sector = self._sector_index(current_trip)
        self._sector_advances_this_step = 0
        if delta > 0 and new_sector != old_sector:
            advance = (new_sector - old_sector) % self.sector_count
            if 0 < advance <= max(1, self.sector_count // 2):
                self._sector_advances_this_step = advance
                self._sector_advances += advance
                self._best_sector = max(self._best_sector, new_sector)
        self._current_sector = new_sector

    def _sector_index(self, trip: float) -> int:
        """Return virtual sector index for simulator progress diagnostics."""
        if self.sector_count <= 1:
            return 0
        return int(np.floor((float(trip) % self.track.total_trip) / self._sector_len)) % self.sector_count

    def _completed_laps(self) -> int:
        """Return completed lap count using the same 98% tolerance as earlier evals."""
        return int(np.floor(max(0.0, self._total_progress) / max(self.track.total_trip, 1e-6) + 0.02))

    def _check_lap_completion(self) -> bool:
        """Backward-compatible one-lap completion helper."""
        return self._completed_laps() >= 1

    # ── utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Signed difference (a1 - a2) normalised to [-pi, pi]."""
        d = a1 - a2
        return float((d + np.pi) % (2 * np.pi) - np.pi)

    def query_fail_reason(self) -> str:
        """Return a human-readable termination reason."""
        if self.COLLISION:
            return "COLLISION"
        if self.WRONG_DIR:
            return "WRONG_DIR"
        if self.STOPPED:
            return "STOPPED"
        if self.FINISHED:
            return "FINISHED"
        if self.TIME_LIMIT:
            return "TIME_LIMIT"
        return "RUNNING"


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== LidarRacingEnv Self-Test ===")

    env = LidarRacingEnv()
    print(f"  STATE_DIM  = {STATE_DIM}")
    print(f"  ACTION_DIM = {ACTION_DIM}")
    print(f"  Track trip = {env.track.total_trip:.2f} m")

    # Test reset
    obs = env.test_reset()
    print(f"  test_reset obs shape: {obs.shape}")
    print(f"  test_reset obs[:5]:   {obs[:5]}")
    assert obs.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {obs.shape}"

    # Random reset
    obs = env.reset()
    print(f"  random reset obs[:5]: {obs[:5]}")
    assert obs.shape == (STATE_DIM,)

    # Run 200 steps with random actions
    obs = env.test_reset()
    total_reward = 0.0
    for i in range(200):
        action = np.array([0.5, np.sin(i * 0.1) * 0.3])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"  Episode ended at step {i}: {env.query_fail_reason()}")
            break

    print(f"  Steps run   : {env.step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final speed : {env.car.spd:.3f} m/s")
    print(f"  Progress    : {env._total_progress:.3f} m")

    # Test with DR
    from domain_randomization import get_dr_config

    env_dr = LidarRacingEnv(dr_config=get_dr_config("full"))
    obs = env_dr.reset()
    print(f"\n  DR 'full' reset obs[:5]: {obs[:5]}")
    for i in range(50):
        obs, r, done, info = env_dr.step([0.3, 0.0])
        if done:
            break
    print(f"  DR episode: {env_dr.step_count} steps, reason={env_dr.query_fail_reason()}")

    # Verify collision detection (drive into wall)
    env2 = LidarRacingEnv()
    obs = env2.test_reset()
    print(f"\n  Collision test: driving hard right into wall...")
    for i in range(500):
        obs, r, done, info = env2.step([0.8, -1.0])  # full right
        if done:
            print(f"    Crashed at step {i}: {env2.query_fail_reason()}")
            break
    assert env2.COLLISION, "Should have collided with wall!"

    print("\n=== All LidarRacingEnv tests passed! ===")
