"""
domain_randomization.py — Noise injection for sim-to-real transfer.

Provides configurable randomization at two levels:
  - Per-step:    sensor noise applied to each LiDAR observation
  - Per-episode: physics variation applied at reset

Three preset phases (get_dr_config):
  'clean'  — no randomization (baseline training)
  'mild'   — light LiDAR noise + friction variation
  'full'   — all channels active (sensor + physics + latency)
"""

import numpy as np
from dataclasses import dataclass


# ── LiDAR constants (must match lidar_sim.py / rc_track.py) ──────────────────
LIDAR_MAX_RANGE = 12.0  # m
LIDAR_MIN_RANGE = 0.05  # m


@dataclass
class DRConfig:
    """Domain randomization configuration."""

    # ── LiDAR noise ──────────────────────────────────────────────────────
    lidar_noise_std: float = 0.0         # Gaussian noise std [m]
    lidar_dropout_rate: float = 0.0      # Fraction of beams to drop
    lidar_salt_pepper_rate: float = 0.0  # Fraction of beams to corrupt
    lidar_range_bias_std: float = 0.0    # Per-scan range offset [m]
    lidar_range_scale_range: tuple = (1.0, 1.0)  # Per-scan multiplicative range calibration
    lidar_quantization_m: float = 0.0    # Range quantization, cheap sensor resolution [m]
    lidar_full_scan_dropout_rate: float = 0.0    # Whole packet lost/stale for one control step

    # ── Physics variation ────────────────────────────────────────────────
    friction_range: tuple = (1.0, 1.0)   # (min_mult, max_mult)
    throttle_range: tuple = (1.0, 1.0)   # (min_mult, max_mult)
    battery_voltage_range: tuple = (1.0, 1.0)    # Low battery weakens positive throttle
    steering_response_range: tuple = (1.0, 1.0)  # Servo travel calibration multiplier
    steer_bias_std: float = 0.0          # Persistent steering trim error [normalized cmd]
    throttle_deadband_range: tuple = (0.0, 0.0)  # ESC neutral deadband [normalized cmd]
    actuator_lag_alpha_range: tuple = (1.0, 1.0) # 1=no lag; lower=smoother/slower actuator
    max_control_latency_steps: int = 0   # Command delay [control steps]
    yaw_jitter_std: float = 0.0          # Small heading disturbance for bumps/slip [rad/step]

    # ── Wall perturbation ────────────────────────────────────────────────
    wall_noise_std: float = 0.0          # Lateral wall offset std [m]

    # ── Latency ──────────────────────────────────────────────────────────
    max_latency_steps: int = 0           # Max observation delay [steps]


# ── Phase presets ─────────────────────────────────────────────────────────────

def get_dr_config(phase: str) -> DRConfig:
    """Return a preset DRConfig for the given training phase.

    Phases
    ------
    'clean'  No randomization at all.
    'mild'   Light sensor noise + small friction variation.
    'full'   All channels active for maximum sim-to-real robustness.
    """
    if phase == "clean":
        return DRConfig()
    elif phase == "mild":
        return DRConfig(
            lidar_noise_std=0.02,
            lidar_dropout_rate=0.05,
            friction_range=(0.85, 1.15),
            throttle_range=(0.90, 1.10),
        )
    elif phase == "full":
        return DRConfig(
            lidar_noise_std=0.04,
            lidar_dropout_rate=0.10,
            lidar_salt_pepper_rate=0.03,
            lidar_range_bias_std=0.015,
            lidar_range_scale_range=(0.985, 1.015),
            lidar_quantization_m=0.01,
            lidar_full_scan_dropout_rate=0.005,
            friction_range=(0.70, 1.30),
            throttle_range=(0.85, 1.15),
            battery_voltage_range=(0.86, 1.04),
            steering_response_range=(0.90, 1.10),
            steer_bias_std=0.025,
            throttle_deadband_range=(0.00, 0.06),
            actuator_lag_alpha_range=(0.45, 0.90),
            max_control_latency_steps=1,
            yaw_jitter_std=0.0008,
            wall_noise_std=0.015,
            max_latency_steps=2,
        )
    elif phase in ("hardware_rough", "realistic"):
        return DRConfig(
            lidar_noise_std=0.055,
            lidar_dropout_rate=0.14,
            lidar_salt_pepper_rate=0.04,
            lidar_range_bias_std=0.025,
            lidar_range_scale_range=(0.97, 1.03),
            lidar_quantization_m=0.02,
            lidar_full_scan_dropout_rate=0.012,
            friction_range=(0.60, 1.25),
            throttle_range=(0.78, 1.12),
            battery_voltage_range=(0.78, 1.03),
            steering_response_range=(0.84, 1.14),
            steer_bias_std=0.045,
            throttle_deadband_range=(0.02, 0.10),
            actuator_lag_alpha_range=(0.28, 0.75),
            max_control_latency_steps=2,
            yaw_jitter_std=0.0018,
            wall_noise_std=0.030,
            max_latency_steps=3,
        )
    # Fallback: no randomization
    return DRConfig()


# ── Per-step functions ────────────────────────────────────────────────────────

def apply_lidar_noise(
    scan: np.ndarray,
    config: DRConfig,
    max_range: float = LIDAR_MAX_RANGE,
    min_range: float = LIDAR_MIN_RANGE,
) -> np.ndarray:
    """Apply sensor noise to a raw LiDAR scan (in-place safe: works on copy).

    Noise pipeline (order matters):
        1. Additive Gaussian noise
        2. Random dropout  (set beam to max_range = "no return")
        3. Salt-and-pepper  (randomly jam beams to min or max range)
        4. Clip to valid sensor range
    """
    scan = scan.copy()
    n = len(scan)

    # 0. Whole scan dropout (simulates a dropped/stale RPLIDAR packet).  With no
    # stateful previous scan here, max range is the conservative "no reliable hit"
    # representation; the safety wrapper will see poor/invalid data in training.
    if config.lidar_full_scan_dropout_rate > 0 and np.random.rand() < config.lidar_full_scan_dropout_rate:
        scan[:] = max_range

    # 1. Gaussian noise + per-scan calibration bias/scale
    if config.lidar_noise_std > 0:
        scan += np.random.normal(0.0, config.lidar_noise_std, n)
    if config.lidar_range_bias_std > 0:
        scan += float(np.random.normal(0.0, config.lidar_range_bias_std))
    if config.lidar_range_scale_range != (1.0, 1.0):
        scan *= float(np.random.uniform(*config.lidar_range_scale_range))
    if config.lidar_quantization_m > 0:
        q = float(config.lidar_quantization_m)
        scan = np.round(scan / q) * q

    # 2. Dropout (simulate missed returns)
    if config.lidar_dropout_rate > 0:
        drop_mask = np.random.rand(n) < config.lidar_dropout_rate
        scan[drop_mask] = max_range

    # 3. Salt-and-pepper
    if config.lidar_salt_pepper_rate > 0:
        sp_mask = np.random.rand(n) < config.lidar_salt_pepper_rate
        # 50/50 salt or pepper
        salt_or_pepper = np.where(np.random.rand(n) > 0.5, max_range, min_range)
        scan[sp_mask] = salt_or_pepper[sp_mask]

    # 4. Clamp
    return np.clip(scan, min_range, max_range)


# ── Per-episode functions ─────────────────────────────────────────────────────

def sample_friction_multiplier(config: DRConfig) -> float:
    """Sample a friction multiplier for one episode."""
    return float(np.random.uniform(*config.friction_range))


def sample_throttle_multiplier(config: DRConfig) -> float:
    """Sample a throttle-response multiplier for one episode."""
    return float(np.random.uniform(*config.throttle_range))


def sample_latency(config: DRConfig) -> int:
    """Sample an observation delay in steps for one episode."""
    if config.max_latency_steps <= 0:
        return 0
    return int(np.random.randint(0, config.max_latency_steps + 1))


def sample_realism_params(config: DRConfig) -> dict:
    """Sample per-episode real-car imperfections.

    These parameters model the issues Farhan will likely see on hardware:
    battery sag, ESC deadband, servo trim/calibration error, actuator lag, and
    command latency.  They are sampled once per episode so the TD3 policy learns
    robust behavior instead of overfitting one perfect simulated car.
    """
    return {
        "battery_mult": float(np.random.uniform(*config.battery_voltage_range)),
        "steering_response": float(np.random.uniform(*config.steering_response_range)),
        "steer_bias": float(np.random.normal(0.0, config.steer_bias_std)) if config.steer_bias_std > 0 else 0.0,
        "throttle_deadband": float(np.random.uniform(*config.throttle_deadband_range)),
        "actuator_lag_alpha": float(np.random.uniform(*config.actuator_lag_alpha_range)),
        "control_latency_steps": int(np.random.randint(0, config.max_control_latency_steps + 1)) if config.max_control_latency_steps > 0 else 0,
        "yaw_jitter_std": float(config.yaw_jitter_std),
    }


def perturb_walls(wall_segments: np.ndarray, config: DRConfig) -> np.ndarray:
    """Add small random perturbations to wall segment positions.

    Each coordinate gets independent Gaussian noise.  This simulates
    slightly wobbly physical track walls.
    """
    if config.wall_noise_std <= 0:
        return wall_segments
    perturbed = wall_segments.copy()
    noise = np.random.normal(0.0, config.wall_noise_std, perturbed.shape)
    perturbed += noise
    return perturbed


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DomainRandomization Self-Test ===")

    for phase in ("clean", "mild", "full"):
        cfg = get_dr_config(phase)
        print(f"\n  Phase '{phase}':")
        print(f"    lidar_noise_std      = {cfg.lidar_noise_std}")
        print(f"    lidar_dropout_rate   = {cfg.lidar_dropout_rate}")
        print(f"    lidar_salt_pepper    = {cfg.lidar_salt_pepper_rate}")
        print(f"    friction_range       = {cfg.friction_range}")
        print(f"    throttle_range       = {cfg.throttle_range}")
        print(f"    wall_noise_std       = {cfg.wall_noise_std}")
        print(f"    max_latency_steps    = {cfg.max_latency_steps}")

    # Test noise application
    fake_scan = np.full(20, 3.0)
    cfg_full = get_dr_config("full")
    noisy = apply_lidar_noise(fake_scan, cfg_full)
    print(f"\n  Original scan (first 5): {fake_scan[:5]}")
    print(f"  Noisy scan    (first 5): {noisy[:5]}")
    assert noisy.shape == fake_scan.shape
    assert np.all(noisy >= LIDAR_MIN_RANGE)
    assert np.all(noisy <= LIDAR_MAX_RANGE)

    # Test per-episode samplers
    fric = sample_friction_multiplier(cfg_full)
    thrt = sample_throttle_multiplier(cfg_full)
    lat = sample_latency(cfg_full)
    print(f"\n  Sampled friction mult : {fric:.3f}")
    print(f"  Sampled throttle mult: {thrt:.3f}")
    print(f"  Sampled latency      : {lat} steps")

    # Test wall perturbation
    dummy_walls = np.array([[0, 0, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
    perturbed = perturb_walls(dummy_walls, cfg_full)
    print(f"\n  Original walls:\n    {dummy_walls}")
    print(f"  Perturbed walls:\n    {perturbed}")
    assert perturbed.shape == dummy_walls.shape

    print("\n=== All DomainRandomization tests passed! ===")
