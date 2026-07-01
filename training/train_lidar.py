"""
train_lidar.py — TD3 training with LiDAR observations.

3-phase curriculum:
  Phase 1 (0 - 100K steps):   Clean training, no domain randomization
  Phase 2 (100K - 300K steps): Mild noise (LiDAR noise + friction variation)
  Phase 3 (300K+ steps):       Full randomization (all DR enabled)
"""
import os
import time
import argparse
import numpy as np
from collections import deque
from td3_lidar import TD3
from utils import ReplayBuffer
from lidar_env import LidarRacingEnv, ACTION_DIM
from domain_randomization import get_dr_config
from safety_wrapper import SafetyConfig
from rc_car_model import MaxSteer
from track_library import list_track_names

# ── Shared constants (must match other files exactly) ──────────────────────
DT = 0.02

# ── Training configuration ─────────────────────────────────────────────────
MAX_EPISODES = 50_000
MAX_STEPS_PER_EPISODE = 5_000  # ~100 seconds at 50 Hz
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
SAVE_EVERY = 25_000   # Save every 25K training steps

# Phase boundaries (in total env steps)
PHASE_1_END = 100_000  # Clean
PHASE_2_END = 300_000  # Mild DR

# Exploration noise
NOISE_INITIAL = 0.25   # High noise for exploration
NOISE_REDUCED = 0.10
NOISE_DROP_THRESHOLD = 0.6  # Completion rate to reduce noise

# Warmup (fill buffer before training)
WARMUP_STEPS = 5_000


def get_phase(total_steps, dr_profile='auto'):
    """Return the domain-randomization phase for the given step count.

    dr_profile='auto' keeps the original clean→mild→full curriculum.
    dr_profile='auto_hardware' ends in the harsher hardware_rough preset with
    IMU/actuator/LiDAR failures for stronger sim-to-real robustness.
    A fixed profile ('clean', 'mild', 'full', 'hardware_rough') disables phase
    transitions and trains entirely under that realism level.
    """
    if dr_profile in ('clean', 'mild', 'full', 'hardware_rough'):
        return dr_profile
    if total_steps < PHASE_1_END:
        return 'clean'
    elif total_steps < PHASE_2_END:
        return 'mild'
    return 'hardware_rough' if dr_profile == 'auto_hardware' else 'full'


def process_state(s):
    """Reshape a flat state vector to a (1, STATE_DIM) array for the policy."""
    return np.reshape(s, [1, -1])


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def pure_pursuit_demo_action(env, target_speed_mps: float = 0.55, lookahead_m: float = 0.75):
    """Deterministic non-RL guide action used to seed replay faster than random warmup."""
    car = env.car
    track = env.track
    target_trip = (track.car_trip + lookahead_m) % track.total_trip
    idx = int(np.argmin(np.abs(track.centerline_cumdist - target_trip)))
    target = track.centerline_xy[idx]
    target_heading = np.arctan2(target[1] - car.pose[1], target[0] - car.pose[0])
    heading_error = _angle_diff(target_heading, car.pose[2])
    curvature = 2.0 * np.sin(heading_error) / max(lookahead_m, 1e-6)
    steer_rad = np.arctan(0.3302 * curvature)
    steer = float(np.clip(steer_rad / MaxSteer, -1.0, 1.0))
    throttle = 0.55 * (target_speed_mps - car.spd)
    if abs(heading_error) > 0.55:
        throttle -= 0.15
    return np.array([np.clip(throttle, -0.35, 0.55), steer], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='Train TD3 on the LiDAR RC simulator.')
    parser.add_argument('--max-total-steps', type=int, default=None,
                        help='Stop after this many environment steps (useful for quick validations).')
    parser.add_argument('--max-episodes', type=int, default=MAX_EPISODES,
                        help='Maximum number of episodes to run.')
    parser.add_argument('--model-prefix', default='models/lidar_td3_model',
                        help='Checkpoint prefix; _N_actor/_critic suffixes are added.')
    parser.add_argument('--load-model-prefix', default=None,
                        help='Optional checkpoint prefix to warm-start from, e.g. models/f1_round_robin_final.')
    parser.add_argument('--max-wall-time-min', type=float, default=None,
                        help='Stop cleanly after this many wall-clock minutes and save a final checkpoint.')
    parser.add_argument('--max-steps-per-episode', type=int, default=MAX_STEPS_PER_EPISODE,
                        help='Per-episode step cap. Use >=12000 for full 2-lap long F1 evaluations/training.')
    parser.add_argument('--save-every-train-steps', type=int, default=SAVE_EVERY,
                        help='Save checkpoint every N TD3 training updates.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional NumPy random seed for repeatable partial runs.')
    parser.add_argument('--track-name', default=None,
                        help='Named track from track_library.py, e.g. f1_bahrain_sakhir.')
    parser.add_argument('--track-list', default=None,
                        help='Comma-separated named tracks. One is sampled each episode for multi-track training.')
    parser.add_argument('--track-list-file', default=None,
                        help='Text file with one track name per line. Blank lines and # comments ignored.')
    parser.add_argument('--safe-training', action='store_true',
                        help='Filter actions through the real-car safety wrapper during training and penalize interventions.')
    parser.add_argument('--safety-max-speed', type=float, default=1.0,
                        help='Safe-training speed cap in m/s.')
    parser.add_argument('--safety-stop-distance', type=float, default=0.55,
                        help='Front LiDAR emergency stop distance in m.')
    parser.add_argument('--safety-caution-distance', type=float, default=1.10,
                        help='Front LiDAR caution distance in m.')
    parser.add_argument('--safe-start-max-speed', type=float, default=1.0,
                        help='Maximum randomized reset speed when --safe-training is enabled.')
    parser.add_argument('--reward-profile', default='slow_completion', choices=['slow_completion', 'lap_improvement'],
                        help='Reward profile: first finish slow laps, then improve lap time.')
    parser.add_argument('--sector-count', type=int, default=12,
                        help='Virtual sectors for sim-only progress diagnostics/reward shaping.')
    parser.add_argument('--demo-action-prob', type=float, default=0.20,
                        help='Probability of using non-RL pure-pursuit guide action after warmup.')
    parser.add_argument('--demo-target-speed', type=float, default=0.55,
                        help='Target speed for pure-pursuit guide actions.')
    parser.add_argument('--demo-lookahead', type=float, default=0.75,
                        help='Lookahead distance for pure-pursuit guide actions.')
    parser.add_argument('--recovery-demo-steps', type=int, default=0,
                        help='For first N steps of an episode, force pure-pursuit guide if progress is below --recovery-demo-progress-m. Fixes brake-to-stop local minima on long F1 tracks.')
    parser.add_argument('--recovery-demo-progress-m', type=float, default=0.35,
                        help='Progress threshold used with --recovery-demo-steps.')
    parser.add_argument('--use-imu-memory', action='store_true',
                        help='Use V3 extended observation: LiDAR + simulated IMU + sector track memory.')
    parser.add_argument('--imu-safety', action='store_true',
                        help='Enable IMU-derived spin/slip/impact rules inside the safety wrapper.')
    parser.add_argument('--dr-profile', default='auto_hardware',
                        choices=['auto', 'auto_hardware', 'clean', 'mild', 'full', 'hardware_rough'],
                        help='Domain randomization schedule/preset. auto_hardware ends with realistic LiDAR/IMU/actuator failures.')
    parser.add_argument('--target-laps', type=int, default=2,
                        help='Finish target for each episode; 2 exposes lap-2 drift instead of stopping at lap 1.')
    parser.add_argument('--routelet-length-m', type=float, default=None,
                        help='If set, finish an episode after this many metres of forward progress from the reset point.')
    parser.add_argument('--adaptive-spawn', action=argparse.BooleanOptionalAction, default=True,
                        help='Sample off-center/curve starts instead of assuming centerline start.')
    parser.add_argument('--spawn-lateral-frac', type=float, default=0.42,
                        help='Max random lateral spawn offset as a fraction of track width, clipped by body clearance.')
    parser.add_argument('--spawn-heading-jitter', type=float, default=0.12,
                        help='Random heading perturbation at reset, radians.')
    parser.add_argument('--spawn-allow-curves', action=argparse.BooleanOptionalAction, default=True,
                        help='Allow some curve/transition starts for adaptive recovery training.')
    parser.add_argument('--push-lap-start', type=int, default=1,
                        help='Lap index where the target speed switches from outlap pace to push-lap pace; 1 = one outlap, 2 = two outlaps.')
    parser.add_argument('--push-target-speed', type=float, default=None,
                        help='Optional faster target speed in m/s after --push-lap-start; keeps outlap conservative.')
    args = parser.parse_args()

    track_pool = []
    if args.track_list:
        track_pool.extend([x.strip() for x in args.track_list.split(',') if x.strip()])
    if args.track_list_file:
        with open(args.track_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.split('#', 1)[0].strip()
                if name:
                    track_pool.append(name)
    if not track_pool and args.track_name:
        track_pool = [args.track_name]
    if not track_pool:
        track_pool = [None]

    available_tracks = set(list_track_names())
    unknown = [t for t in track_pool if t is not None and t not in available_tracks]
    if unknown:
        raise ValueError(f'Unknown tracks in --track-list: {unknown}. Available: {sorted(available_tracks)}')

    model_parent = os.path.dirname(args.model_prefix)
    if model_parent:
        os.makedirs(model_parent, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
    start_wall_time = time.time()

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Start with the selected DR curriculum/preset
    current_phase = get_phase(0, args.dr_profile)
    dr_config = get_dr_config(current_phase)
    safety_config = SafetyConfig(
        max_speed_mps=args.safety_max_speed,
        stop_distance_m=args.safety_stop_distance,
        caution_distance_m=args.safety_caution_distance,
        use_imu_safety=args.imu_safety,
    )

    def make_env(config, track_name=None):
        return LidarRacingEnv(
            dr_config=config,
            track_name=track_name,
            safe_training=args.safe_training,
            safety_config=safety_config,
            random_start_speed_max=(args.safe_start_max_speed if args.safe_training else 2.0),
            reward_profile=args.reward_profile,
            sector_count=args.sector_count,
            use_imu_memory=args.use_imu_memory,
            target_laps=args.target_laps,
            routelet_length_m=args.routelet_length_m,
            adaptive_spawn=args.adaptive_spawn,
            spawn_lateral_frac=args.spawn_lateral_frac,
            spawn_heading_jitter_rad=args.spawn_heading_jitter,
            spawn_allow_curves=args.spawn_allow_curves,
            push_lap_start=args.push_lap_start,
            push_target_speed_mps=args.push_target_speed,
            max_steps=args.max_steps_per_episode,
        )

    active_track_name = track_pool[0]
    env = make_env(dr_config, active_track_name)
    state_dim = int(env.observation_dim)

    # ── Initialise policy & replay buffer ──────────────────────────────────
    policy = TD3(state_dim=state_dim, action_dim=ACTION_DIM, max_action=1.0)
    if args.load_model_prefix:
        loaded_critic = policy.load(args.load_model_prefix)
        print(f'Warm-started policy from {args.load_model_prefix} (critic_loaded={loaded_critic})')
    replay_buffer = ReplayBuffer(state_dim, ACTION_DIM, max_size=BUFFER_SIZE)

    # ── Tracking variables ─────────────────────────────────────────────────
    total_steps = 0
    train_steps = 0
    save_counter = 1
    completions = deque(maxlen=100)
    best_lap_time = float('inf')
    trainlog = []

    print('=' * 70)
    print('  SimToReal TD3 LiDAR Training')
    print(f'  State dim: {state_dim}  Action dim: {ACTION_DIM}')
    print(f'  Track pool: {", ".join([t or "default" for t in track_pool])}')
    print(f'  Active track: {active_track_name or "default"}')
    print(f'  Track length: {env.track.total_trip:.2f} m')
    print(f'  Reward profile: {args.reward_profile}  sectors: {args.sector_count}')
    print(f'  Target laps: {args.target_laps}  routelet: {args.routelet_length_m or "off"} m  adaptive spawn: {args.adaptive_spawn} lateral_frac={args.spawn_lateral_frac:.2f}')
    print(f'  Outlap/push: push_lap_start={args.push_lap_start}  push_target_speed={args.push_target_speed or "profile"} m/s')
    print(f'  DR profile: {args.dr_profile}  initial phase: {current_phase}')
    print(f'  Demo guide: pure_pursuit prob={args.demo_action_prob:.2f} speed={args.demo_target_speed:.2f} lookahead={args.demo_lookahead:.2f}')
    print(f'  Safe training: {args.safe_training}')
    if args.safe_training:
        print(f'  Safety cap: {args.safety_max_speed:.2f} m/s  caution: {args.safety_caution_distance:.2f} m  stop: {args.safety_stop_distance:.2f} m')
    print(f'  Buffer size: {BUFFER_SIZE:,}  Batch: {BATCH_SIZE}')
    print('=' * 70)

    for episode in range(args.max_episodes):
        # ── Phase transition check ─────────────────────────────────────────
        new_phase = get_phase(total_steps, args.dr_profile)
        if new_phase != current_phase:
            current_phase = new_phase
            dr_config = get_dr_config(current_phase)
            print(f'\n>>> Phase transition to: {current_phase.upper()} '
                  f'at step {total_steps:,}')

        # Multi-track curriculum: sample a track per episode. Recreating the env
        # keeps wall geometry/sector length correct while sharing one policy and
        # replay buffer because observation/action dimensions are fixed.
        active_track_name = track_pool[int(np.random.randint(len(track_pool)))]
        env = make_env(dr_config, active_track_name)

        # ── Reset environment ──────────────────────────────────────────────
        if args.adaptive_spawn and np.random.rand() < 0.85:
            ob = env.reset()
        else:
            # Deterministic reset is still off-center sometimes so the policy
            # sees start-line offset recovery without relying on a perfect center.
            off = float(np.random.uniform(-0.28, 0.28)) if args.adaptive_spawn else 0.0
            ob = env.test_reset(lateral_offset_m=off)
        ob = process_state(ob)

        episode_reward = 0.0
        episode_best_sector = 0
        episode_sector_advances = 0
        episode_laps = 0
        episode_center_abs_sum = 0.0
        episode_smooth_delta_sum = 0.0

        # Noise schedule
        comp_rate = np.mean(completions) if completions else 0.0
        noise_scale = (NOISE_REDUCED
                       if comp_rate > NOISE_DROP_THRESHOLD
                       else NOISE_INITIAL)

        # ── Rollout ────────────────────────────────────────────────────────
        for step in range(args.max_steps_per_episode):
            total_steps += 1

            # Select action
            force_recovery_demo = (
                args.recovery_demo_steps > 0
                and step < args.recovery_demo_steps
                and getattr(env, '_total_progress', 0.0) < args.recovery_demo_progress_m
            )
            if force_recovery_demo:
                action = pure_pursuit_demo_action(env, args.demo_target_speed, args.demo_lookahead)
                action += np.random.normal(0, noise_scale * 0.05, size=ACTION_DIM)
                action = action.clip(-1, 1)
            elif total_steps < WARMUP_STEPS:
                if np.random.rand() < 0.90:
                    action = pure_pursuit_demo_action(env, args.demo_target_speed, args.demo_lookahead)
                    action += np.random.normal(0, noise_scale * 0.20, size=ACTION_DIM)
                    action = action.clip(-1, 1)
                else:
                    action = np.random.uniform(-0.2, 0.8, size=ACTION_DIM)
            elif args.demo_action_prob > 0 and np.random.rand() < args.demo_action_prob:
                action = pure_pursuit_demo_action(env, args.demo_target_speed, args.demo_lookahead)
                action += np.random.normal(0, noise_scale * 0.15, size=ACTION_DIM)
                action = action.clip(-1, 1)
            else:
                noise = np.random.normal(0, noise_scale, size=ACTION_DIM)
                action = (policy.select_action(ob) + noise).clip(-1, 1)

            # Step environment
            next_ob, reward, done, info = env.step(action)
            episode_best_sector = max(episode_best_sector, int(info.get('best_sector', 0)))
            episode_sector_advances = max(episode_sector_advances, int(info.get('sector_advances', 0)))
            episode_laps = max(episode_laps, int(info.get('lap_count', 0)))
            episode_center_abs_sum += abs(float(info.get('centerline_dist', 0.0)))
            episode_smooth_delta_sum += float(np.mean(np.abs(info.get('applied_action_delta', np.zeros(2)))))
            next_ob = process_state(next_ob)

            # Store transition
            replay_buffer.add(ob, action, next_ob, reward, float(done))
            ob = next_ob
            episode_reward += reward

            if done:
                break

            # Train TD3
            if (total_steps >= WARMUP_STEPS
                    and replay_buffer.size >= BATCH_SIZE):
                policy.train(replay_buffer, BATCH_SIZE)
                train_steps += 1

            # Save checkpoint
            if (train_steps > 0
                    and args.save_every_train_steps > 0
                    and train_steps % args.save_every_train_steps == 0):
                name = f'{args.model_prefix}_{save_counter}'
                policy.save(name)
                save_counter += 1
                print(f'  >> Saved: {name}', flush=True)

            if args.max_total_steps is not None and total_steps >= args.max_total_steps:
                break

            if (args.max_wall_time_min is not None
                    and (time.time() - start_wall_time) >= args.max_wall_time_min * 60.0):
                break

        # ── Episode bookkeeping ────────────────────────────────────────────
        reason = env.query_fail_reason()
        completed = (reason == 'FINISHED' and (args.routelet_length_m is not None or episode_laps >= args.target_laps))
        completions.append(1 if completed else 0)

        lap_time = step * DT if completed else None
        if lap_time is not None and lap_time < best_lap_time:
            best_lap_time = lap_time

        comp_pct = np.mean(completions) * 100 if completions else 0
        lap_str = f'{lap_time:.2f}s' if lap_time else 'DNF'
        best_str = (f'{best_lap_time:.2f}s'
                    if best_lap_time < float('inf') else '--')

        print(
            f'ep:{episode:<5d} '
            f'rew:{episode_reward:>8.1f}  '
            f'steps:{step:<5d}  '
            f'total:{total_steps:>8,}  '
            f'train:{train_steps:>7,}  '
            f'track:{(active_track_name or "default")[:18]:<18s}  '
            f'[{current_phase.upper():>5s}]  '
            f'noise:{noise_scale:.3f}  '
            f'comp:{comp_pct:>4.0f}%  '
            f'sector:{episode_sector_advances:>2d}/{args.sector_count:<2d}  '
            f'laps:{episode_laps}/{args.target_laps}  '
            f'smoothD:{(episode_smooth_delta_sum/max(step+1,1)):.3f}  '
            f'cdist:{(episode_center_abs_sum/max(step+1,1)):.2f}  '
            f'lap:{lap_str}  '
            f'best:{best_str}  '
            f'{reason}'
        )

        trainlog.append([episode, episode_reward, step, train_steps,
                         comp_pct, lap_time if lap_time else -1,
                         episode_sector_advances, args.sector_count, episode_laps,
                         episode_smooth_delta_sum / max(step + 1, 1),
                         episode_center_abs_sum / max(step + 1, 1)])

        if episode % 25 == 0:
            np.save('results/trainlog_lidar.npy', np.array(trainlog))

        if args.max_total_steps is not None and total_steps >= args.max_total_steps:
            print(f'Reached --max-total-steps={args.max_total_steps}; stopping early.')
            break

        if (args.max_wall_time_min is not None
                and (time.time() - start_wall_time) >= args.max_wall_time_min * 60.0):
            print(f'Reached --max-wall-time-min={args.max_wall_time_min}; stopping early.')
            break

    # ── Final save ─────────────────────────────────────────────────────────
    policy.save(args.model_prefix + '_final')
    np.save('results/trainlog_lidar.npy', np.array(trainlog))
    print('Training complete!')


if __name__ == '__main__':
    main()
