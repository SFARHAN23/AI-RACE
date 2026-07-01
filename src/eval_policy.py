"""Quick policy evaluation to debug LiDAR TD3 learning.

Examples:
    py.exe -3 eval_policy.py
    py.exe -3 eval_policy.py --model-prefix models/fixed_track_lidar_td3
    py.exe -3 eval_policy.py --model models/fixed_track_lidar_td3_final
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from lidar_env import LidarRacingEnv, STATE_DIM, ACTION_DIM
from td3_lidar import TD3


def find_latest_model(prefix: str) -> str | None:
    """Return latest checkpoint base path without _actor suffix."""
    candidates = sorted(glob.glob(prefix + "_*_actor"))
    final = prefix + "_final_actor"
    if os.path.exists(final):
        candidates.append(final)
    if not candidates:
        return None
    return candidates[-1].replace("_actor", "")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TD3 LiDAR policy.")
    parser.add_argument("--model-prefix", default="models/lidar_td3_model",
                        help="Checkpoint prefix used during training.")
    parser.add_argument("--model", default=None,
                        help="Exact checkpoint base path without _actor suffix.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of deterministic eval episodes.")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Maximum steps per episode.")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print telemetry every N steps.")
    args = parser.parse_args()

    model_name = args.model or find_latest_model(args.model_prefix)
    if not model_name:
        print(f"No models found for prefix: {args.model_prefix}")
        raise SystemExit(1)

    print(f"Loading: {model_name}")
    policy = TD3(STATE_DIM, ACTION_DIM, 1.0)
    policy.load_cpu(model_name)

    summaries = []
    for ep in range(args.episodes):
        env = LidarRacingEnv()
        obs = env.test_reset()
        total_r = 0.0
        done = False
        info = {"progress": 0.0, "reason": ""}

        for i in range(args.max_steps):
            action = policy.select_action(obs)
            obs, r, done, info = env.step(action)
            total_r += r
            if i % args.log_every == 0:
                print(
                    f"ep {ep:02d} step {i:4d}: "
                    f"spd={info['speed']:.2f} "
                    f"trip={info['trip']:.2f} "
                    f"progress={info['progress']:.2f} "
                    f"clDist={info['centerline_dist']:.3f} "
                    f"r={r:.3f} "
                    f"action=[{action[0]:.2f},{action[1]:.2f}]"
                )
            if done:
                break

        reason = info["reason"] if done else "SURVIVED_MAX_STEPS"
        summary = {
            "episode": ep,
            "reason": reason,
            "steps": i + 1,
            "total_reward": total_r,
            "progress": info["progress"],
            "track_length": env.track.total_trip,
            "progress_pct": 100.0 * info["progress"] / env.track.total_trip,
        }
        summaries.append(summary)
        print(
            f"DONE ep {ep}: {reason}, steps={summary['steps']}, "
            f"total_r={total_r:.1f}, "
            f"progress={info['progress']:.2f}/{env.track.total_trip:.2f} "
            f"({summary['progress_pct']:.1f}%)"
        )

    if len(summaries) > 1:
        print("\n=== Aggregate ===")
        for key in ("progress_pct", "total_reward", "steps"):
            vals = [s[key] for s in summaries]
            print(f"{key}: mean={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")


if __name__ == "__main__":
    main()
