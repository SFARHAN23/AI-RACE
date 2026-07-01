"""
model_rollout_visualizer.py — run a trained TD3 LiDAR model in simulation and
render what the RC car/model sees.

This is the bridge between training and real-car deployment:
- loads model weights (_actor checkpoint)
- runs the model in the same LiDAR environment used for training
- records the exact observation/action telemetry a real-car controller would use
- draws cardboard boundaries, car body, trajectory, LiDAR field/rays, action HUD
- writes CSV + JSON + PNG + optional GIF/MP4

Example:
    py.exe -3 model_rollout_visualizer.py --model models/fixed_track_lidar_td3_2 --prefix results/model_view

Outputs:
    results/model_view.csv
    results/model_view_summary.json
    results/model_view_final.png
    results/model_view.gif      (default, if Pillow writer is available)
    results/model_view.mp4      (if --format mp4 and ffmpeg is available)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import lidar_env as lidar_env_module
from lidar_env import LidarRacingEnv, STATE_DIM, ACTION_DIM, N_BEAMS, LIDAR_ANGLE_MIN, LIDAR_ANGLE_MAX
from rc_car_model import CarLength, CarWidth, SPD_MAX
from domain_randomization import LIDAR_MAX_RANGE
from td3_lidar import TD3
from safety_wrapper import SafetyActionWrapper, SafetyConfig


@dataclass
class RolloutConfig:
    model: str
    max_steps: int = 2000
    fps: int = 25
    frame_stride: int = 2
    output_format: str = "gif"  # gif, mp4, none
    prefix: str = "results/model_rollout"
    render_lidar_every_beam: bool = True
    safety_enabled: bool = True
    safety_max_speed_mps: float = 1.0
    safety_stop_distance_m: float = 0.55
    safety_caution_distance_m: float = 1.10
    track_name: str = "default"
    start_speed_mps: float = 1.0
    use_imu_memory: bool = False
    imu_safety: bool = False
    target_laps: int = 1


def load_policy(model_path: str, state_dim: int = STATE_DIM) -> TD3:
    policy = TD3(state_dim, ACTION_DIM, 1.0)
    policy.load_cpu(model_path)
    return policy


def collect_rollout(policy: TD3, max_steps: int, safety: Optional[SafetyActionWrapper] = None, track_name: str = "default", start_speed_mps: float = 1.0, use_imu_memory: bool = False, target_laps: int = 1):
    # Match the environment time-limit to the requested rollout length.
    # The old module default was 2000 sim steps, so longer --steps values still
    # ended early at about 40s / ~501 rendered frames with stride=4.
    lidar_env_module.MAX_STEPS = int(max_steps)
    env = LidarRacingEnv(track_name=track_name, use_imu_memory=use_imu_memory, target_laps=target_laps)
    obs = env.test_reset(spd=start_speed_mps)
    rows = []
    frames = []
    total_reward = 0.0
    done = False
    reason = "MAX_STEPS"

    beam_angles = np.linspace(LIDAR_ANGLE_MIN, LIDAR_ANGLE_MAX, N_BEAMS)
    if safety is not None:
        safety.reset()

    for step in range(max_steps):
        pose_before = env.car.pose.copy()
        scan_norm = obs[3:3 + N_BEAMS].copy()
        scan_m = scan_norm * LIDAR_MAX_RANGE
        speed_norm = float(obs[0])
        yaw_rate_norm = float(obs[1])
        steer_norm = float(obs[2])

        raw_action = policy.select_action(obs)
        if safety is not None:
            action, safety_meta = safety.filter_action(raw_action, obs, beam_angles, imu_meta=getattr(env, "_last_imu_meta", None))
        else:
            action = np.clip(raw_action, -1.0, 1.0)
            safety_meta = {
                "raw_throttle_cmd": float(raw_action[0]),
                "raw_steer_cmd": float(raw_action[1]),
                "safe_throttle_cmd": float(action[0]),
                "safe_steer_cmd": float(action[1]),
                "front_min_lidar_m": float(np.min(scan_m)),
                "interventions": [],
                "intervention_count": 0,
            }
        next_obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        row = {
            "step": step,
            "time_s": step * 0.02,
            "x_m": float(pose_before[0]),
            "y_m": float(pose_before[1]),
            "heading_rad": float(pose_before[2]),
            "speed_mps": float(speed_norm * SPD_MAX),
            "yaw_rate_norm": yaw_rate_norm,
            "steer_norm": steer_norm,
            "raw_throttle_cmd": float(raw_action[0]),
            "raw_steer_cmd": float(raw_action[1]),
            "throttle_cmd": float(action[0]),
            "steer_cmd": float(action[1]),
            "safe_throttle_cmd": float(action[0]),
            "safe_steer_cmd": float(action[1]),
            "front_min_lidar_m": float(safety_meta.get("front_min_lidar_m", np.min(scan_m))),
            "safety_intervention_count": int(safety_meta.get("intervention_count", 0)),
            "safety_interventions": ";".join(safety_meta.get("interventions", [])),
            "reward": float(reward),
            "total_reward": float(total_reward),
            "trip_m": float(info.get("trip", 0.0)),
            "progress_m": float(info.get("progress", 0.0)),
            "progress_pct": float(100.0 * info.get("progress", 0.0) / env.track.total_trip),
            "centerline_dist_m": float(info.get("centerline_dist", 0.0)),
            "done": bool(done),
            "reason": info.get("reason", "") if done else "",
        }
        for i, d in enumerate(scan_m):
            row[f"lidar_{i:02d}_m"] = float(d)
        rows.append(row)

        frames.append({
            "step": step,
            "pose": pose_before.copy(),
            "scan_m": scan_m.copy(),
            "beam_angles": beam_angles.copy(),
            "raw_action": raw_action.copy(),
            "action": action.copy(),
            "safety_meta": safety_meta.copy(),
            "reward": float(reward),
            "total_reward": float(total_reward),
            "speed_mps": float(speed_norm * SPD_MAX),
            "progress_pct": row["progress_pct"],
            "centerline_dist_m": row["centerline_dist_m"],
            "reason": row["reason"],
        })

        obs = next_obs
        if done:
            reason = info.get("reason", env.query_fail_reason())
            break

    summary = {
        "model": None,
        "reason": reason,
        "steps": len(rows),
        "sim_time_s": rows[-1]["time_s"] if rows else 0.0,
        "total_reward": total_reward,
        "progress_m": rows[-1]["progress_m"] if rows else 0.0,
        "progress_pct": rows[-1]["progress_pct"] if rows else 0.0,
        "track_length_m": float(env.track.total_trip),
        "target_laps": int(target_laps),
        "progress_laps": float((rows[-1]["progress_m"] if rows else 0.0) / max(env.track.total_trip, 1e-9)),
        "max_speed_mps": max((r["speed_mps"] for r in rows), default=0.0),
        "max_abs_centerline_dist_m": max((abs(r["centerline_dist_m"]) for r in rows), default=0.0),
        "safety_enabled": safety is not None and safety.config.enabled,
        "safety_config": safety.config_dict() if safety is not None else None,
        "safety_intervention_steps": sum(1 for r in rows if r.get("safety_intervention_count", 0) > 0),
        "total_safety_interventions": sum(int(r.get("safety_intervention_count", 0)) for r in rows),
        "max_raw_throttle_cmd": max((r["raw_throttle_cmd"] for r in rows), default=0.0),
        "max_safe_throttle_cmd": max((r["safe_throttle_cmd"] for r in rows), default=0.0),
        "min_front_lidar_m": min((r["front_min_lidar_m"] for r in rows), default=0.0),
        "n_lidar_beams": N_BEAMS,
        "lidar_angle_min_deg": float(np.degrees(LIDAR_ANGLE_MIN)),
        "lidar_angle_max_deg": float(np.degrees(LIDAR_ANGLE_MAX)),
        "lidar_max_range_m": LIDAR_MAX_RANGE,
    }
    return env, rows, frames, summary


def write_csv(rows, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def car_transform(x: float, y: float, heading: float):
    return transforms.Affine2D().rotate(heading).translate(x, y)


def draw_start_finish_line(ax, track):
    """Draw the actual track start/finish gate in the original final visual style."""
    left = track.left_wall_xy[0]
    right = track.right_wall_xy[0]
    mid = (left + right) / 2.0
    ax.plot([left[0], right[0]], [left[1], right[1]], color="#ffd33d", lw=4.0, alpha=0.95, solid_capstyle="round", zorder=6)
    ax.plot([left[0], right[0]], [left[1], right[1]], color="#0f1117", lw=1.2, alpha=0.95, solid_capstyle="round", zorder=7)
    ax.text(mid[0], mid[1], " START / FINISH", color="#ffd33d", fontsize=8, fontweight="bold", zorder=8)


def draw_final_png(env: LidarRacingEnv, frames, summary: dict, path: str):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax_lidar = fig.add_subplot(2, 2, 2)
    ax_action = fig.add_subplot(2, 2, 4)

    track = env.track
    ax.plot(track.left_wall_xy[:, 0], track.left_wall_xy[:, 1], "k-", lw=2, label="cardboard boundary")
    ax.plot(track.right_wall_xy[:, 0], track.right_wall_xy[:, 1], "k-", lw=2)
    ax.plot(track.centerline_xy[:, 0], track.centerline_xy[:, 1], "--", color="0.65", lw=1, label="centerline")
    draw_start_finish_line(ax, track)

    xs = [float(f["pose"][0]) for f in frames]
    ys = [float(f["pose"][1]) for f in frames]
    ax.plot(xs, ys, "b-", lw=2, label="model trajectory")
    ax.plot(xs[0], ys[0], "go", ms=8, label="start")
    ax.plot(xs[-1], ys[-1], "mo" if summary["reason"] == "FINISHED" else "ro", ms=8, label=summary["reason"])

    # Final car and final LiDAR rays.
    final = frames[-1]
    x, y, heading = final["pose"]
    car = patches.Rectangle((-CarLength / 2, -CarWidth / 2), CarLength, CarWidth,
                            facecolor="red", edgecolor="darkred", alpha=0.75)
    car.set_transform(car_transform(x, y, heading) + ax.transData)
    ax.add_patch(car)

    for angle, dist in zip(final["beam_angles"], final["scan_m"]):
        abs_angle = heading + angle
        ex = x + dist * np.cos(abs_angle)
        ey = y + dist * np.sin(abs_angle)
        color = "limegreen" if dist > 2.0 else "orange" if dist > 0.5 else "red"
        ax.plot([x, ex], [y, ey], color=color, alpha=0.45, lw=0.8)

    margin = 0.8
    ax.set_xlim(track.centerline_xy[:, 0].min() - margin, track.centerline_xy[:, 0].max() + margin)
    ax.set_ylim(track.centerline_xy[:, 1].min() - margin, track.centerline_xy[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"Model rollout: {summary['reason']} | progress {summary['progress_pct']:.1f}%")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    angles_deg = np.degrees(final["beam_angles"])
    colors = ["limegreen" if d > 2.0 else "orange" if d > 0.5 else "red" for d in final["scan_m"]]
    ax_lidar.bar(angles_deg, final["scan_m"], width=10, color=colors, alpha=0.8)
    ax_lidar.set_ylim(0, LIDAR_MAX_RANGE)
    ax_lidar.set_title("Final simulated RPLIDAR scan")
    ax_lidar.set_xlabel("beam angle relative to car [deg]")
    ax_lidar.set_ylabel("range [m]")
    ax_lidar.grid(True, alpha=0.25)

    steps = [f["step"] for f in frames]
    throttle = [f["action"][0] for f in frames]
    steer = [f["action"][1] for f in frames]
    speed = [f["speed_mps"] for f in frames]
    raw_throttle = [f.get("raw_action", f["action"])[0] for f in frames]
    raw_steer = [f.get("raw_action", f["action"])[1] for f in frames]
    ax_action.plot(steps, raw_throttle, label="raw throttle", color="tab:green", alpha=0.25, linestyle="--")
    ax_action.plot(steps, raw_steer, label="raw steer", color="tab:blue", alpha=0.25, linestyle="--")
    ax_action.plot(steps, throttle, label="SAFE throttle/brake", color="tab:green")
    ax_action.plot(steps, steer, label="SAFE steer", color="tab:blue")
    ax_action.plot(steps, np.array(speed) / max(SPD_MAX, 1e-9), label="speed/SPD_MAX", color="tab:red", alpha=0.7)
    ax_action.set_ylim(-1.05, 1.05)
    ax_action.set_title("Raw model output vs safety-filtered actions")
    ax_action.set_xlabel("step")
    ax_action.grid(True, alpha=0.25)
    ax_action.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_animation(env: LidarRacingEnv, frames, summary: dict, path: str, fps: int, stride: int, output_format: str):
    if output_format == "none":
        return None

    render_frames = frames[::max(1, stride)]
    if render_frames[-1] is not frames[-1]:
        render_frames.append(frames[-1])

    track = env.track
    # 16:9 professional layout: large track view on the left, telemetry panels
    # on the right. No big text box is drawn over the driving area.
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="#0f1117")
    # Manual axes gives the track the screen real-estate.  GridSpec left too much
    # dead space and made the track feel small.
    ax_title = fig.add_axes([0.035, 0.925, 0.93, 0.055])
    ax = fig.add_axes([0.035, 0.075, 0.705, 0.825])
    ax_lidar = fig.add_axes([0.775, 0.545, 0.195, 0.350])
    ax_hud = fig.add_axes([0.775, 0.075, 0.195, 0.390])
    for a in (ax_title, ax, ax_lidar, ax_hud):
        a.set_facecolor("#151922")
    ax_title.axis("off")

    ax.plot(track.left_wall_xy[:, 0], track.left_wall_xy[:, 1], color="#f4f4f5", lw=3.0)
    ax.plot(track.right_wall_xy[:, 0], track.right_wall_xy[:, 1], color="#f4f4f5", lw=3.0)
    ax.plot(track.centerline_xy[:, 0], track.centerline_xy[:, 1], "--", color="#8b949e", lw=1.4, alpha=0.85)
    draw_start_finish_line(ax, track)
    xs_all = np.concatenate([track.left_wall_xy[:, 0], track.right_wall_xy[:, 0], track.centerline_xy[:, 0]])
    ys_all = np.concatenate([track.left_wall_xy[:, 1], track.right_wall_xy[:, 1], track.centerline_xy[:, 1]])
    margin = 0.18
    ax.set_xlim(xs_all.min() - margin, xs_all.max() + margin)
    ax.set_ylim(ys_all.min() - margin, ys_all.max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, color="#8b949e")
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    ax.set_xlabel("x [m]", color="#c9d1d9")
    ax.set_ylabel("y [m]", color="#c9d1d9")

    car = patches.Rectangle((-CarLength / 2, -CarWidth / 2), CarLength, CarWidth,
                            facecolor="#ff4d4d", edgecolor="#ffd1d1", linewidth=1.5, alpha=0.92)
    ax.add_patch(car)
    traj_line, = ax.plot([], [], color="#58a6ff", lw=2.6, alpha=0.95)
    beam_lines = [ax.plot([], [], lw=0.9, alpha=0.34)[0] for _ in range(N_BEAMS)]
    status_text = ax.text(0.012, 0.018, "", transform=ax.transAxes, va="bottom", ha="left",
                          color="#c9d1d9", fontsize=9,
                          bbox=dict(facecolor="#0f1117", alpha=0.62, edgecolor="#30363d", pad=4))

    def style_panel(panel, title):
        panel.set_title(title, color="#f0f6fc", fontsize=11, pad=8)
        panel.tick_params(colors="#c9d1d9", labelsize=8)
        panel.grid(True, alpha=0.16, color="#8b949e")
        for spine in panel.spines.values():
            spine.set_color("#30363d")

    def update(i):
        f = render_frames[i]
        x, y, heading = f["pose"]
        car.set_transform(car_transform(x, y, heading) + ax.transData)

        all_until = render_frames[:i + 1]
        traj_line.set_data([q["pose"][0] for q in all_until], [q["pose"][1] for q in all_until])

        for line, angle, dist in zip(beam_lines, f["beam_angles"], f["scan_m"]):
            abs_angle = heading + angle
            ex = x + dist * np.cos(abs_angle)
            ey = y + dist * np.sin(abs_angle)
            line.set_data([x, ex], [y, ey])
            line.set_color("#3fb950" if dist > 2.0 else "#d29922" if dist > 0.5 else "#f85149")

        raw = f.get("raw_action", f["action"])
        safety = ','.join(f.get('safety_meta', {}).get('interventions', [])) or 'none'
        ax_title.text(
            0.01, 0.50,
            f"AIRACE TD3 RAW SIM ROLLOUT  |  track: {summary.get('track_name','')}  |  frame {i+1}/{len(render_frames)}  |  sim step {f['step']}  |  progress {f['progress_pct']:.1f}%  |  result {summary.get('reason','')}",
            transform=ax_title.transAxes, va="center", ha="left", color="#f0f6fc", fontsize=16, fontweight="bold",
        )
        # Clear previous title text each frame without clearing the axes object.
        if len(ax_title.texts) > 1:
            ax_title.texts[0].remove()
        status_text.set_text(
            f"speed {f['speed_mps']:.2f} m/s   raw T/S {raw[0]:+.2f}/{raw[1]:+.2f}   "
            f"applied T/S {f['action'][0]:+.2f}/{f['action'][1]:+.2f}   front {f.get('safety_meta', {}).get('front_min_lidar_m', 0.0):.2f} m   safety {safety}"
        )

        ax_lidar.cla()
        ax_lidar.set_facecolor("#151922")
        angles_deg = np.degrees(f["beam_angles"])
        colors = ["#3fb950" if d > 2.0 else "#d29922" if d > 0.5 else "#f85149" for d in f["scan_m"]]
        ax_lidar.bar(angles_deg, f["scan_m"], width=9, color=colors, alpha=0.88)
        ax_lidar.set_ylim(0, LIDAR_MAX_RANGE)
        ax_lidar.set_xlabel("angle [deg]", color="#c9d1d9")
        ax_lidar.set_ylabel("range [m]", color="#c9d1d9")
        style_panel(ax_lidar, "MODEL INPUT: LiDAR")

        ax_hud.cla()
        ax_hud.set_facecolor("#151922")
        upto = render_frames[:i + 1]
        steps = [q["step"] for q in upto]
        raw_throttle = [q.get("raw_action", q["action"])[0] for q in upto]
        raw_steer = [q.get("raw_action", q["action"])[1] for q in upto]
        throttle = [q["action"][0] for q in upto]
        steer = [q["action"][1] for q in upto]
        progress = [q["progress_pct"] / 100.0 for q in upto]
        ax_hud.plot(steps, raw_throttle, color="#3fb950", alpha=0.42, linestyle="--", label="raw throttle")
        ax_hud.plot(steps, raw_steer, color="#58a6ff", alpha=0.42, linestyle="--", label="raw steer")
        ax_hud.plot(steps, throttle, color="#3fb950", lw=1.7, label="applied throttle")
        ax_hud.plot(steps, steer, color="#58a6ff", lw=1.7, label="applied steer")
        ax_hud.plot(steps, progress, color="#d2a8ff", lw=1.9, label="progress")
        ax_hud.set_ylim(-1.05, 1.05)
        ax_hud.set_xlabel("sim step", color="#c9d1d9")
        style_panel(ax_hud, "TD3 OUTPUT + PROGRESS")
        leg = ax_hud.legend(fontsize=7, loc="lower right", framealpha=0.65)
        for txt in leg.get_texts():
            txt.set_color("#f0f6fc")
        return [car, traj_line, status_text] + beam_lines

    ani = FuncAnimation(fig, update, frames=len(render_frames), interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if output_format == "gif":
        ani.save(path, writer=PillowWriter(fps=fps))
    elif output_format == "mp4":
        ani.save(path, writer=FFMpegWriter(fps=fps, bitrate=4200, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize trained TD3 policy, LiDAR rays, and actions.")
    parser.add_argument("--model", default="models/fixed_track_lidar_td3_2",
                        help="Checkpoint base path without _actor suffix")
    parser.add_argument("--steps", type=int, default=2000, help="Max rollout steps")
    parser.add_argument("--prefix", default="results/model_rollout", help="Output prefix")
    parser.add_argument("--format", choices=["gif", "mp4", "none"], default="gif",
                        help="Animation output format")
    parser.add_argument("--fps", type=int, default=25, help="Animation FPS")
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Render every Nth simulation step to keep file size reasonable")
    parser.add_argument("--safe", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable safety wrapper around model actions")
    parser.add_argument("--safety-max-speed", type=float, default=1.0,
                        help="Safety speed cap in m/s for first real-car style tests")
    parser.add_argument("--safety-stop-distance", type=float, default=0.55,
                        help="Emergency brake if forward LiDAR clearance is below this")
    parser.add_argument("--safety-caution-distance", type=float, default=1.10,
                        help="Limit throttle/steer if forward LiDAR clearance is below this")
    parser.add_argument("--track-name", default="default",
                        help="Named track from track_library.py, e.g. f1_bahrain_sakhir")
    parser.add_argument("--start-speed", type=float, default=1.0,
                        help="Fixed rollout reset speed in m/s; use <= safety cap for slow-safe videos")
    parser.add_argument("--use-imu-memory", action="store_true",
                        help="Load/run a V3 policy with extended LiDAR+IMU+track-memory observations")
    parser.add_argument("--imu-safety", action="store_true",
                        help="Enable IMU spin/slip/impact safety rules during visualization")
    parser.add_argument("--target-laps", type=int, default=1,
                        help="Number of laps before FINISHED; use 2 for two-lap videos")
    args = parser.parse_args()

    cfg = RolloutConfig(
        model=args.model,
        max_steps=args.steps,
        fps=args.fps,
        frame_stride=args.frame_stride,
        output_format=args.format,
        prefix=args.prefix,
        safety_enabled=args.safe,
        safety_max_speed_mps=args.safety_max_speed,
        safety_stop_distance_m=args.safety_stop_distance,
        safety_caution_distance_m=args.safety_caution_distance,
        track_name=args.track_name,
        start_speed_mps=args.start_speed,
        use_imu_memory=args.use_imu_memory,
        imu_safety=args.imu_safety,
        target_laps=args.target_laps,
    )
    os.makedirs(os.path.dirname(cfg.prefix) or ".", exist_ok=True)

    preview_env = LidarRacingEnv(track_name=cfg.track_name, use_imu_memory=cfg.use_imu_memory)
    print(f"Loading model: {cfg.model}  state_dim={preview_env.observation_dim}")
    policy = load_policy(cfg.model, state_dim=preview_env.observation_dim)
    safety = None
    if cfg.safety_enabled:
        safety = SafetyActionWrapper(SafetyConfig(
            enabled=True,
            max_speed_mps=cfg.safety_max_speed_mps,
            stop_distance_m=cfg.safety_stop_distance_m,
            caution_distance_m=cfg.safety_caution_distance_m,
            use_imu_safety=cfg.imu_safety,
        ))
        print(f"Safety wrapper ENABLED: max_speed={cfg.safety_max_speed_mps:.2f} m/s, "
              f"caution={cfg.safety_caution_distance_m:.2f} m, stop={cfg.safety_stop_distance_m:.2f} m")
    else:
        print("Safety wrapper DISABLED: raw model actions will drive the sim")
    env, rows, frames, summary = collect_rollout(
        policy,
        cfg.max_steps,
        safety=safety,
        track_name=cfg.track_name,
        start_speed_mps=cfg.start_speed_mps,
        use_imu_memory=cfg.use_imu_memory,
        target_laps=cfg.target_laps,
    )
    summary["model"] = cfg.model
    summary["track_name"] = cfg.track_name
    summary["config"] = asdict(cfg)

    csv_path = cfg.prefix + ".csv"
    summary_path = cfg.prefix + "_summary.json"
    png_path = cfg.prefix + "_final.png"
    anim_path = cfg.prefix + ("." + cfg.output_format if cfg.output_format != "none" else "")

    write_csv(rows, csv_path)
    draw_final_png(env, frames, summary, png_path)
    if cfg.output_format != "none":
        render_animation(env, frames, summary, anim_path, cfg.fps, cfg.frame_stride, cfg.output_format)

    summary["outputs"] = {
        "csv": csv_path,
        "summary": summary_path,
        "final_png": png_path,
        "animation": anim_path if cfg.output_format != "none" else None,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Model Rollout Visualization Complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
