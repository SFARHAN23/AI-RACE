from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from lidar_env import LidarRacingEnv
from td3_lidar import TD3
from rc_car_model import SPD_MAX, CarLength, CarWidth


def load_policy(model_prefix: str, state_dim: int) -> TD3:
    policy = TD3(state_dim, 2, 1.0)
    policy.load_cpu(model_prefix)
    return policy


def world_to_px(points, bounds, w, h, margin=36):
    pts = np.asarray(points, dtype=np.float64)
    xmin, xmax, ymin, ymax = bounds
    sx = (w - 2 * margin) / max(1e-9, xmax - xmin)
    sy = (h - 2 * margin) / max(1e-9, ymax - ymin)
    s = min(sx, sy)
    xoff = (w - s * (xmax - xmin)) / 2.0
    yoff = (h - s * (ymax - ymin)) / 2.0
    x = xoff + (pts[..., 0] - xmin) * s
    y = h - (yoff + (pts[..., 1] - ymin) * s)
    return np.stack([x, y], axis=-1).astype(np.int32)


def padded_bounds(track, pad=2.0):
    allxy = np.vstack([track.left_wall_xy, track.right_wall_xy, track.centerline_xy])
    xmin, ymin = allxy.min(axis=0)
    xmax, ymax = allxy.max(axis=0)
    return float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad)


def follow_bounds(pose, span=30.0):
    x, y = float(pose[0]), float(pose[1])
    half = span / 2.0
    return x - half, x + half, y - half, y + half


def draw_polyline(img, xy, bounds, color, thickness=2, closed=False):
    h, w = img.shape[:2]
    pts = world_to_px(xy, bounds, w, h).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], closed, color, thickness, lineType=cv2.LINE_AA)


def draw_start_finish(img, track, bounds):
    h, w = img.shape[:2]
    left = track.left_wall_xy[0]
    right = track.right_wall_xy[0]
    mid = (left + right) / 2.0
    p = world_to_px(np.vstack([left, right, mid]), bounds, w, h)
    cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 220, 255), 7, cv2.LINE_AA)
    cv2.line(img, tuple(p[0]), tuple(p[1]), (8, 17, 31), 2, cv2.LINE_AA)
    cv2.putText(img, "START / FINISH", tuple(p[2] + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2, cv2.LINE_AA)


def draw_car(img, pose, bounds):
    h, w = img.shape[:2]
    x, y, th = [float(v) for v in pose]
    L = CarLength * 1.7
    W = CarWidth * 2.1
    corners = np.array([[L/2, 0], [-L/2, W/2], [-L/3, 0], [-L/2, -W/2]], dtype=np.float64)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pts = corners @ R.T + np.array([x, y])
    px = world_to_px(pts, bounds, w, h).reshape((-1, 1, 2))
    cv2.fillPoly(img, [px], (65, 185, 255), lineType=cv2.LINE_AA)
    cv2.polylines(img, [px], True, (255, 255, 255), 2, cv2.LINE_AA)


def draw_scene(track, frames, idx, bounds, mode, track_name, reason, out_size):
    w, h = out_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (14, 25, 42)
    # subtle grid
    for gx in range(0, w, 80):
        cv2.line(img, (gx, 0), (gx, h), (25, 43, 68), 1)
    for gy in range(0, h, 80):
        cv2.line(img, (0, gy), (w, gy), (25, 43, 68), 1)

    draw_polyline(img, track.left_wall_xy, bounds, (235, 242, 250), 3, True)
    draw_polyline(img, track.right_wall_xy, bounds, (235, 242, 250), 3, True)
    draw_polyline(img, track.centerline_xy, bounds, (110, 125, 150), 1, True)
    draw_start_finish(img, track, bounds)

    poses = np.array([f["pose"][:2] for f in frames[:idx+1]], dtype=np.float64)
    if len(poses) > 1:
        draw_polyline(img, poses, bounds, (255, 190, 55), 4, False)
        draw_polyline(img, poses, bounds, (50, 210, 255), 2, False)

    f = frames[idx]
    draw_car(img, f["pose"], bounds)
    # car start marker
    start_px = world_to_px(np.array([frames[0]["pose"][:2]]), bounds, w, h)[0]
    cv2.circle(img, tuple(start_px), 6, (70, 230, 120), -1, cv2.LINE_AA)
    cv2.putText(img, "CAR START", tuple(start_px + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 230, 120), 1, cv2.LINE_AA)

    title = f"AIRACE best TD3 model | {track_name} | {mode.upper()} CAM"
    cv2.rectangle(img, (0, 0), (w, 78), (6, 17, 31), -1)
    cv2.putText(img, title, (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 248, 252), 2, cv2.LINE_AA)
    cv2.putText(img, f"lap {f['lap']}/2 | progress {f['progress_pct']:.1f}% | speed {f['speed_mps']:.2f} m/s | {f['status']}", (24, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 205, 225), 1, cv2.LINE_AA)
    cv2.putText(img, "yellow gate = START / FINISH", (w - 360, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
    return img


def rollout(policy, track_name, max_steps, frame_stride):
    env = LidarRacingEnv(track_name=track_name, use_imu_memory=True, target_laps=2, adaptive_spawn=False, max_steps=max_steps)
    obs = env.test_reset(spd=0.45, lateral_offset_m=0.0)
    frames = []
    reason = "MAX_STEPS"
    for step in range(max_steps):
        obs_vec = np.array(obs, dtype=np.float64)
        raw = policy.select_action(obs_vec)
        action = np.clip(raw, -1, 1)
        next_obs, reward, done, info = env.step(action)
        if step % frame_stride == 0 or done:
            frames.append({
                "step": step,
                "pose": env.car.pose.copy(),
                "lap": int(info.get("lap_count", 0)) + 1,
                "progress_pct": float(100.0 * info.get("progress", 0.0) / env.track.total_trip),
                "speed_mps": float(obs_vec[0] * SPD_MAX),
                "status": info.get("reason", "RUNNING") if done else "RUNNING",
            })
        obs = next_obs
        if done:
            reason = info.get("reason", env.query_fail_reason())
            break
    if not frames:
        frames.append({"step": 0, "pose": env.car.pose.copy(), "lap": 1, "progress_pct": 0.0, "speed_mps": 0.0, "status": reason})
    if frames[-1]["status"] == "RUNNING":
        frames[-1]["status"] = reason
    return env, frames, reason


def write_video(path, track, frames, mode, track_name, reason, fps, out_size):
    path = str(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    global_bounds = padded_bounds(track, 2.5)
    for i, f in enumerate(frames):
        bounds = global_bounds if mode == "global" else follow_bounds(f["pose"], span=32.0)
        img = draw_scene(track, frames, i, bounds, mode, track_name, reason, out_size)
        writer.write(img)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--track-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--frame-stride", type=int, default=20)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = [x.strip() for x in Path(args.track_list).read_text().splitlines() if x.strip()]
    preview = LidarRacingEnv(track_name=tracks[0], use_imu_memory=True)
    policy = load_policy(args.model, preview.observation_dim)

    for idx, track_name in enumerate(tracks, 1):
        safe_name = track_name.replace("/", "_")
        print(f"[{idx:02d}/{len(tracks)}] rollout {track_name}", flush=True)
        env, frames, reason = rollout(policy, track_name, args.max_steps, args.frame_stride)
        for mode in ("global", "follow"):
            out_path = out_dir / f"{idx:02d}_{safe_name}_{mode}_cam.mp4"
            print(f"  render {mode}: {out_path.name} frames={len(frames)} reason={reason}", flush=True)
            write_video(out_path, env.track, frames, mode, track_name, reason, args.fps, (args.width, args.height))
    print(f"DONE {out_dir}", flush=True)


if __name__ == "__main__":
    main()
