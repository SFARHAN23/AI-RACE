"""
AIRACE runner for the Git upload package.

Two modes:
1. Run all complete F1 tracks.
2. Run one selected F1 track by number.

Each selected track produces two MP4 videos:
- global camera
- follow camera

The three incomplete V26 tracks are intentionally excluded:
- f1_japan_suzuka
- f1_monaco
- f1_singapore_marina_bay
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TRACK_LIST = ROOT / "tracks" / "complete_f1_tracks_19.txt"
DEFAULT_MODEL = ROOT / "models" / "v26_recommended" / "v26_resume357__all__f1_2lap_final"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "videos"


def load_tracks() -> list[str]:
    tracks = [line.strip() for line in TRACK_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tracks:
        raise RuntimeError(f"No tracks found in {TRACK_LIST}")
    return tracks


def print_tracks(tracks: list[str]) -> None:
    print("\nComplete F1 tracks included in this package:\n")
    for i, name in enumerate(tracks, 1):
        short = name.replace("f1_", "").replace("", "").replace("_", " ")
        print(f"  {i:2d}. {short}  ({name})")
    print()


def write_temp_track_list(tracks: list[str], out_dir: Path) -> Path:
    p = out_dir / "selected_tracks.txt"
    p.write_text("\n".join(tracks) + "\n", encoding="utf-8")
    return p


def run_renderer(selected_tracks: list[str], output_dir: Path, model_prefix: Path, max_steps: int, frame_stride: int, fps: int, width: int, height: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_track_list = write_temp_track_list(selected_tracks, output_dir)
    cmd = [
        sys.executable,
        str(SRC / "render_all22_follow_global_videos.py"),
        "--model", str(model_prefix),
        "--track-list", str(temp_track_list),
        "--out-dir", str(output_dir),
        "--max-steps", str(max_steps),
        "--frame-stride", str(frame_stride),
        "--fps", str(fps),
        "--width", str(width),
        "--height", str(height),
    ]
    env = None
    print("\nRunning AIRACE renderer...")
    print("Command:", " ".join(cmd))
    print("Output folder:", output_dir)
    subprocess.run(cmd, check=True, cwd=str(SRC), env=env)
    print("\nDone. Videos are in:")
    print(output_dir)
    print("\nEach track should have:")
    print("- *_global_cam.mp4")
    print("- *_follow_cam.mp4")


def interactive_select(tracks: list[str]) -> list[str]:
    print("AIRACE run menu")
    print("1. Run all complete F1 tracks")
    print("2. Run a specific track by number")
    choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        return tracks
    if choice == "2":
        print_tracks(tracks)
        raw = input(f"Enter track number 1-{len(tracks)}: ").strip()
        try:
            idx = int(raw)
        except ValueError as exc:
            raise SystemExit("Track choice must be a number.") from exc
        if idx < 1 or idx > len(tracks):
            raise SystemExit(f"Track number must be between 1 and {len(tracks)}.")
        return [tracks[idx - 1]]
    raise SystemExit("Choice must be 1 or 2.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run AIRACE model videos for all complete F1 tracks or one selected track.")
    ap.add_argument("--mode", choices=["menu", "all", "one"], default="menu", help="menu=interactive, all=all complete tracks, one=single track")
    ap.add_argument("--track", type=int, default=None, help="Track number for --mode one. Use --list-tracks to see numbers.")
    ap.add_argument("--list-tracks", action="store_true", help="Print track numbers and exit.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Model prefix without _actor/_critic suffix.")
    ap.add_argument("--out", default=None, help="Output folder. Default creates outputs/videos/run_<timestamp>.")
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--frame-stride", type=int, default=8, help="Lower = smoother videos, larger files/slower rendering.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tracks = load_tracks()

    if args.list_tracks:
        print_tracks(tracks)
        return 0

    if args.mode == "all":
        selected = tracks
    elif args.mode == "one":
        if args.track is None:
            print_tracks(tracks)
            raise SystemExit("Use --track NUMBER with --mode one.")
        if args.track < 1 or args.track > len(tracks):
            raise SystemExit(f"Track number must be between 1 and {len(tracks)}.")
        selected = [tracks[args.track - 1]]
    else:
        selected = interactive_select(tracks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(args.out) if args.out else DEFAULT_OUTPUT_ROOT / f"run_{timestamp}").resolve()
    model_prefix = Path(args.model).resolve()

    actor_file = Path(str(model_prefix) + "_actor")
    if not actor_file.exists():
        raise SystemExit(f"Model actor file not found: {actor_file}")

    print("\nSelected tracks:")
    for t in selected:
        print("-", t)

    run_renderer(
        selected_tracks=selected,
        output_dir=output_dir,
        model_prefix=model_prefix,
        max_steps=args.max_steps,
        frame_stride=args.frame_stride,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
