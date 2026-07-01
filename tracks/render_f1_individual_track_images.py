"""Render one PNG per generated 1/7-scale F1 track."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from rc_track import RCTrackClass
from track_library import get_track_config, list_track_names


OUT_DIR = Path("results") / "f1_track_images"


def _nice_name(track_name: str) -> str:
    return track_name.replace("f1_", "").replace("", "")


def render_track(track_name: str) -> Path:
    track = RCTrackClass(track_name=track_name)
    cfg = get_track_config(track_name)
    meta = cfg.get("metadata", {})
    title = _nice_name(track_name)

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Track lane/walls: red outer boundaries, black centerline.
    ax.plot(track.left_wall_xy[:, 0], track.left_wall_xy[:, 1], color="#e11d48", linewidth=1.8, label="left wall")
    ax.plot(track.right_wall_xy[:, 0], track.right_wall_xy[:, 1], color="#e11d48", linewidth=1.8, label="right wall")
    ax.plot(track.centerline_xy[:, 0], track.centerline_xy[:, 1], color="#0f172a", linewidth=1.1, alpha=0.90, label="centerline")

    # Start marker.
    start = track.centerline_xy[0]
    ax.scatter([start[0]], [start[1]], s=70, color="#22c55e", edgecolor="#052e16", linewidth=1.0, zorder=5)
    ax.text(start[0], start[1], " START", fontsize=9, color="#14532d", va="center", ha="left")

    # Equal aspect and padded bounds.
    all_xy = track.centerline_xy
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    pad = max(8.0, span * 0.08)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")

    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.6)
    ax.set_xlabel("meters, 1/7-scale sim coordinate")
    ax.set_ylabel("meters, 1/7-scale sim coordinate")

    official = meta.get("official_length_m")
    target = meta.get("target_length_m")
    width = track.width
    label = meta.get("label", title)
    subtitle = f"{label}\nGenerated length {track.total_trip:.1f} m | official/7 target {target:.1f} m | lane width {width:.2f} m" if target else f"{label}\nGenerated length {track.total_trip:.1f} m | lane width {width:.2f} m"
    fig.suptitle(f"{title} — F1 1/7-scale AIRACE track", fontsize=18, fontweight="bold", y=0.965)
    ax.set_title(subtitle, fontsize=11, pad=12)

    info = (
        "Source: FastF1 qualifying fastest-lap telemetry X/Y\n"
        "Scale: official F1 lap length / 7\n"
        "Red: simulated cardboard/boundary walls | Black: centerline | Green: start"
    )
    ax.text(
        0.01,
        0.01,
        info,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#94a3b8", alpha=0.90),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{track_name}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    names = [n for n in list_track_names() if n.startswith("f1_")]
    print(f"Rendering {len(names)} F1 track images into {OUT_DIR.resolve()}")
    for i, name in enumerate(names, start=1):
        out = render_track(name)
        print(f"[{i:02d}/{len(names):02d}] {out}")
    print("DONE")


if __name__ == "__main__":
    main()
