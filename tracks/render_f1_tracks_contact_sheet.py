"""Render a contact sheet for all 1/7-scale F1 tracks."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from rc_track import RCTrackClass
from track_library import list_track_names, get_track_config


def main() -> None:
    names = [n for n in list_track_names() if n.startswith("f1_")]
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        track = RCTrackClass(track_name=name)
        cfg = get_track_config(name)
        meta = cfg.get("metadata", {})
        ax.plot(track.left_wall_xy[:, 0], track.left_wall_xy[:, 1], color="#c0392b", linewidth=0.7)
        ax.plot(track.right_wall_xy[:, 0], track.right_wall_xy[:, 1], color="#c0392b", linewidth=0.7)
        ax.plot(track.centerline_xy[:, 0], track.centerline_xy[:, 1], color="#111111", linewidth=0.9)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        short = name.replace("f1_", "").replace("", "")
        ax.set_title(f"{short}\n{track.total_trip:.1f} m, width {track.width:.2f} m", fontsize=9)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("F1 circuits imported from FastF1 telemetry, scaled 1/7", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("results") / "f1_tracks_contact_sheet.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
