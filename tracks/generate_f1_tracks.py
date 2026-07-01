"""Generate 1/7-scale F1 circuit centerlines from FastF1 telemetry.

This creates data/f1_tracks.npz plus a JSON manifest.  The track
library loads those centerlines by name.  Coordinates come from each  race's
fastest-lap telemetry (X/Y position channels), then are normalized, rotated,
closed, resampled, and scaled so the centerline length equals official F1 lap
length / 7.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Official race lap lengths in meters for the 22-race  F1 calendar.
# Source values are the standard FIA/F1 circuit lap distances.
F1_TRACKS = [
    (1, "f1_bahrain_sakhir", "Bahrain International Circuit / Sakhir", 5412.0, 14.0),
    (2, "f1_saudi_jeddah", "Jeddah Corniche Circuit", 6174.0, 14.0),
    (3, "f1_australia_melbourne", "Albert Park / Melbourne", 5278.0, 14.0),
    (4, "f1_imola", "Autodromo Enzo e Dino Ferrari / Imola", 4909.0, 12.0),
    (5, "f1_miami", "Miami International Autodrome", 5412.0, 14.0),
    (6, "f1_spain_barcelona", "Circuit de Barcelona-Catalunya", 4675.0, 12.0),
    (7, "f1_monaco", "Circuit de Monaco", 3337.0, 10.0),
    (8, "f1_azerbaijan_baku", "Baku City Circuit", 6003.0, 13.0),
    (9, "f1_canada_montreal", "Circuit Gilles-Villeneuve / Montreal", 4361.0, 12.0),
    (10, "f1_britain_silverstone", "Silverstone Circuit", 5891.0, 13.0),
    (11, "f1_austria_spielberg", "Red Bull Ring / Spielberg", 4318.0, 12.0),
    (12, "f1_france_paul_ricard", "Circuit Paul Ricard / Le Castellet", 5842.0, 12.0),
    (13, "f1_hungary_hungaroring", "Hungaroring / Budapest", 4381.0, 12.0),
    (14, "f1_belgium_spa", "Spa-Francorchamps", 7004.0, 12.0),
    (15, "f1_netherlands_zandvoort", "Circuit Zandvoort", 4259.0, 10.0),
    (16, "f1_italy_monza", "Autodromo Nazionale Monza", 5793.0, 12.0),
    (17, "f1_singapore_marina_bay", "Marina Bay Street Circuit", 5063.0, 10.0),
    (18, "f1_japan_suzuka", "Suzuka Circuit", 5807.0, 12.0),
    (19, "f1_usa_cota", "Circuit of the Americas / Austin", 5513.0, 14.0),
    (20, "f1_mexico_mexico_city", "Autodromo Hermanos Rodriguez", 4304.0, 12.0),
    (21, "f1_brazil_interlagos", "Interlagos / Sao Paulo", 4309.0, 12.0),
    (22, "f1_abu_dhabi_yas_marina", "Yas Marina Circuit", 5281.0, 12.0),
]


def _clean_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 50:
        raise ValueError(f"too few telemetry points: {len(xy)}")
    # Drop consecutive duplicate/stationary samples.
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, d > 1e-6]
    xy = xy[keep]
    return xy


def _resample_closed(xy: np.ndarray, target_ds: float = 0.50) -> tuple[np.ndarray, float]:
    """Close and resample a polyline uniformly by arc length."""
    xy = _clean_xy(xy)
    # Close if not already closed.
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
        xy = np.vstack([xy, xy[0]])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    xy = xy[keep]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    total = float(s[-1])
    n = max(200, int(round(total / target_ds)))
    new_s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(new_s, s, xy[:, 0])
    y = np.interp(new_s, s, xy[:, 1])
    out = np.column_stack([x, y])
    return out, total


def _normalize_orientation(xy: np.ndarray) -> np.ndarray:
    """Center and rotate PCA-long-axis to X while preserving telemetry lap start.

    Earlier versions also re-indexed the path to the left-most point for prettier
    repeatable display. That made the simulator lap start/wrap arbitrary instead
    of the real FastF1 lap timing point. For training/evaluation we preserve row
    0 from FastF1 fastest-lap telemetry, which is the lap start/finish timing
    point, then only translate/rotate the whole shape.
    """
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    ang = math.atan2(axis[1], axis[0])
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    xy = xy @ rot.T
    return xy


def _telemetry_centerline(year: int, round_no: int) -> np.ndarray:
    import fastf1

    # Qualifying is much smaller than the race and still provides a clean fast
    # racing-line lap around the same physical circuit.  Fall back to Race if a
    # Q session download/API call fails.
    last_error = None
    for session_code in ("Q", "R"):
        try:
            session = fastf1.get_session(year, round_no, session_code)
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            laps = session.laps.pick_quicklaps()
            if laps.empty:
                laps = session.laps
            lap = laps.pick_fastest()
            tel = lap.get_telemetry()[["X", "Y"]].dropna()
            xy = tel.to_numpy(dtype=np.float64)
            if len(xy) >= 50:
                return xy
        except Exception as exc:  # pragma: no cover - network/data fallback
            last_error = exc
            print(f"    {session_code} telemetry failed for round {round_no}: {exc!r}; trying fallback", flush=True)
    raise RuntimeError(f"Could not load telemetry for  round {round_no}: {last_error!r}")


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    npz_path = out_dir / "f1_tracks.npz"
    manifest_path = out_dir / "f1_tracks_manifest.json"

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}

    for round_no, key, label, official_len_m, official_width_m in F1_TRACKS:
        print(f"[{round_no:02d}/22] downloading/building {key} ...", flush=True)
        raw = _telemetry_centerline(2000 + 22, round_no)
        raw = _normalize_orientation(raw)
        raw_resampled, raw_len = _resample_closed(raw, target_ds=5.0)
        target_len = official_len_m / 7.0
        scale = target_len / raw_len
        scaled = raw_resampled * scale
        scaled = _normalize_orientation(scaled)
        # Resample final scaled centerline at 0.50 m. This is much lighter than
        # 0.02 m for ~500-1000 m F1 layouts, but still detailed enough for LiDAR.
        final_xy, final_len = _resample_closed(scaled, target_ds=0.50)
        arrays[key] = final_xy.astype(np.float32)
        width_scaled = official_width_m / 7.0
        manifest[key] = {
            "round": round_no,
            "label": label,
            "official_length_m": official_len_m,
            "scale": "1/7",
            "target_length_m": target_len,
            "generated_length_m": final_len,
            "official_width_m_assumed": official_width_m,
            "width_m": width_scaled,
            "points": int(len(final_xy)),
            "source": "FastF1 race fastest-lap telemetry X/Y, scaled to official lap length / 7",
        }
        print(f"    length {final_len:.1f} m, width {width_scaled:.2f} m, points {len(final_xy)}")

    np.savez_compressed(npz_path, **arrays)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {npz_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
