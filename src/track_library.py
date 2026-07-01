"""F1-only track library for AIRACE Git package."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

_F1_TRACK_DATA = Path(__file__).with_name("data") / "f1_tracks.npz"
_F1_TRACK_MANIFEST = Path(__file__).with_name("data") / "f1_tracks_manifest.json"
_F1TENTH_TRACK_DATA = Path(__file__).with_name("data") / "f1tenth_imported_tracks.npz"
_F1TENTH_TRACK_MANIFEST = Path(__file__).with_name("data") / "f1tenth_imported_tracks_manifest.json"
_F1_MANIFEST_CACHE = None
_F1_DATA_CACHE = None
_F1TENTH_MANIFEST_CACHE = None
_F1TENTH_DATA_CACHE = None

def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def _load_f1_manifest():
    global _F1_MANIFEST_CACHE
    if _F1_MANIFEST_CACHE is None:
        _F1_MANIFEST_CACHE = _load_json(_F1_TRACK_MANIFEST)
    return _F1_MANIFEST_CACHE

def _load_f1_data():
    global _F1_DATA_CACHE
    if _F1_DATA_CACHE is None:
        if not _F1_TRACK_DATA.exists():
            raise FileNotFoundError(f"Track data missing: {_F1_TRACK_DATA}")
        _F1_DATA_CACHE = np.load(_F1_TRACK_DATA)
    return _F1_DATA_CACHE

def _load_f1tenth_manifest():
    global _F1TENTH_MANIFEST_CACHE
    if _F1TENTH_MANIFEST_CACHE is None:
        _F1TENTH_MANIFEST_CACHE = _load_json(_F1TENTH_TRACK_MANIFEST)
    return _F1TENTH_MANIFEST_CACHE

def _load_f1tenth_data():
    global _F1TENTH_DATA_CACHE
    if _F1TENTH_DATA_CACHE is None:
        if not _F1TENTH_TRACK_DATA.exists():
            raise FileNotFoundError(f"Track data missing: {_F1TENTH_TRACK_DATA}")
        _F1TENTH_DATA_CACHE = np.load(_F1TENTH_TRACK_DATA)
    return _F1TENTH_DATA_CACHE

def _get_f1_config(name: str):
    manifest = _load_f1_manifest()
    if name not in manifest:
        raise KeyError(name)
    data = _load_f1_data()
    meta = manifest[name]
    return {
        "description": f"1/7-scale F1 telemetry-derived layout: {meta['label']}. Length {meta['target_length_m']:.1f} m, lane width {meta['width_m']:.2f} m.",
        "width_m": float(meta["width_m"]),
        "ds_m": 0.50,
        "centerline_xy": np.asarray(data[name], dtype=np.float64),
        "metadata": meta,
    }

def _get_f1tenth_config(name: str):
    manifest = _load_f1tenth_manifest()
    if name not in manifest:
        raise KeyError(name)
    data = _load_f1tenth_data()
    meta = manifest[name]
    return {
        "description": f"Imported F1TENTH-style layout: {meta.get('label', name)}. Length {meta.get('target_length_m', 0.0):.1f} m, lane width {meta.get('width_m', 1.20):.2f} m.",
        "width_m": float(meta.get("width_m", 1.20)),
        "ds_m": float(meta.get("ds_m", 0.05)),
        "centerline_xy": np.asarray(data[name], dtype=np.float64),
        "metadata": meta,
    }

def list_track_names():
    return sorted(list(_load_f1_manifest()) + list(_load_f1tenth_manifest()))

def get_track_config(name: str | None):
    if name in (None, "", "default"):
        names = sorted(_load_f1_manifest())
        if not names:
            raise KeyError("No F1 tracks are available")
        name = names[0]
    if name in _load_f1_manifest():
        return _get_f1_config(name)
    if name in _load_f1tenth_manifest():
        return _get_f1tenth_config(name)
    raise KeyError(f"Unknown track {name!r}. Available: {', '.join(list_track_names())}")
