# F1 1/7-Scale Track Pack

Created for AIRACE TD3/LiDAR simulator.

## What changed

- Added `generate_f1_tracks.py`.
  - Downloads/imports FastF1 qualifying telemetry.
  - Extracts each circuit's fastest-lap X/Y path.
  - Normalizes and closes the centerline.
  - Scales each layout to official F1 lap length / 7.
  - Saves all tracks into `data/f1_tracks.npz`.
  - Saves metadata into `data/f1_tracks_manifest.json`.

- Added high-detail centerline track support to `rc_track.py`.
  - Existing old segment tracks still work.
  - New F1 tracks can load direct Nx2 centerline coordinates.
  - Walls, LiDAR, collision checks, adaptive spawn, progress, and lap logic use the same downstream simulator path.

- Updated `track_library.py`.
  - It now lazily exposes all generated F1 tracks through normal `--track-name` usage.

- Added `render_f1_tracks_contact_sheet.py`.
  - Renders all 22 imported tracks for inspection.

## Generated files

- `data/f1_tracks.npz`
- `data/f1_tracks_manifest.json`
- `results/f1_tracks_contact_sheet.png`

## Track source and accuracy

These are telemetry-derived shapes from FastF1, not hand-drawn toy approximations. Each track uses the fastest-lap telemetry path from the  race weekend qualifying session, then is scaled to official lap length / 7.

The path therefore follows an F1 racing line/telemetry centerline approximation. It is much more realistic than manually using a few straights/arcs, but it is still not a surveyed CAD blueprint of FIA track boundaries.

## Available track names

- `f1_bahrain_sakhir`
- `f1_saudi_jeddah`
- `f1_australia_melbourne`
- `f1_imola`
- `f1_miami`
- `f1_spain_barcelona`
- `f1_monaco`
- `f1_azerbaijan_baku`
- `f1_canada_montreal`
- `f1_britain_silverstone`
- `f1_austria_spielberg`
- `f1_france_paul_ricard`
- `f1_hungary_hungaroring`
- `f1_belgium_spa`
- `f1_netherlands_zandvoort`
- `f1_italy_monza`
- `f1_singapore_marina_bay`
- `f1_japan_suzuka`
- `f1_usa_cota`
- `f1_mexico_mexico_city`
- `f1_brazil_interlagos`
- `f1_abu_dhabi_yas_marina`

## Verification

Passed:

```bash
py.exe -3 -m py_compile rc_track.py track_library.py generate_f1_tracks.py render_f1_tracks_contact_sheet.py
```

Passed simulator load test:

- 22 F1 tracks found in `track_library.list_track_names()`.
- All 22 instantiate with `RCTrackClass(track_name=...)`.
- Every generated track length is within 2 meters of official length / 7.
- Contact sheet rendered successfully with all 22 outlines.

## Important note

No TD3 training was started. These are only simulator track assets/code changes. Long training on these F1 layouts still needs explicit authorization because the tracks are much longer than the earlier small test layouts.
