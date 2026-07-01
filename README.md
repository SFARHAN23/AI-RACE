# AIRACE TD3 F1 LiDAR Racing Simulator

This is a GitHub-ready AIRACE package focused on F1-style autonomous racing simulation.

It includes:

- TD3 neural-network policy code.
- F1 telemetry-derived track pack.
- Recommended trained model checkpoint.
- Training and evaluation scripts.
- A simple runner that generates two video outputs per selected track:
  - global camera
  - follow camera


## Recommended model

Use the V26 final checkpoint as the main model:

`models/v26_recommended/v26_resume357_all_f1_2lap_final`

Why:

- Clean final checkpoint from the V26 all-F1 round-robin 2-lap run.
- 19/22 tracks achieved at least one 2-lap finish.
- Overall finish rate: 75.0%.
- Last-100 finish rate: 77.0%.

V27 is included only as an experimental faster candidate:

`models/v27_experimental/v27_11h30_all_f1_2lap_530`

Do not use V27 as the default unless it passes separate rollout evaluation.

## Excluded incomplete tracks

The runner excludes these three incomplete V26 tracks:

- `f1_japan_suzuka`
- `f1_monaco`
- `f1_singapore_marina_bay`

The included complete/working list is in:

`tracks/complete_f1_tracks_19.txt`

## Folder layout

```text
GITUPLOAD_F1_ONLY/
  README.md
  PACKAGE_MANIFEST.md
  requirements.txt
  run_airace.py

  src/
    Simulator, TD3, environment, renderer, and track-loading code.

  training/
    Training scripts.

  tracks/
    F1 track data, manifests, and included/excluded track lists.

  models/
    v26_recommended/
    v27_experimental/

  reports/
    Result summaries and model comparison notes.

  outputs/videos/
    Generated videos go here when you run the model.
```


## Included final videos

This final upload includes the actual final final videos, not quick validation videos.

Folder:

`output/videos/f1_complete_track_videos/`

Contents:

- 38 MP4 files
- 19 complete F1 tracks
- 2 cameras per track: global and follow

The three incomplete tracks are not included in the video folder.

These videos are ready to show directly; you do not need to run the renderer unless you want to regenerate them.

## Install

```bash
pip install -r requirements.txt
```

On Windows Python from WSL:

```bash
py.exe -3 -m pip install -r requirements.txt
```

## Run option 1: menu

```bash
python run_airace.py
```

It asks:

```text
1. Run all complete F1 tracks
2. Run a specific track by number
```

## Run option 2: all complete tracks

```bash
python run_airace.py --mode all
```

This runs the 19 included complete F1 tracks and creates two videos per track.

Output folder example:

```text
outputs/videos/run_YYYYMMDD_HHMMSS/
  01_f1_abu_dhabi_yas_marina_global_cam.mp4
  01_f1_abu_dhabi_yas_marina_follow_cam.mp4
  ...
```

## Run option 3: one specific track

First list the track numbers:

```bash
python run_airace.py --list-tracks
```

Then run one track:

```bash
python run_airace.py --mode one --track 1
```

Example with Windows Python:

```bash
py.exe -3 run_airace.py --mode one --track 1
```

## Video output

Every selected track generates:

- `*_global_cam.mp4`
- `*_follow_cam.mp4`

Default render settings:

- 1280x720
- 30 FPS
- frame stride 8
- max steps 24000

For faster test rendering, increase stride:

```bash
python run_airace.py --mode one --track 1 --frame-stride 30
```

For smoother videos, keep stride low:

```bash
python run_airace.py --mode one --track 1 --frame-stride 8
```

## Model input/output

Observation shape for V26: 36 floats.

Action output: 2 floats.

```text
action[0] = throttle/brake
action[1] = steering
```

Both outputs are in the range `[-1, 1]`.

## Reports

See:

- `reports/V26_FINAL_RESULTS.md`
- `reports/v26_final_training_summary.json`
- `reports/V27_STOP_AND_STATUS.md`

## License

MIT License.
