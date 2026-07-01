# Package Manifest

GitHub-ready F1-only AIRACE upload package.

## Main entry point

`run_airace.py`

It supports:

1. Run all complete F1 tracks.
2. Run one specific track by number.

Each selected track creates two MP4 outputs:

- global camera
- follow camera

## Included tracks

Track list:

`tracks/complete_f1_tracks_19.txt`

The three incomplete V26 tracks are excluded and listed in:

`tracks/excluded_incomplete_tracks.txt`

## Main model

`models/v26_recommended/v26_resume357_all_f1_2lap_final`

Files:

- `_actor`
- `_critic`
- `_actor_optimizer`
- `_critic_optimizer`

## Experimental model

`models/v27_experimental/v27_11h30_all_f1_2lap_530`

Use only for comparison.

## Core code

`src/`

Contains the environment, TD3 model, renderer, car dynamics, LiDAR simulator, safety utilities, and F1 track loader.

## Training code

`training/`

Contains the training scripts for the TD3 setup.

## Reports

`reports/`

Contains final result summaries and model status notes.

## Notes

This package is focused on the model, tracks, training/evaluation code, and video rollout runner.

## Included final video files

`output/videos/f1_complete_track_videos/` contains 38 MP4 files: 19 complete tracks x global/follow cameras.

These are the final final outputs and are included for direct GitHub handoff/showcase.


## Added handoff files

- `CAR_SPECIFICATIONS.md` — car dimensions, acceleration, force/torque equations, LiDAR specs, bicycle-model equations, and reward constants.
- `output/videos/f1_complete_track_videos/` — final paired global/follow demonstration videos.
