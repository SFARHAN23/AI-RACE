# V27 stop + status

Training was stopped on Farhan's instruction. Verified no `version_27_11h30`, `train_lidar_round_robin`, or `py.exe` training process remains active.

## Run

Run folder:
`/mnt/c/Users/farha/Documents/AIRACE/04_TRAINING_RUNS/version_27_11h30_all_f1_round_robin_2lap_resume_v26_20260630_212410`

Log:
`/mnt/c/Users/farha/Documents/AIRACE/04_TRAINING_RUNS/version_27_11h30_all_f1_round_robin_2lap_resume_v26_20260630_212410/logs/train_round_robin.log`

Latest saved checkpoint prefix:
`/mnt/c/Users/farha/Documents/AIRACE/03_TD3_MODEL_CODE/version_07_adaptive_spawn_2lap_smooth_source/models/version_27_11h30_all_f1_round_robin_2lap_resume_v26_20260630_212410/resume_from_v26_final/v27_11h30_all_f1_2lap_530`

Note: because the run was manually killed, there is no clean `_final` checkpoint for V27. Use `_530` as the latest saved checkpoint.

## Final parsed training status

latest_ep=754 total_env_steps=5306293 train_steps=5300539 rolling_completion=71%

| # | Track | Latest result | Latest laps | 2-lap finishes | Best 2-lap time |
|---|---|---:|---:|---:|---:|
| 1 | bahrain sakhir | FINISHED | 2/2 | 32 | 192.80s |
| 2 | saudi jeddah | FINISHED | 2/2 | 30 | 219.74s |
| 3 | australia melbourne | FINISHED | 2/2 | 33 | 188.26s |
| 4 | imola | FINISHED | 2/2 | 33 | 174.58s |
| 5 | miami | FINISHED | 2/2 | 24 | 192.96s |
| 6 | spain barcelona | FINISHED | 2/2 | 32 | 166.58s |
| 7 | monaco | COLLISION | 0/2 | 0 | not solved |
| 8 | azerbaijan baku | FINISHED | 2/2 | 13 | 213.76s |
| 9 | canada montreal | FINISHED | 2/2 | 24 | 155.56s |
| 10 | britain silverstone | COLLISION | 0/2 | 0 | not solved |
| 11 | austria spielberg | FINISHED | 2/2 | 34 | 154.02s |
| 12 | france paul ricard | FINISHED | 2/2 | 31 | 207.36s |
| 13 | hungary hungaroring | FINISHED | 2/2 | 34 | 155.94s |
| 14 | belgium spa | COLLISION | 0/2 | 20 | 249.36s |
| 15 | netherlands zandvoort | FINISHED | 2/2 | 9 | 150.74s |
| 16 | italy monza | COLLISION | 0/2 | 26 | 206.54s |
| 17 | singapore marina bay | COLLISION | 0/2 | 0 | not solved |
| 18 | japan suzuka | WRONG_DIR | 0/2 | 0 | not solved |
| 19 | usa cota | FINISHED | 2/2 | 34 | 196.14s |
| 20 | mexico mexico city | FINISHED | 2/2 | 31 | 153.20s |
| 21 | brazil interlagos | FINISHED | 2/2 | 32 | 153.44s |
| 22 | abu dhabi yas marina | FINISHED | 2/2 | 19 | 188.32s |

## Improvement vs V26 baseline log

Short answer: partially improved.

Positive:
- Every comparable solved track got faster: 18/18 comparable tracks improved best 2-lap time.
- Average best-time gain on comparable solved tracks: 2.33 seconds faster.
- Total best-time gain across those comparable tracks: 41.86 seconds.
- Azerbaijan latest status improved from V26 latest COLLISION to V27 latest FINISHED.

Negative:
- Overall reliability did not improve yet: rolling completion is lower, V26 77% -> V27 71%.
- Solved-any count decreased from 19/22 to 18/22 because Britain Silverstone had one solved run in V26 but zero solved runs in V27.
- Still unsolved in V27: Monaco, Britain Silverstone, Singapore, Japan.
- Latest episodes still show collisions on hard tracks: Monaco, Britain, Belgium, Italy Monza, Singapore; Japan ended WRONG_DIR.

Recommendation:
- Do not call V27 globally better yet.
- Keep V26 final as the safer general working model unless rollout evaluation of V27 checkpoint `_530` proves better.
- V27 `_530` is worth evaluating because the solved tracks are faster, but it may be less reliable on the hard/problem tracks.
