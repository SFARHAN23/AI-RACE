# Version 26 final working results
Training resume from checkpoint 357 completed normally at the requested 660 minute wall-time limit.
Parsed episodes: 524 (ep 169 to 692).
Final counters: total steps 5,393,114; train steps 5,387,423.
Overall finish episodes: 393/524 = 75.0%. Last 100 finish rate: 77.0%.
Tracks with at least one 2-lap finish: 19/22.

Best deployable model prefix:
C:/Users/farha/Documents/AIRACE/03_TD3_MODEL_CODE/version_07_adaptive_spawn_2lap_smooth_source/models/version_26_all_f1_round_robin_2lap_20260629_0945/resume_from_357/v26_resume357_all_f1_2lap_final

Latest numbered checkpoint prefix:
C:/Users/farha/Documents/AIRACE/03_TD3_MODEL_CODE/version_07_adaptive_spawn_2lap_smooth_source/models/version_26_all_f1_round_robin_2lap_20260629_0945/resume_from_357/v26_resume357_all_f1_2lap_538

Per-track training log result:
- f1_abu_dhabi_: finishes 21/24, best 189.48s, latest FINISHED laps 2/2 ep 681
- f1_australia_: finishes 24/24, best 189.70s, latest FINISHED laps 2/2 ep 684
- f1_austria_sp: finishes 22/24, best 155.80s, latest RUNNING laps 0/2 ep 692
- f1_azerbaijan: finishes 14/24, best 217.28s, latest COLLISION laps 0/2 ep 689
- f1_bahrain_sa: finishes 24/24, best 193.66s, latest FINISHED laps 2/2 ep 682
- f1_belgium_sp: finishes 19/23, best 251.78s, latest FINISHED laps 2/2 ep 673
- f1_brazil_int: finishes 24/24, best 154.66s, latest FINISHED laps 2/2 ep 680
- f1_britain_si: finishes 1/24, best 225.42s, latest COLLISION laps 0/2 ep 691
- f1_canada_mon: finishes 21/24, best 157.52s, latest FINISHED laps 2/2 ep 690
- f1_france_pau: finishes 23/23, best 210.68s, latest FINISHED laps 2/2 ep 671
- f1_hungary_hu: finishes 22/23, best 158.42s, latest FINISHED laps 2/2 ep 672
- f1_imola: finishes 24/24, best 177.44s, latest FINISHED laps 2/2 ep 685
- f1_italy_monz: finishes 22/24, best 209.24s, latest FINISHED laps 2/2 ep 675
- f1_japan_suzu: finishes 0/24, best not solved, latest WRONG_DIR laps 0/2 ep 677
- f1_mexico_mex: finishes 24/24, best 154.80s, latest FINISHED laps 2/2 ep 679
- f1_miami: finishes 20/24, best 195.20s, latest FINISHED laps 2/2 ep 686
- f1_monaco: finishes 0/24, best not solved, latest WRONG_DIR laps 0/2 ep 688
- f1_netherland: finishes 17/23, best 155.56s, latest FINISHED laps 2/2 ep 674
- f1_saudi_jedd: finishes 24/24, best 222.30s, latest FINISHED laps 2/2 ep 683
- f1_singapore_: finishes 0/24, best not solved, latest WRONG_DIR laps 0/2 ep 676
- f1_spain_barc: finishes 23/24, best 168.90s, latest FINISHED laps 2/2 ep 687
- f1_usa_cota_1: finishes 24/24, best 198.74s, latest FINISHED laps 2/2 ep 678

Known weak/unsolved: Monaco, Singapore, Japan/Suzuka have zero 2-lap finishes in the resume log; Britain/Silverstone has only one finish and is unreliable.

Artifacts in this folder:
- final_training_summary.json
- resume_from_357_training_console.log
- eval_full/ and eval_none/ dashboard CSV/PNG/JSON sanity checks
