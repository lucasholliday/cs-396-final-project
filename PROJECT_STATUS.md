# Project status — 2026-11-17 checkpoint

## Current phase
Initial audit, synchronization, whole-swing pose/features and baseline/nested ML research are executed. Native ball/shaft tracking remains experimental. A camera-only event detector was evaluated on Sid's full session. No polished app, calibrated metric speed, reliable nine-phase detector, human mesh, gaze or practice-swing validation.

## Completed and tested
- 79 originals / 11,976,204,529 bytes hashed and independently rehashed unchanged. All workbook sheets/CSV schemas/timing and media metadata audited.
- 76 reviewed swing identities, 72 labeled outcomes, five golfers/six sessions. User confirms ordinal matching including unregistered strikes; no practice swings.
- 68 club-covered events (65 with labels); eight missing tails consistent with SD-card failures. Affine gyro alignment residuals <36 ms. Exact impacts remain unverified.
- 76 pose clips / 7,980 frames, contact sheets visually checked. Pose availability 100%, mean both-wrists confidence validity only 18.9%. Excluded wrist pose predictors.
- 400 fixed-model evaluations and 20 nested carry comparisons with persisted shot/golfer splits and all OOF predictions; 360 learning-curve fits; paired uncertainty and error analysis; plots inspected.
- Primary nested n=72 carry MAE: video 25.45 yd known-session / 31.93 held-out golfer; video+watch 26.60 / 29.87. Matched n=65 all-sensor held-out-golfer MAE 28.58. Small-sample uncertainty remains large.
- Twelve fixed-ridge full-data reference fits saved under models/reference_ridge, excluded from Git. These are not independently validated deployment models.
- Native ball demonstration: manually initialized sid_1_01, 31 candidate points over 0.179 s. Automatic v4 batch ran on all 76: 36 internally gated proposals, all awaiting independent point review before ML use. Version failures preserved.
- Motion-gated Hough: nine plausible partial shaft segments on sid_1_01; unmasked baseline rejected for background false detections. Saved head-centroid trace, without gaze/metric-height interpretation.
- Seven tests pass; isolated .venv imports and pip dependency check pass. Python 3.10.7, FFmpeg 4.3.2, pinned dependencies in requirements/lock.

## Latest camera-only experiment
Sid full-session 20-fps mat-ball presence scan completed: 29 disappearance candidates; 18 within 1.5 s of the 18 distinct watch-reference swing neighborhoods. Eleven extra candidates and one noticeably early event prevent calling it an exact impact detector. Watch is used only for audit comparison, not detection.
Native trajectory filtering retained 18 proposals: 15 matched distinct reference neighborhoods, three were extras, and three reference swings were missed. Inspected events 05/12/19/27: plausible flight in 05/19, missed visible flight in 12, false leg trajectory in 27. These are exploratory single-session neighborhood counts, not exact impact or point-tracking accuracy. Results are preserved in reports/camera_events/.

## User-confirmed equipment and unresolved limits
All right handed; left lead wrist watch; standard Wilson Staff CB seven iron; sensor ~1 inch below distal grip edge. Nominal Wilson length 37.25 in / 0.94615 m. Exact chip-to-head geometry can be measured later. Current video PTS approximately matches wall time but variable frame cadence must be honored. No physical-speed conversion from constant FPS or total club length.
Club acceleration is saturated in all 68 windows and excluded from predictors. Watch units/range limiting are not independently established. Missing original label photographs prevent transcription rechecking; labels and supplemental corrections retained.
Current video benchmarks use watch-assisted clips and include follow-through. Ball proposals are not predictor inputs. No external-day test; only five golfers. See DATA_ISSUES and reports/model_results.md.

## Reproduction and artifacts
Full command order is in README. Key folders: configs/, scripts/, tests/, annotations/, data/manifests/, data/processed/, experiments/baseline_v1/, experiments/nested_carry_v1/, outputs/, models/reference_ridge/, reports/.
Use `.venv/bin/python`. Initial audit resets enriched manifest; follow with detect_events, align_club and extract_features. Source-bound shot_review annotations must not be auto-approved on changed files. Do not repeat cached expensive decoding.

Latest checks:
```
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_checkpoint.py
```
The final command rehashes ~12 GB; last run passed. No need to repeat without source changes.

## Next three priorities
1. Improve ball/foreground discrimination using the recorded camera-only false positives and misses; freeze parameters before evaluating another golfer.
2. Independently annotate representative native ball/shaft/impact frames across golfers, reject stationary-background tails and club confusions, then evaluate a separate post-impact feature benchmark. Do not train on the 36 unreviewed proposals.
3. Improve autonomous camera segmentation and phase detection; collect calibrated/securely logged new sessions per reports/data_collection_recommendations.md (100 strikes in two 50-shot checkpoints, plus 20 tagged practice swings).

## Git
A project-local repository on main was initialized with approval. Verified research checkpoint prepared for local commit; use `git log -1 --oneline` to identify the saved revision. No remote, no push. An unrelated home repository also exists: always verify repository root. Raw data, original workbook, generated media/cache, model binaries and environment are ignored. Stage only intended code/docs/annotations/derived research tables.
