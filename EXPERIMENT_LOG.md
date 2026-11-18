# Experiment log

## A001 — Integrity/timing audit, 2026-11-17
Executed `python3 scripts/audit_dataset.py`: 79 files, 11.98 GB hashed; 76 candidates, 72 populated TrackMan labels. No byte-identical recording duplicates (shared metadata only). All CSVs parse. MOV nominal 240 fps but average ~159 playback fps; club mean ~188.6 Hz, watch ~99.23 Hz. 2,052 acceleration-limit flags, no gyro flags. Rohan 2 microphone has invalid container. Initial script stopped on corrupt audio; fixed to record failure and continue, rerun succeeded. Seven reports saved in `data/manifests/`.

## A002 — Sparse video decode
Executed `python3 scripts/sample_videos.py`: 18 windows, 0.25 playback seconds at 10/50/90% per video, 38–40 frames per window, zero exact adjacent duplicates. Six contact sheets visually inspected. Saved samples/hashes/results in `outputs/audit/`. Full-file decode, near-duplicates and actual strike visibility still untested. Failures fixed: incompatible system Pillow replaced by FFmpeg hstack; old FFmpeg rejects `.25`, changed to `0.25`.

## Validation
`python3 -m unittest discover -s tests -v`: three timing tests passed (units, reversal/gap, singleton). User clarifications recorded in DATA_ISSUES and config. No prediction experiment or trained model yet. No splits until event windows verified. Future seed 396.

## A003 — Event detection and provisional synchronization
`scripts/detect_events.py`: watch gyro norm threshold 25, minimum separation 3 seconds yields expected 18/5/10/12/11/20. Counts also stable at 30; full threshold sensitivity CSV saved. Peaks are not impact labels. Python 3.10 ISO parser rejected offset without colon; fixed with explicit strptime.
`scripts/check_event_alignment.py`: inspected all six first/last video montage pairs using watch epoch minus camera creation tag. Both endpoints show intended swings, supporting approximate real-time video PTS (not proof of millisecond alignment).
`scripts/align_club.py`: robust affine candidate-peak match, 68 matches, residual maxima <36 ms. Missing tails: sid_1_18; rohan_1_04–05; rohan_2_08–10; lucas_1_12; andrew_1_11. Rohan 1 scale based on only 3 events is weakly determined. No club samples fabricated.

## A004 — Pose baseline
Created isolated `.venv` after system Pillow/TensorFlow architecture conflicts. Approved downloads: dependencies plus Google's MediaPipe full pose model, no uploads. Six representative 30-fps 3.5-second clips initially showed body detection 100%, visible landmark fraction ~0.81–0.88, wrist confidence often low. This measures availability, not localization accuracy. Source storage is landscape with rotation; displayed video is portrait. Pose inputs preserve aspect ratio; diagnostic contact thumbnails corrected to portrait. Full 76-shot run in progress. Includes follow-through and uses sensor-assisted clip selection: cannot call this end-to-end camera-only or practice-swing analysis.

## A005 — Native ball tracking diagnostic
`scripts/ball_diagnostic.py`: native portrait crop [500,800,400,900] around sid_1_01, 106 frames. Bright-white candidate detector produced 2,876 blobs and failed to isolate flight (ball darkens against sky/background). `scripts/track_ball_demo.py`: static-background difference with manually selected first-flight point and temporal gating yielded 31 candidate positions across 0.179 s (44.039–44.218 playback seconds). Track visually follows flight, but frame 71 has a tiny 5-pixel component and apparent localization error near background crossing. This is a promising manually initialized demonstration, not validated automatic tracking or physical speed. Saved CSV, trajectory JPG, slowed overlay MP4 and full provenance. No calibration-derived outputs.

## A006 — Full pose batch and initial benchmarks
76 clips / 7,980 pose frames executed; all 76 three-frame contact sheets visually reviewed. Accepted physical identities using user order + stable gyro counts + visible swing. Original outcome review notes retained. Exact impacts remain unverified. Pose detection 100%; mean visible fraction 0.843; mean both-wrists-valid fraction 0.189. Excluded wrist pose predictors. `annotations/shot_review.csv` records review and source hashes.
`extract_features.py`: 76 watch windows, 68 club windows; 72 labeled rows / 65 matched labeled rows. All 68 club windows have acceleration limit flags (median 7.96% of samples); acceleration excluded from club predictors. Saved sensor window quality/features.
`benchmark.py`: 400 fixed-model comparisons, 2,000 external-fold fits, four targets, two protocols, five models, all seven viable feature sets on matched cohort. All OOF predictions/config/splits saved. Known-session video SVR carry MAE 25.451 yd; leave-golfer-out video Extra Trees 28.907 yd (retrospective best, selection optimism applies).

## A007 — Nested carry and uncertainty
`nested_carry.py`: 20 nested comparisons, three inner folds selecting mean/ridge10/Extra Trees/SVR without outer-fold access. Primary n=72 carry MAE: video 25.451 known-session / 31.925 held-out golfer; video+watch 26.599 / 29.869. Matched n=65 all-sensor held-out-golfer MAE 28.577. `analyze_benchmarks.py`: 360 learning-curve fits plus paired cluster bootstrap. Known-session ridge video learning curves plateau ~28.5 yd at 40–50 training shots; golfer-held-out variability large. Seven scientific-integrity/timing tests passed. Three plots visually checked. See reports/model_results.md; no trustworthy accuracy claim for physical clubhead speed, spin or practice swings.

## A008 — Head and shaft diagnostics
Saved smoothed/confidence-masked 2D face-landmark centroid trace for sid_1_01; no gaze or calibrated height inference. Hough shaft baseline yielded 106/106 misleading background proposals; rejected. Motion-gated Hough yields nine visually plausible partial shaft segments around 44.006–44.060 playback seconds. Endpoints are not verified clubhead/grip. OpenCV 5 changed line array shape; normalized to reshape(-1,4). Both baseline and improved diagnostics preserved.

## A009 — Automatic ball seed/departure trial
`track_balls.py --representatives` v1: first shot of each session; seeds found in all six, three candidate tracks (Sid 54 points, Rohan 1 63, Lucas 45), failures in Rohan 2/Andrew/Eduardo due streak-shaped blur and club confusion. Visually inspected all six montages. Lucas initial point may be club rather than ball: do not treat automatic count as validation. Preserved v1 outputs/JSON under ball_batch_v1 and ball_v1. Version 2 relaxes streak geometry and tightens initial seed proximity; currently running. No ball features have entered benchmark predictors.

## A010 — Full ball batch and camera-only event evaluation
Automatic ball v3 produced four of six representative proposals; v4 full batch produced 36/76 internally gated proposals. Preserved earlier failures and all review flags. No ball inputs entered ML.
Executed camera_events.py across Sid's entire session at 20 fps: 29 disappearance candidates, 18 near distinct reference swings. Executed filter_camera_events.py on all 29 candidates without watch inputs: retained 18 proposals, 15 near distinct reference swings (15/18 coverage), three extras and three misses. Comparison tolerance ±1.5 s against watch/metadata neighborhoods is not exact impact ground truth. Exploratory scene-tuned evaluation, no held-out detection validation.
Visually inspected camera events 05/12/19/27. Events 05 and 19 plausibly follow flight; 12 misses visible ball flight; 27 tracks a leg and stationary background. Preserve failures; trajectory gating is insufficient. Compact results copied into reports/camera_events/ for Git; cached images/native tracks remain local.
All 79 original files were independently rehashed unchanged; checkpoint verification passed. Seven unit tests and dependency consistency check passed. Local repository initialized on main; raw data, generated media, caches, models and environment excluded.
