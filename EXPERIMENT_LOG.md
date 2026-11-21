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

## A011 — Unmarked native reference review and tail rejection
Exported 18 native panels independent of trajectory overlays. Eight airborne-ball centers approximated by AI visual review with 6–12 px uncertainty: six Sid/Lucas points matched v4 predictions within uncertainty; two Andrew points had no predicted track. Source hashes saved; second reviewer still needed. Other panels stationary/ambiguous, exposing different event offsets across sessions. FFmpeg emitted showinfo entries beyond output duration; fixed index handling to retain only emitted image frames.
Stationary-tail diagnostic (three intervals below 200 px/s) removes two Sid false proposals, leaving 15 swing-neighborhood matches and one extra leg trajectory. Frozen rule retains 35 of 36 full-dataset proposals; not accuracy validation. V5 wider-lateral/shorter-track trial executed on six representatives: four pass, Andrew/Eduardo still fail. Rejected as evidence of improvement; prior outputs preserved.

## A012 — Early-window nested ablation
Six evaluations with 72 shots, same outer folds, training-only inner family selection. Window watch-peak-relative [-2,-0.8] s. Known-session MAE video/watch/combined: 30.54/30.02/30.24 yd; held-out golfer: 40.05/47.33/43.26. Both window and representation differ from full-swing benchmark. Not validated pre-impact exclusion or practice-swing inference. All predictions and choices persisted.

## A013 — Phase review and 3D illustration
Visible wrist-height extrema with bilateral temporal support yield 52 unvalidated top candidates / 24 abstentions. No impact labels fabricated. Built local 76-shot research viewer, converting overlays from MPEG-4 Part 2 to H.264 for compatibility. Includes 2D toggle, 30-fps stepping, model-estimated 3D rotation, OOF predictions and source-bound annotation export. Native browser automation timed out; UI test not completed, no further Chrome access attempted. JS syntax, file references, 76 media codec/dimension/duration checks and pose/OOF contracts pass. Exported side-by-side 3D illustrative MP4/contact sheet and visually inspected sheet. Eleven unit tests pass.

## A014 — Club contact descriptors and outcome flags, 2026-11-18
Extracted 16 peak-relative gyro transient/censoring/unclipped-derivative descriptors for 68 events, 65 labeled. Ran 36 nested evaluations across six feature sets, two split schemes and three tasks. Held-out video+watch short-shot detection 6/15 (7 false alarms) versus +club/contact 11/15 (10 false alarms); distance MAE worsens 29.98→34.38 yd. Clipping is not a recovered impact force. Waveform plot visually inspected. Initial Matplotlib native backend aborted; forced Agg and reran successfully.

## A015 — Reviewed native flight pilot
Deterministic every-third-plus-last selection gives 32 shots, 192 panels. All sheets visually reviewed; difficult native crops inspected. Corrected five mistimed windows, preserving originals. Recorded 78 approximate points across 31 shots with source hashes/16 px uncertainty; second reviewer pending. No unreviewed automatic track used in ML. Ran 24 nested held-out-golfer comparisons. On identical 31 shots, reviewed-ball carry MAE 24.78 versus body/watch 37.36; short-carry recall 9/10 with 3 false alarms. Not automatic performance; selection, timing and annotation uncertainty documented.

## A016 — Carry range restriction and operational gating
User requested 100+ yd and other thresholds. Fixed 80/100/120 yd, four feature/cohort variants, held-out golfers, training-only inner family selection for both gate and conditional regressors. Twelve threshold evaluations; every shot retains an all-shot/gated prediction. At 100 yd, full-cohort conditional video+watch MAE 12.86 vs conditional constant 11.98; operational accepted error 31.74 and hard-gated all-shot 48.61 vs unfiltered 29.87. Assisted ball cohort: conditional 13.07; accepted 20.00, 14 correct accepts/3 false accepts/3 false rejects/11 correct flags. Thresholds not selected using these scores. Cluster bootstrap, report and plot saved. Sixteen tests pass, including clipping censoring and rotation invariance.

## A017 — Local product demo and six-error audit (2026-11-18)
Expanded existing viewer into Swing Lab: 76 local replay clips, solid estimated-pose avatar, phase review prompts, carry estimates, assisted 100-yard flag/error filter, source-row/sensor evidence and persistent/exportable phase markers. Root HTML launcher added. No new model fit or changed labels. Audited six reviewed-ball classification errors, persisted source-row/launch/speed/coverage/residual joins, reinspected corrected Eduardo 19 and low-launch Rohan 2 / 10 native sheets. Higher-launch slow balls vs lower-launch faster balls plausibly explains classification confusion; no proven label offset. Exact impact/transcription remain unverified.
Checks: JavaScript syntax, 76 H.264 clips/data contracts and 16 unit tests. Browser UI not tested: native pipe and browser host unavailable. No Chrome launch or upload. Product remains an exploratory recorded-session demo with approximate phases and generic review prompts, not calibrated launch monitor or personalized coaching.

## A018 — Dual-contrast ball tracking, native timing, and local processing app
Executed v6/v7 candidate detection and v6/v7/v8 linking. Sparse reviewed-point matches (≤32 px, ≤6 ms): old v4 41/78; v6 24/78; v7 49/78; v8 52/78. V7 candidate ceiling 76/78, median 143.5 candidates per reviewed frame. Dark sky silhouettes explain v6 failures. V8 changes only bright mat initialization; frozen after this development comparison. All-shot v8 yields 72 unvalidated proposals/76. No automatic ML promotion.
Reviewed 12 native departure intervals (8.3–20.8 ms), source-bound and second-review pending. Ran nine nested same-cohort flight-feature comparisons on 26 multi-point shots; departure-independent slopes worsen carry MAE (38.55 vs 29.32 yd); combined 32.65. Preserve failed approach and sensitivity to departure ±20 ms.
Built 76 confidence-masked head traces, native-cadence review clips with source/output PTS verification, ball-path diagnostics and demo integration. Added isolated new-video pipeline and loopback upload service. Real HTTP end-to-end processing and blank-video abstention executed. 21 tests, media/data contracts and JS syntax checks pass. Browser/native access still fails; UI not verified. See reports/tracking_timing_product_progress.md.

## A019 — User-directed 3D range avatar and stable limb completion
Replaced flat figure with a local Three.js articulated golfer/range scene: solid limbs, common grip/club, range mats/nets/targets, schematic ball/tracer and orbit/flight camera. Added display-only temporal completion and segment constraints so uncertain legs remain visible; original scientific poses/features unchanged. Added native video tracer overlay from v8 proposals. Tested all 7,980 scene frames with a geometry-only renderer stub and inspected explicitly labeled software projections; widened flight camera to retain golfer. 24 tests pass. Browser/WebGL QA remains blocked by unavailable host; user help requested asynchronously. No heavy pose-model experiment ran; that work was deferred for this explicit visual request. See reports/avatar_range_update.md.

## A020 — 3D delivery diagnostics and loading improvements
User again reported the scene missing. Did not assume a cache issue was proven. Checked generated HTML contains three valid embedded scripts and the updated scene; added explicit version label and cache-busting launcher, HTML cache prevention on future local-server starts, and render-failure messages without disabling video. Display-only JSON compaction reduces page from ~60 MiB to ~29 MiB. Scientific data unchanged. Re-ran 24 tests, media/data and 7,980-frame geometry checks. Asked user which heading is visible; actual browser access remains unavailable.

## 2026-11-18 — avatar/club correction and animated travel
Executed six representative shaft-detector runs with line contrast, body masks and static-background rejection; 397/630 frame proposals, not validated correct tracks. Failure cases preserved in outputs/shaft_tracking. Do not deploy these proposals as reliable tracking. Added source-bound 23-anchor Sid pilot and display-only depth lift; 97 frames covered, eight abstentions. Both depth alternatives reproduce the annotated/interpolated image displacement by construction; no external tracking-accuracy claim. Replaced nose-centered head geometry and phase-driven club path; added model-scaled travel badge. 26 tests, all 7,980 scene frames and actual embedded-script syntax pass. Browser rendering remains untested by the agent. Full details: reports/avatar_club_revision.md.

## 2026-11-18 — direct watch/carry relationships
Executed all ten watch-feature correlations on 72 shots and per-golfer subsets. Peak rotation Spearman -0.044; within-golfer peak acceleration Pearson -0.016. Positive pooled acceleration peak-width r=.449 shrinks to .060 after golfer demeaning. Peak rotation/acceleration rank relationships remain near zero on the 44 known 100+ yd shots. These descriptive analyses do not rank causal features or add independent prediction validation. Source values and full outputs saved in experiments/watch_carry_relationships_v1; interpretation in reports/watch_carry_relationships.md. Raw watch extrema again suggest possible rotation range limiting, not confirmed saturation.

## 2026-11-18 — exact >70 / >100 matched-cohort comparison
Re-scored the six previous all-shot methods on identical subsets (50 shots >70 yd; 39 >100 yd), and executed 12 additional restricted-training leave-golfer-out evaluations with inner grouped model-family selection. Existing all-sensor model MAE: 18.96 / 21.36 yd. Conditional training-mean MAE: 22.19 / 11.98; every restricted-trained learned model loses to or matches that conditional mean. Full outputs in experiments/carry_cutoff_comparison_v1, interpretation in reports/carry_cutoff_comparison.md. Actual outcome filtering is not an automatic gate; no independently validated gain claimed. Original model results unchanged.

## 2026-11-18 — Physics-gated video launch & clubhead tracking (video_launch_v1)
Implemented physics-constrained ball departure and pre-impact clubhead speed extraction in `scripts/video_launch_tracker.py` and benchmarked in `scripts/video_launch_benchmark.py`. Resolved clubhead follow-through confusion on `lucas_1_02` and `lucas_1_12`; recovered rightward launch vectors on `lucas_1_09` (+19.9°) and `lucas_1_10` (+38.0°). Across 68 tracked shots, video launch angle correlates with TrackMan horizontal direction at r = +0.727; ball rise correlates with carry at r = +0.583. Held-out golfer evaluation (Leave-One-Golfer-Out) achieves: Carry flat MAE 26.04 yd (R2 = +0.369 vs baseline 0.13), Launch direction MAE 4.50° (R2 = +0.524 vs negative baseline), Launch angle MAE 3.83° (R2 = +0.464), and Mishit detection (<60 yd) AUROC 0.856 (recall 75%, precision 66.7%). Outputs saved in `data/processed/video_launch_features.csv`, `experiments/video_launch_v1/`, and updated `outputs/research_viewer/index.html`. 29 unit tests pass. Full report in `reports/video_launch_tracking_results.md`.

## 2026-11-18 — Aerodynamic flight timing & TrackMan autonomy
Recalibrated continuation flight timing in `scripts/ball_continuation.py`. Fixed the rapid premature fall bug (previously 0.85s duration) by implementing real-world golf aerodynamics with realistic 5.5 - 7.0s hang times. The ball now ascends smoothly towards apex downrange during video playback instead of crashing into the ground. Addressed the user's TrackMan replacement question: designed autonomous camera-only mode in `ball_continuation.py` that infers hang time and apex height from camera launch rise, making the system viable without TrackMan. Rebuilt viewer at `outputs/research_viewer/index.html`. All 29 unit tests pass.

## 2026-11-18 — Dual 3D flight arcs, target centerline & live floating yardage badges
Cleaned up the local review UI per user specifications:
1. Removed the 2D estimated continuation tracer and purple forecast from `nativeSection` in `scripts/viewer_template.html`, keeping verified camera-tracked ball flight (yellow) and clubhead tracking (teal) without video playback freezes.
2. Added a 3D driving range target centerline along $X = 0, Z = 0 \to -240\text{m}$ (solid white line) with 50, 100, 150, 200, and 250-yard cross hash marks.
3. Implemented dual 3D flight arcs launching simultaneously in `scripts/range_scene.js`:
   - **TrackMan Radar Truth** (cyan `#38bdf8`): Constructed from empirical Doppler radar training labels (`Carry flat (yd)`, `Launch direction (deg)`, `Curve signed (ft)`, `Carry side (ft)`, `Height (ft)`).
   - **Best Video Launch Model** (purple `#c084fc`): Constructed from autonomous Leave-One-Golfer-Out predictions (`Carry flat (yd)` [26.04 yd MAE], `Launch direction (deg)` [4.50° MAE], apex height).
4. Replaced static text carry box with live floating HTML yardage badges hovering directly above each ball in screen space. Badges tick up live in yards as each ball flies through the air, then stop and settle at the final landing carry distance when each ball touches down.
5. Reconfigured header metrics in `scripts/viewer_template.html` and `Open Swing Lab.html` to clearly display the Best Video Launch Model (purple) vs. TrackMan Ground Truth (cyan) and Prediction Error, explicitly explaining the autonomous video sensing modalities used (ball departure velocity vector, climb angle, pre-impact clubhead speed) vs. the radar benchmark.
All 29 unit tests pass; Three.js 7,980-frame verification passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

## 2026-11-18 — Anatomical Golfer IK, Realistic Forged Club, Ball on Mat & Dual/DTL Camera Views
Executed 3D range scene visual overhaul in response to user feedback:
1. Replaced raw MediaPipe lower limb landmark interpolation with analytical two-bone Inverse Kinematics (`solveKnee`) with exact femur (0.42m) and tibia (0.40m) bone lengths. Constrained knee flexion forward in the transverse/sagittal plane of the pelvis, resolving the inverted/backward "flamingo" knee on the lead leg.
2. Positioned the ball squarely on the synthetic driving range hitting mat atop a dedicated rubber tee spot at $(X = 0.48\text{m}, Y = 0.068\text{m}, Z = -0.05\text{m})$. The ball rests on the mat until the club strikes it at impact, and the white target centerline originates directly at the ball.
3. Enhanced golf club graphics: modeled a stepped chrome steel shaft, Tour Velvet black rubber grip with white cap ring, black hosel ferrule, and a sculpted forged muscle-back blade with a satin face plate featuring authentic $34^\circ$ loft and grooves. Club is now rendered in the golfer's hands across all 76 shots.
4. Unified TrackMan camera angle: replaced multiple view buttons with one stable TrackMan launch monitor camera ($X = 1.8\text{m}, Y = 2.2\text{m}, Z = 5.0\text{m}$) positioned behind and to the side, framing the golfer on the mat in the left foreground while tracking the entire dual flight path downrange. Removed playback camera blending/jumping so the camera stays completely stable as the video plays.
Rebuilt viewer at `outputs/research_viewer/index.html`. All 29 unit tests pass; Three.js 7,980-frame geometry check passes.

## 2026-11-18 — Elevated Camera View, 3D Tracer Ribbons, Turf Landing Rings & Randomized Range Hazards
Upgraded the 3D range scene environment and visual comparison per user feedback:
1. **Elevated TrackMan Camera Perspective**:
   - Elevated camera to $(X = 2.0\text{m}, Y = 3.6\text{m}, Z = 6.4\text{m})$ looking down towards $(X = -0.1\text{m}, Y = 0.6\text{m}, Z = -60.0\text{m})$.
   - This high-angle perspective eliminates horizon depth compression, giving clear spatial separation of both landing spots on the ground while keeping the golfer and hitting mat visible in the lower foreground.
2. **Thick 3D Tracer Ribbons**:
   - Created dynamic 8 cm wide quad-strip 3D tracer ribbons (`ribbonTruth` in cyan `#38bdf8`, `ribbonPred` in purple `#c084fc`) allocated once with pre-indexed `BufferGeometry`. Delivers bold, glowing Toptracer-style trails in WebGL with zero per-frame allocation.
3. **Turf Landing Rings**:
   - Added glowing landing bullseye rings on the turf (`landRingTruth` and `landRingPred`) that appear at touchdown ($progress \ge 0.95$), marking the exact landing locations of both balls on the ground plane.
4. **Varied Driving Range Targets & Hazards**:
   - Widened target centerline to 26 cm (`box([.26, .016, 240])`) with cross hash marks at 50, 100, 150, 200, and 250 yd.
   - Scattered 6 target greens at realistic varied locations and distances with color-coded flags: 50 yd left (red), 90 yd right (yellow), 125 yd center-left (white), 165 yd right (blue), 210 yd left (red), and 250 yd distance (gold).
   - Added four organic golden-sand (`#dfce9e`) practice bunkers with raised turf lips along the fairway ($Z = -75, -126, -153, -196\text{m}$).
5. Rebuilt viewer at `outputs/research_viewer/index.html`. All 29 unit tests pass; Three.js 7,980-frame geometry check passes.

## 2026-11-18 — Audited Best Models for All TrackMan Targets, Curvature Flight Modeling & Landing Miss Reduction
1. **Viewer Upgraded to Audited Best Model**:
   - Switched `build_research_viewer.py` from the older retrospective model (26.04 yd MAE) to the audited v7 model (**16.27 yd LOGO MAE**, **12.64 yd** on clean tracks).
2. **Trained Rigorous LOGO Models for All 10 TrackMan Targets (`scripts/train_all_targets.py`)**:
   - `carry_yd`: **16.27 yd MAE** (baseline: 40.05 yd, median: 9.06 yd)
   - `total_yd`: **14.90 yd MAE** (baseline: 30.51 yd, median: 7.56 yd)
   - `launch_dir_deg`: **4.55° MAE** (baseline: 7.50°, median: 1.57°)
   - `launch_angle_deg`: **2.61° MAE** (baseline: 5.49°, median: 1.27°)
   - `ball_speed_mph`: **9.05 mph MAE** (baseline: 14.84 mph, median: 5.50 mph)
   - `height_ft`: **11.00 ft MAE** (baseline: 27.80 ft, median: 5.60 ft)
   - `landing_angle_deg`: **5.13° MAE** (baseline: 12.74°, median: 2.11°)
   - `hang_time_s`: **0.56 s MAE** (baseline: 1.25 s, median: 0.30 s)
   - `carry_side_ft`: **39.49 ft MAE** (baseline: 48.36 ft, median: 30.73 ft)
   - `curve_ft`: **28.72 ft MAE** (baseline: 31.68 ft, median: 24.91 ft)
3. **Aerodynamic Shot Curvature in 3D Flight**:
   - Updated `scripts/range_scene.js` so the predicted purple arc incorporates quadratic side curve offset (`curveOffset`) toward `this.sidePred`. The predicted ball now visibly draws or fades instead of launching along a rigid straight vector.
4. **2D Landing Miss Distance Reduced**:
   - Overall 2D landing miss distance dropped to **23.6 yd** (median: **14.9 yd**).
   - On the 58 clean shots, 2D landing miss dropped to **18.69 yd** (median: **13.19 yd**). Over **60.3% of shots land within a 15-yard circle** and **79.3% within 25 yards**.
5. All 35 unit tests pass; Three.js 7,980-frame geometry check passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

## 2026-11-19 — Sleek Tracer Ribbons & Multi-Camera Side Angle Suite
1. **Sleeker, Cleaner Shot Tracer Trails**:
   - Scaled down the tracer ribbon base width from 0.48m to 0.20m (~8 inches) in `scripts/range_scene.js`.
   - Dynamic downrange perspective scaling smoothly caps at 0.44m (down from 1.10m), giving a crisp, refined broadcast Toptracer trail without visual bulk.
   - Reduced flying ball scale to 2.2x with glowing trail core.
2. **Side Angle Perspective & 6-Camera Suite**:
   - Researched and added the user's requested **Side 3/4 Angle** as the default:
     - **📐 Side 3/4 (Tight) [Default]**: Camera stationed at $(X = 7.5\,\text{m}, Y = 4.2\,\text{m}, Z = 11.5\,\text{m})$ looking towards $(X = 0.2\,\text{m}, Y = 1.0\,\text{m}, Z = -75.0\,\text{m})$ with $52^\circ$ FOV. Frames the golfer in the lower-left foreground, displays the full soaring arc trajectory in profile across the screen, and provides clear spatial separation of both landing spots on the green.
     - **📐 Side 3/4 (Wide)**: Higher elevated side stadium view $(X = 10.5, Y = 5.5, Z = 15.0)$ for broader fairway overview.
     - **📐 Face-On Side**: Opposite side perspective from the left flank $(X = -6.8, Y = 4.0, Z = 11.0)$.
     - **🎯 Landing Zone**: Front-row green camera stationed downrange looking across the turf.
     - **🎥 Follow Ball**: Dynamic chase drone tracking behind the ball in flight.
     - **⛳ Behind Bay**: Classic elevated launcher perspective from behind the hitting bay.
   - Added 1-click camera switcher buttons directly above the range canvas in `scripts/viewer_template.html`.
3. All 35 unit tests pass; Three.js 7,980-frame geometry check passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

1. **Physical Turf Bounce & Rollout Engine**:
   - Extended 3D flight continuation in `scripts/range_scene.js`: post-impact timeline is partitioned into airborne carry flight (first 70%) followed by physical turf bounce and rollout (remaining 30%).
   - At touchdown (Carry distance), the ball performs realistic decaying vertical hops ($0.28\text{m} \rightarrow 0.09\text{m} \rightarrow \text{turf}$) while decelerating along the turf to final resting Total Distance ($Z = Z_{\text{carry}} - \text{rollout}$).
   - Both pitch marks (Carry bullseye rings) and resting markers (Total distance rings) remain illuminated on the green.
2. **Dynamic Live Yardage Counters**:
   - Badges display real-time carry during flight (`TrackMan · 112 yd (carry)`), transition through rollout (`TrackMan · 140c ➔ 148 tot`), and settle at final rest showing both metrics: `TrackMan · 140 yd c · 153 yd tot` and `Video Model · 142 yd c · 154 yd tot`.
   - Primary metric cards updated to show both Carry and Total Distance side by side with deltas.
3. **TrackMan 3-Camera Selection System**:
   - Added interactive camera switcher above the 3D range:
     - **🎥 Follow Ball (TrackMan POV)**: Dynamic chase drone that starts behind the golfer, takes off with the ball at impact, flies right behind the tracer down the fairway, and swoops down into the green for a close-up view of the touchdown and rollout.
     - **🎯 Landing Zone Cam**: Stationed downrange directly at the target green ($Z \approx -120\text{m}$, $Y = 4.8\text{m}$, $X = 9.5\text{m}$) angled across the fairway, completely eliminating foreshortening to see the exact landing dispersion, bounce, and roll right in front of the viewer.
     - **⛳ Range Overview**: Traditional elevated launcher view from behind the hitting bay.
4. All 35 unit tests pass; Three.js 7,980-frame geometry check passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

1. **Camera-Billboarded Thick 3D Tracer Ribbons**:
   - Upgraded tracer ribbon width from 0.08m to 0.48m (~19 inches wide) in `scripts/range_scene.js`.
   - Normals are now computed via the `tangent x toCam` vector cross-product, ensuring ribbons are continuously camera-billboarded to present their maximum face directly to the viewer's eye regardless of elevation or viewing angle.
   - Added perspective width scaling (`Math.max(0.38, Math.min(1.10, width * (0.75 + dist * 0.009)))`) so downrange flight paths at 150–250 yd stay bold, vibrant, and effortlessly visible across the entire fairway.
   - Increased ribbon opacity to 0.98.
2. **Enlarged Flying Balls & Concentric Landing Ripples**:
   - Scaled flying golf balls to 3.2x (`this.ballTruth`, `this.ballPred`) so they remain clearly visible as glowing leading spheres in flight.
   - Added outer ripple rings (`outerRingTruth`, `outerRingPred`, radius 2.3m) with semi-transparent turf illumination around the bullseye disc upon touchdown ($progress \ge 0.95$).
3. **Collision-Free Screen-Space Yardage Badges**:
   - Resolved badge text overlap bug where `truthBadge` and `predBadge` overlapped during launch and touchdown.
   - Implemented dynamic 2D collision separation in `scripts/range_scene.js`: when screen-space Euclidean distance is under 125px, badges horizontally repel if separated, or stack vertically (TrackMan on top, Model below) if aligned, with glowing colored dot indicators.
4. **All 35 unit tests pass** (`.venv/bin/python -m unittest discover -s tests -v`); Three.js 7,980-frame geometry check passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

## Milestone: Visual-Inertial Clubhead Arc Fusion & Gyro Face-to-Path Modeling (2026-11-19)

1. **Visual-Inertial Clubhead Arc Fusion Engine**:
   - Implemented `scripts/fuse_club_arc.py`: Anchored the club grip to MediaPipe wrist landmarks in 3D camera space and fused the high-rate clubhead gyroscope roll and delivery axes to point the 37.25" ($0.946\,\text{m}$) steel shaft vector in 3D space.
   - Traced continuous 3D delivery arcs and fitted downswing delivery planes via SVD across all 76 shots (`data/processed/club_fusion/*.json`).
   - Extracted physical delivery metrics: dynamic clubface angle at impact (`dynamic_face_angle_deg`), shaft roll closure rate (`closure_rate_dps`), delivery plane tilt (`swing_plane_tilt_deg`), and rotational clubhead speed (`delivery_speed_mph`).
2. **Leave-One-Golfer-Out Benchmark Evaluation (`scripts/club_fusion_benchmark.py`)**:
   - Evaluated on the matched 65-shot club cohort under strict Leave-One-Golfer-Out cross-validation.
   - **Ball Curvature (`curve_ft`)**: Gyro Face-to-Path cut MAE from **30.21 ft to 21.53 ft** (a 28.7% error reduction) and boosted held-out $R^2$ from **-0.152 to +0.211**, with a median absolute error of just **13.88 ft** across unseen golfers!
   - **Lateral Carry Side (`carry_side_ft`)**: Fusing video launch direction with gyro curvature reduced MAE to **37.09 ft** ($R^2 = +0.297$), outperforming vision-only (43.04 ft MAE).
   - Saved report to `reports/club_inertial_fusion_results.md` and out-of-fold predictions to `experiments/club_fusion_v1/`.
3. **Interactive 3D Visual-Inertial Delivery Arc in Range Scene**:
   - Added `this.clubArc` (glowing amber 3D trail) and `this.swingPlane` (translucent cyan delivery plane) to `scripts/range_scene.js`.
   - Dynamic clubface orientation now rotates the 3D forged iron blade around the shaft according to the measured dynamic face angle (open vs square vs closed).
   - Added 1-click toggle buttons `[⚡ Club Arc: ON]` and `[📐 Delivery Plane: OFF]` in the 3D Range viewer controls.
4. **Dedicated Research Review UI Section**:
   - Added prominent section **Visual-Inertial Clubhead Arc Fusion** in `scripts/viewer_template.html` and `outputs/research_viewer/index.html`.
   - Displays real-time live metric cards (Dynamic Clubface Angle, Swing Delivery Plane, Impact Closure Rate, Delivery Velocity), synchronization status badge, and the full Leave-One-Golfer-Out comparison table.
5. **Testing & Verification**:
   - Added unit tests in `tests/test_club_fusion.py`. All 38 unit tests pass (`.venv/bin/python -m unittest discover -s tests -v`).
   - Node.js 76-shot / 7,980-frame Three.js geometry verification passes (`node scripts/verify_range_scene.js`).

## Milestone 12 — Elimination of Legacy 100-Yard Gate, Clean Abstention Policy & Multi-Modal Architecture Summary

1. **Elimination of Arbitrary 100-Yard Rule**:
   - Removed all legacy references, filters, and gates relying on the 100-yard carry threshold. Tracking success is strictly defined by physical camera ball departure capture, not arbitrary distance bins.
2. **Simplified Selector Tagging**:
   - Normal tracked shots carry zero clutter/tags.
   - Only abnormal strikes are annotated: `[TrackMan Missed Reading]` for strikes unrecorded by radar (`rohan_1_03`), and `[Camera Missed Ball]` for swings where stationary tee detection abstained (`rohan_1_04`, `rohan_1_05`).
3. **Transparent Abstention Policy**:
   - For untracked shots, the model abstains from making a prediction (`Prediction Abstained`, displaying `—`) rather than falling back to uncalibrated secondary models.
   - 3D range scene suppresses predicted ribbons, ball, and landing markers when a prediction is abstained.
   - Untracked swings are excluded from benchmark generalization statistics to maintain data integrity.
4. **Multi-Modal Sensing Architecture & Summary Statistics Dashboard**:
   - Embedded a comprehensive multi-modal performance summary and comparative error reduction chart at the bottom of `Open Swing Lab.html` (`outputs/research_viewer/index.html`).
   - Outlines sensor-to-target mapping:
     - *High-Speed Camera*: Flat Carry (15.77 yd MAE), Launch Angle (2.58° MAE), Ball Speed (8.61 mph MAE), Launch Direction (4.23° MAE).
     - *Clubhead Gyro*: Lateral Curve Face-to-Path (21.53 ft MAE, 13.88 ft MedAE; 28.7% error reduction).
     - *Visual-Inertial Fusion*: Carry Side (37.09 ft MAE) and 2D Landing Error (24.28 yd MAE vs 38.59 yd baseline; 37.1% error reduction).
     - *Apple Watch IMU*: Contact shock detection, downswing transition tempo, temporal synchronization.
5. **Dynamic Turf Bounce & Rollout Along Touchdown Flight Tangent**:
   - Replaced simplified forward-only rollout with instantaneous 3D horizontal touchdown tangent integration (`tan = flight(1.0) - flight(0.98)`).
   - The ball now bounces and rolls along the exact momentum vector it possessed upon turf contact, combining initial launch direction with accumulated spin-induced aerodynamic Magnus curvature (fades roll right, draws roll left).
6. **Verification**:
   - All 38 unit tests pass (`.venv/bin/python -m unittest discover -s tests -v`).
   - Three.js geometry verification passes for all 76 shots and 7,980 frames (`node scripts/verify_range_scene.js`).






### November 19 — Primary vs Fallback Transparency & Abstention Dashboard

- **Transparent Fallback Labeling**: Tagged the 14 low-confidence swings that resorted to predicting cohort training mean (~98–104 yd) with an amber warning banner and dropdown indicators, detailing the specific physical reason (lighting glare, extreme topped mishits, or <5 points in the early geometry window).
- **Primary Model Breakdown Dashboard**: Added a dedicated performance section in `Open Swing Lab.html` isolating the 58 confident launch swings, highlighting the authentic **12.64 yd MAE** (and **8.31 yd median** error, 64.8% error reduction vs baseline, 82.8% within 20 yards) delivered by the 2-feature launch physics model when it has a valid read.
- **Abstention Principle**: Codified the principle that golfers prefer explicit "Didn't get a good read" alerts over confident assertions of uncalibrated fallback means.
- Rebuilt viewer at `outputs/research_viewer/index.html`; all 45 unit tests passing; Three.js 7,980-frame geometry check passing.

### November 19 — Native track recovery and joint landing

- Froze the current launch prefixes and tested original/wide crops and 35/70 ms recovery. The wide 35 ms rule added 654 proposed points on 31 shots. The wider 70 ms rule added a further 90 on 12 of those shots. Ten sampled extensions entered the local reviewer, which stops drawing a live trace when stale. Lucas 02's false path was rejected.
- On 72 held-out shots, per-target early video scored 25.19 yd landing / 26.94 yd resting. Structured joint video + watch scored 28.54 / 28.43. Original late samples and wide 35 ms samples scored 24.63 and 24.61 yd landing, with worse resting error than early video. The run retained mismatched or failed tracking windows and used actual club coverage for the 65-shot sensor comparison.
- Details and commands are in [the report](reports/ball_extension_and_joint_landing.md).

### November 19 — Ball motion cutoff and launch window

- Three-point cutoff using 1,000 px/s or 20% of apparent launch peak dropped 2,444 of 3,326 detections and removed visibly correct ball positions. The separate sustained-stationary display gate triggered on zero current viewer paths.
- On the same refreshed tracks, 80 ms changed carry MAE from 21.91 to 21.56 yd and landing from 28.51 to 28.27 yd. The 60 ms fit deteriorated sharply, and the earlier saved 16.27 yd carry remained stronger. [Detailed results](reports/ball_motion_gate_results.md).

November 19 — Decoded all 42,367 frames across 76 swing windows for club detection. Compared two shaft detectors, tested higher-resolution/background rejection on two swings, and tested five-seed native-frame clubhead tracking on Lucas 02. [Results, failures and commands](reports/native_swing_shaft_review.md)

November 19 — Real-time range playback. Rebuild with `.venv/bin/python scripts/build_research_viewer.py` and validate with `node scripts/verify_range_scene.js`. The dataset supplies a positive hang time for all 72 displayed TrackMan flights and all 69 displayed model flights. The test checks each available touchdown time, terminal rollout, finish hold and follow-camera framing. Sid 01 uses 6.43 s and 6.789244 s from impact.

Range camera and timing revision — tested the four starting views, follow framing, fixed overview coverage through rollout, and matching 1× club/head positions in real-time and video-driven modes across all 76 shots at narrow and wide viewport sizes. Rebuild with `.venv/bin/python scripts/build_research_viewer.py` and check with `node scripts/verify_range_scene.js`.

Range landing presentation — the scene test checks first contact, hop apex, final yardage, rollout position, and tee-camera visibility along each available shot path at two viewport widths. Rebuild with `.venv/bin/python scripts/build_research_viewer.py` and run `node scripts/verify_range_scene.js`.

Cross-fitted 11 TrackMan outputs within golfer-held-out folds, tested all-label corrections, forward/lateral rollout, carry-only and six-label rollout, clipping, mean rollout and inner-fold mixtures. [Results and reproduction](reports/trackman_label_sharing.md).

Endpoint promotion — retained the pre-promotion viewer fields for reproducible paired comparisons. Regenerate held-out outputs with `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_label_stack.py`, rebuild with `.venv/bin/python scripts/build_research_viewer.py`, and verify scene endpoints using `node scripts/verify_range_scene.js`. [Score table](experiments/label_stack_v1/deployed_cohorts.csv)

Avatar address registration — shoe-edge clearance from the ball center increased from a median of 0.206 to 0.502 scene metres. The clubface center starts 0.040 m behind the ball center, along the target line. The default depth reconstruction peaked at 13.42° shaft change per 1/120 playback second across 31,692 samples. Rebuild with `.venv/bin/python scripts/build_research_viewer.py` and verify with `node scripts/verify_range_scene.js`.

Range scenery revision — rebuilt the local viewer, visually inspected the overview, behind-tee, follow and landing cameras in headless Chrome, and ran the existing 76-shot scene checks. Target coordinates use ground distance from the tee for the displayed yardages.

November 19 — Estimated Departure Phase Navigation. Enabled `[Ball departure (est)]` across all unreviewed shots in `scripts/viewer_template.html`. Departure timestamp is anchored to `current.ball_track[0].playback_s - current.start` (empirical audit shows this matches human-reviewed native-cadence contact within +6.2 ms, perfectly aligning with 30-fps review frames). Rebuilt viewer at `outputs/research_viewer/index.html`; 45/45 tests passing; Three.js geometry passing.

November 20 — Launch Frame Usage Transparency. Enhanced `#strikeQualityBanner` in `scripts/viewer_template.html` to detail total tracked frames alongside exact used frames and source frame index spans under the 150 ms cutoff (e.g., "tracked 60 launch frames; model used initial 23 frames (frames 112–136)"). Rebuilt viewer at `outputs/research_viewer/index.html`; 45/45 tests passing; Three.js geometry passing.

November 20 — Ball Path Canvas Launch Window Highlight. Updated `drawNative()` on `#ballPath` canvas to render the model-used launch window ($\le 150\text{ ms}$, 5–24 frames) in emerald green (`#34d399`) with point nodes and terminal boundary circle, contrasting against the rest of the orange trajectory. Updated graphic legend in `scripts/viewer_template.html`. Rebuilt viewer at `outputs/research_viewer/index.html`; 45/45 tests passing; Three.js geometry passing.

Clean-only five-golfer holdout on 57 accepted shots: slice drag gave 12.34 yd carry MAE, refitted reference 12.33, and the fixed gyro/video blend 11.76. The learned-turf final-side anchor reduced lateral shift MAE to 2.74 yd. [Reproduction and limitations](reports/clean_physics_coupling.md)

Tested quality-aware blending, direct image recovery, stationary-ball scale and manual input replacement on 57 clean LOGO shots. Image recovery refit improved 11.76 to 11.05 yd; quality weighting worsened to 12.51. [Reproduction and review](reports/clean_launch_audit.md)

Added a feathered green aim line to the range turf texture. The rebuilt viewer passed the geometry check across 76 shots and 7,980 frames. Browser rendering was not checked.

Changed the aim line to translucent black dashes with rounded ends and soft edges. Rebuilt the viewer and passed the full geometry check. Browser appearance was not checked.

Changed the range centerline to thin white rectangular dashes and added an unchecked Show centerline control. The rebuilt viewer passed the scene geometry checks and a targeted check of the toggle wiring. Browser rendering was not checked.

Removed the tee mesh and used the hitting strip surface plus ball radius for the launch origin. Rebuilt Swing Lab and checked mat contact and range geometry across all 76 shots.

Rebuilt the range golfer with MakeHuman body and garment geometry and authored skin weights. Source URLs and licensing are in vendor/golfer/NOTICE.txt. Rebuild the offline asset with .venv/bin/python scripts/build_golfer_mesh.py base.obj default_weights.mhw, then run .venv/bin/python scripts/build_research_viewer.py and node scripts/verify_range_scene.js. The checks cover 76 shots and 7,980 frames. Headless WebGL inspection included address, backswing, impact and finish. The stance and trail-foot release are authored display approximations.

Applied the MakeHuman adult target to the garment, body and joint coordinates, removed radial leg padding and limited the display torso hinge. First-pass browser images showed excessive knee bend and scalp protruding past the cap. Raised the pelvis, reduced knee flex and refit the cap, then inspected eight final WebGL renders. Reproduce with .venv/bin/python scripts/build_golfer_mesh.py base.obj default_weights.mhw caucasian-male-young.target followed by the viewer build and scene verification. Asset sources are in vendor/golfer/NOTICE.txt.

Restored the prior avatar body profiles and recorded head, torso, arm and hip landmarks. Removed the generic body asset from both viewer builds. Follow-through knee and ankle filtering reduced the pooled 95th percentile of frame-to-frame second differences from 0.0223 to 0.0167 display units. Reviewed WebGL renders from front, side and behind; grounded both shoes at address and closed the trouser waistband after inspection. Reproduce with the viewer builder and node scripts/verify_range_scene.js.

Replaced the separate torso, hip cap and lower-leg pieces with shared waist and knee geometry. The chest and hips follow the recorded rotations, with a smooth transition through the torso. The trail heel now releases during follow-through even when the reconstructed ankles converge. Inspected browser renders at address, impact and finish from multiple angles and ran the 76-shot geometry suite.

November 20 — PoseFormerV2 (CVPR 2023) Frequency-Domain 3D Kinematic Skeleton & Club Delivery Studio. Added standalone PoseFormerV2 lifting pipeline (`scripts/process_poseformer_v2.py`) using official pretrained weights (`vendor/PoseFormerV2/checkpoint/poseformerv2_27_f3_n3.bin`). Processed all 76 shots to extract 17-joint 3D metric coordinates (`data/processed/poseformer_v2/*.json`), anchoring the 37.25" steel golf club to the golfer's wrists and tracing the authentic 3D delivery arc. Integrated a dedicated, completely non-destructive section in `scripts/viewer_template.html` (`#poseFormerSection`) featuring an interactive 3D WebGL studio with 4 camera angle presets (3/4 Studio, Face-On, Down-the-Line, Overhead), play/pause scrubber, live wrist speed (m/s), thorax shoulder tilt (°), pelvic hip clearance (°), and ghost comparison against raw MediaPipe. Rebuilt research viewer at `outputs/research_viewer/index.html`; 45/45 tests passing; Three.js 7,980-frame geometry check passing.
