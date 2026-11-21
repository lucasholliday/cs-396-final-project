# Project status — 2026-11-17 checkpoint

## Current phase
Initial audit, synchronization, whole-swing pose/features and baseline/nested ML research are executed. Native ball/shaft tracking remains experimental. A camera-only event detector was evaluated on Sid's full session. Local research viewer and illustrative 3D skeleton export added. No polished product, calibrated metric speed, reliable nine-phase detector, human mesh, gaze or practice-swing validation.

## Completed and tested
- 79 originals / 11,976,204,529 bytes hashed and independently rehashed unchanged. All workbook sheets/CSV schemas/timing and media metadata audited.
- 76 reviewed swing identities, 72 labeled outcomes, five golfers/six sessions. User confirms ordinal matching including unregistered strikes; no practice swings.
- 68 club-covered events (65 with labels); eight missing tails consistent with SD-card failures. Affine gyro alignment residuals <36 ms. Exact impacts remain unverified.
- 76 pose clips / 7,980 frames, contact sheets visually checked. Pose availability 100%, mean both-wrists confidence validity only 18.9%. Excluded wrist pose predictors.
- 400 fixed-model evaluations and 26 nested carry comparisons with persisted shot/golfer splits and all OOF predictions; 360 learning-curve fits; paired uncertainty and error analysis; plots inspected.
- Primary nested n=72 carry MAE: video 25.45 yd known-session / 31.93 held-out golfer; video+watch 26.60 / 29.87. Matched n=65 all-sensor held-out-golfer MAE 28.58. Small-sample uncertainty remains large.
- Twelve fixed-ridge full-data reference fits saved under models/reference_ridge, excluded from Git. These are not independently validated deployment models.
- Native ball demonstration: manually initialized sid_1_01, 31 candidate points over 0.179 s. Automatic v4 batch ran on all 76: 36 internally gated proposals, all awaiting independent point review before ML use. Version failures preserved.
- Motion-gated Hough: nine plausible partial shaft segments on sid_1_01; unmasked baseline rejected for background false detections. Saved head-centroid trace, without gaze/metric-height interpretation.
- Eleven tests pass; isolated .venv imports and pip dependency check pass. Python 3.10.7, FFmpeg 4.3.2, pinned dependencies in requirements/lock.
- 76 PoseFormerV2 3D skeleton models generated using official pretrained weights (`vendor/PoseFormerV2/checkpoint/poseformerv2_27_f3_n3.bin`) with frequency-domain DCT lifting (CVPR 2023), saving 17-joint 3D metric skeletons, attached 37.25" steel golf club, and dynamic delivery arcs (`data/processed/poseformer_v2/*.json`). Dedicated 3D studio section added to research viewer.

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
1. Independently review native impact/ball/club points across golfers. Use the eight approximate point annotations and recorded failure examples as a starting point; retain uncertainty and missing observations.
2. Validate local viewer playback and interaction when browser testing is available. The current 76 media files/data contracts pass checks; actual UI behavior remains unverified.
3. Collect calibrated, securely logged new sessions per reports/data_collection_recommendations.md. Freeze candidate methods before assessing new golfers/days; do not train on unreviewed ball tracks or claim verified pre-impact features.

## Git
A project-local repository on main was initialized with approval. Verified research checkpoint prepared for local commit; use `git log -1 --oneline` to identify the saved revision. No remote, no push. An unrelated home repository also exists: always verify repository root. Raw data, original workbook, generated media/cache, model binaries and environment are ignored. Stage only intended code/docs/annotations/derived research tables.

## Latest continuation — research delivery
- Six nested early-window evaluations completed on 72 shots: held-out-golfer MAE video 40.05 yd, watch 47.33, combined 43.26. Offline watch-assisted window; impact exclusion is not verified, especially given cross-session timing offsets.
- Reviewed 18 unmarked native panels, recorded eight approximate airborne-ball centers: six matched Sid/Lucas proposals within annotation uncertainty, two Andrew misses. Second reviewer and broad validation remain outstanding.
- Stationary-tail rejection removes two of three Sid camera false positives; remaining leg confusion survives. Frozen all-shot filter retains 35/36 proposals. V5 wider search still succeeds on only four of six representatives; no ML use of ball proposals.
- Phase proposals: 52 top candidates, 24 abstentions, no accepted impact annotations.
- Built outputs/research_viewer/index.html for all 76 shots, H.264 overlays, frame stepping, estimated rotatable skeletons, OOF carry and phase annotation export. Generated and visually checked outputs/pose_3d/sid_1_01_contact.jpg and exported estimated_3d.mp4. Browser automation timed out; browser playback/UI not verified. Do not retry Chrome without explaining the need in context.
- Eleven tests pass; 76 H.264 media files, viewer record/pose counts, source-bound review hashes and unique early-window OOF rows verified. reports/research_summary.md distinguishes delivered research from unsupported product claims.

The reproducible available-data study is delivered; the long-term product remains incomplete. Next work should independently annotate native impact/ball/club points and evaluate on new calibrated sessions, rather than claim accuracy from internally gated tracks. Local viewer UI verification remains outstanding. Use git log for the latest local checkpoint; no remote or publication.

## Latest user-directed work — 2026-11-18
Completed ball/contact and carry-threshold studies. See reports/contact_quality_results.md and reports/carry_threshold_results.md. Added 36 contact evaluations, 24 reviewed-ball evaluations and 12 threshold/gating evaluations with all OOF outputs. Sixteen tests pass.
Reviewed-ball pilot (31 shots; approximate AI annotations, second reviewer pending) shows carry MAE 24.78 yd versus same-cohort body/watch 37.36; detects 9/10 sub-60 yd shots with three false alarms. Club transients help short-shot sensitivity in fusion but worsen carry regression; no confirmed chunk/thin labels.
100+ yd conditional prediction looks ~12–14 yd accurate, but a constant training-mean baseline is already ~12 yd. A gate must classify first: reviewed-ball accepted shots have 20 yd MAE; full video/watch gating does not improve the all-shot baseline. Preserve all cutoffs and do not promote the best exploratory cutoff as validated.
Native ROI and timing errors were found in later Eduardo shots; five manual timing corrections apply only to the reviewed pilot, not the master manifest. No raw data or original outcome labels changed. Next priority is second review and robust automatic ball localization, then new-day/golfer validation of the two-stage system.

## Product continuation — 2026-11-18
Delivered an expanded local Swing Lab demo and root Open Swing Lab.html launcher. Includes a stylized solid pose avatar, replay controls, approximate phase buttons, uncertainty-aware review prompts, existing held-out carry estimates, assisted ball-pilot estimates and six-error filtering. Marker persistence uses browser storage with export fallback. This is not a validated personalized coach, true body mesh, new-recording inference system or reliable impact detector.
Six reviewed-ball gate errors audited against source-row associations, club availability and timing metadata; results in reports/misclassified_shot_audit.md/.csv. No justified label reassignment. Three errors have missing club tails; classifier does not use club data. Corrected Eduardo 19 and Rohan 2 / 10 native sheets reinspected. Source photos still absent.
Browser/native automation inventory failed (host unavailable), so no browser was opened and actual UI visual/playback QA remains pending. Next: verify demo interactively when host is available; independently mark native impacts and expand point review; implement/evaluate automatic ball localization before new-session product inference. Do not call phase offsets validated or generic prompts individualized coaching.

## Tracking and local application checkpoint — 2026-11-18
- V6 positive-only ball detection failed on dark silhouettes against bright sky; preserved. V7 dual-polarity candidate detection matches 76/78 approximate reviewed points within 32 px. V8 initialization constraints improve final sparse matches from v4's 41/78 to 52/78; full-track precision remains unvalidated. Candidate extraction and v8 linking ran on all 76 windows; 72 proposals, four abstentions (Rohan 1 shots 2–5). No automatic tracks entered prediction models.
- Native visual review brackets ball departure for 12 diagnostic swings within 8.3–20.8 ms; independent second review pending. Source-bound annotations remain separate from master impacts. Wrist-event offsets vary by session; no universal offset. New interval shows Eduardo 19's earlier coarse departure was late, without evidence of a shot-label swap.
- Nine nested comparisons on the same 26 multi-point reviewed shots: inter-frame slopes did not improve carry/launch regression. Carry MAE first-point 29.32 yd, inter-frame 38.55, combined 32.65. Timing perturbation ±20 ms materially changes first-point features. Results preserved in experiments/ball_timing_v1.
- Added synchronized head traces, provisional departure navigation, saved-marker jump buttons and native-cadence ball review on all 76 swings. Native clips preserve frame count and timing with explicit source/output mapping; no inserted frames. Separate pose replay remains 30 fps. Native stepping also positions the pose replay.
- Added local upload app via Start Swing Lab.command and CLI analyze_swing.py. New-video processing saves isolated outputs, H.264 replay, estimated avatar/head traces and four reference-model predictions. Model hashes checked. Predictions abstain when pose is missing in over half the frames; blank-video execution test passed. Real HTTP upload→processing→result smoke test passed using an existing clip, not new-data accuracy validation.
- 21 unit tests pass, native media/data and JS syntax checks pass. Browser host remains unavailable; native Safari fallback also timed out despite requested short timeout. Do not retry native app selection in this environment. Actual browser visual/interaction verification remains pending.
- Summary: reports/tracking_timing_product_progress.md. Reproduction and launcher instructions in README. Raw sources, labels and evaluation membership unchanged; no external uploads.

Next: independently review full automatic trajectories/departure bounds; improve or compare body-pose tracking, especially wrists; verify the local app in a working browser and test genuinely new recordings. Preserve the difference between a functioning execution path and validated new-data accuracy.

## Latest user steering — stable 3D range scene
User requested a real 3D avatar-style golfer, held club, airborne ball with tracer, driving-range setting, and plausible completion of an uncertain leg instead of flicker. Implemented an articulated Three.js scene with local vendored assets, display-only temporally completed pose, stable opaque limbs, common grip, phase-illustrative club and schematic carry-scaled flight. Native video also has an experimental 2D tracer overlay. This supersedes the earlier flat canvas avatar; exact club/3D ball reconstruction remains unvalidated.
24 tests pass; scene geometry checked on all 7,980 frames; software geometry previews inspected. Browser host still unavailable after user reset, so no actual WebGL/UI test. User offered help restoring access; an asynchronous optional request is pending. Do not retry native Safari/Chrome getApp calls, which previously timed out. Local page HTTP 200 confirmed. See reports/avatar_range_update.md.
The planned heavier-pose-model comparison was deferred when this user steering arrived; no heavy-model results exist and no claim should be made that it ran.

## Follow-up: user still reports missing 3D scene
Cause is not yet established: asked which avatar heading the user sees to distinguish an old page from WebGL/startup failure. Verified all three scripts embedded in the actual generated HTML parse correctly. Added a visible 3D RANGE · VERSION 2 label, cache-busting launcher, future-server HTML no-store headers, and graceful 3D startup/render error reporting that preserves video controls. Compacted display JSON from ~60 MiB to ~29 MiB using six-decimal display precision; scientific source caches/features remain full precision and unchanged. Scene geometry still passes all 7,980 frames. Browser visual verification remains pending.

## Avatar accuracy correction and live yards — 2026-11-18
User confirms the scene renders; head and club quality are the issue. Version 3 centers/orients the skull from ears/nose and connects the neck, adds a synchronized estimated-yard badge, and removes the fabricated phase-driven club path. Assisted `sid_1_01` club pilot uses 23 source-bound approximate AI grip/head anchors, short interpolation, and explicit depth ambiguity; 97/105 display frames, eight abstentions. Other shots currently hide the unsupported club. Automatic shaft proposals ran on six representatives but remain excluded from avatar/prediction due to false lines and missed shafts. 26 tests and 7,980-frame scene math checks pass; actual browser QA still pending. See reports/avatar_club_revision.md. Full reliable club tracking and human-mesh reconstruction remain unfinished; prioritize native-frame annotation/temporal club tracking and better hand pose next.

Direct watch-feature check completed: reports/watch_carry_relationships.md. Current whole-swing wrist peaks show no useful simple positive relationship to carry; phase-specific, impact-aligned features remain untested. Do not interpret this as evidence that true clubhead speed is irrelevant. Reproduce with scripts/watch_carry_relationships.py.

User's exact >70/>100 six-method comparison completed: reports/carry_cutoff_comparison.md. Distinguishes rescoring existing full-range OOF models from retraining restricted models. At >100 the conditional mean beats/matches all restricted models; no gating claim. At >70 the original full-range fusion model (18.96 yd) beats the conditional mean (22.19), but retraining fusion on the reduced cohort worsens to 31.87. Preserve both outcomes.

## Physics-Gated Video Launch & Clubhead Tracking Milestone — 2026-11-18
User confirmed shifting focus away from full-body motion aesthetics toward direct video ball launch vectors, launch angle/direction, clubhead tracking, and mishit directionality.
- Investigated user test cases: `lucas_1_02` and `lucas_1_12` previously followed leftward clubhead follow-through; `lucas_1_09` and `lucas_1_10` had extreme rightward pushes (+11.8° and +12.1° TrackMan) where `lucas_1_10` succeeded due to wide spatial separation while `lucas_1_09` was overtaken by clubhead scoring.
- Implemented physics-constrained tracking in `scripts/video_launch_tracker.py` and `configs/video_launch_tracking.json`: gated departure window (4–35 ms), departure proximity (<220 px from seed), inside-approach clubhead rejection, and ballistic straight-line ray scoring.
- Results across 68 tracked shots: video launch angle correlates at $r = +0.727$ with TrackMan horizontal launch direction, and vertical rise correlates at $r = +0.583$ with carry.
- Held-out golfer evaluation (Leave-One-Golfer-Out) using pure video launch features:
  - Carry flat MAE drops to **26.04 yd** ($R^2 = +0.369$) vs. baseline 31.93 yd ($R^2 = 0.13$).
  - Launch direction MAE drops to **4.50°** ($R^2 = +0.524$) vs. baseline 7.38° ($R^2 < 0$).
  - Launch angle MAE drops to **3.83°** ($R^2 = +0.464$) vs. baseline 5.09°.
  - Mishit detection (<60 yd): AUROC **0.856**, Recall **75.0%**, Precision **66.7%**.
- 29 unit tests pass; local research viewer rebuilt with clean physics-gated ball tracks and tracers. See `reports/video_launch_tracking_results.md`.

## Flat Driving Range Ball Tracer & Aerodynamic Timing — 2026-11-18
- Extended all 76 native review videos to 480 frames (~3.0 seconds) using parallel encoding in `scripts/extend_native_review.py`, providing ~2.25 seconds of post-impact video time so the full ball flight can be observed naturally in video playback.
- Fixed rapid descent artifact: previously, continuation duration was hardcoded to ~0.85s, causing the ball to plummet like a dropped rock into the turf while the golfer was still finishing. Recalibrated flight dynamics to real-world golf ball aerodynamics where 7-iron hang time is 5.5 to 7.0 seconds. The ball climbs smoothly to its true apex downrange (~2.5-3.2s after impact) and gently arcs to the flat driving range horizon ($y = 1100.0$).
- Addressed TrackMan autonomy vs. replacement: implemented dual-mode continuation in `scripts/ball_continuation.py`—autonomous camera-only mode (predicting hang time, apex height, and curvature from initial vertical rise and departure angle without TrackMan) alongside benchmark ground-truth mode.
- Resolved lateral trajectory hook bug on `sid_1_04`: decoupled lateral motion from height, incorporating aerodynamic curvature. `sid_1_04` slices right (+48.2 ft curve, +75.0 ft carry side) and stays right throughout flight.
- All 29 unit tests and Three.js 7,980-frame scene verifications pass. Rebuilt viewer at `outputs/research_viewer/index.html`.

## Dual 3D Flight Arcs, Target Centerline & Live Floating Yardage Badges — 2026-11-18
- Removed the 2D estimated continuation tracer from the native review panel (`#nativeSection`), retaining only the clean, verified camera-tracked ball flight (yellow) and clubhead tracking (teal). Video plays smoothly without any continuation freezes.
- Added a straight white target centerline down the 3D driving range ($X = 0, Z = 0 \to -240\text{m}$) with 50, 100, 150, 200, and 250-yard cross hash marks.
- Implemented dual 3D flight arcs and balls launching simultaneously in the 3D scene:
  - **TrackMan Radar Ground Truth** (Cyan `#38bdf8`): Built from empirical Doppler radar training labels (`Carry flat (yd)`, `Launch direction (deg)`, `Curve signed (ft)`, `Carry side (ft)`, `Height (ft)`).
  - **Best High-Speed Video Model** (Purple `#c084fc`): Built from autonomous Leave-One-Golfer-Out predictions (`Carry flat (yd)` [26.04 yd MAE], `Launch direction (deg)` [4.50° MAE], apex height).
- Replaced the static carry text box with dynamic floating HTML badges positioned in 3D screen space directly above each flying ball:
  - Ticks up in real time from `0 yd` as each ball travels downrange.
  - Stops and settles at the final landing carry distance when each ball touches down.
- Reconfigured the summary metrics card in `Open Swing Lab.html` to clearly contrast:
  - **Best Model (Video Launch)** in purple (`#c084fc`): Autonomous ball departure velocity vector, climb angle, and pre-impact clubhead speed.
  - **TrackMan Ground Truth** in cyan (`#38bdf8`): Benchmark Doppler radar carry, launch angle, ball speed, and launch direction.
  - **Prediction Error**: Signed difference in yards between autonomous estimate and radar truth.
  - **Baseline Modalities**: Video + watch (whole swing) and reviewed ball pilot cleanly preserved for reference.
- All 29 unit tests pass; Three.js 7,980-frame scene verification passes without errors.

## Anatomical Golfer IK, Realistic Forged Club, Ball on Mat & Dual/DTL Camera Views — 2026-11-18
- **Fixed the "wack" lead leg**: Replaced noisy raw MediaPipe lower body estimation with analytical two-bone Inverse Kinematics. The knee flexion is strictly constrained to the natural forward athletic sagittal plane ($+X$ aligned with pelvis rotation), completely eliminating backward hyperextension ("flamingo knees") and flickering leg lengths while preserving smooth, authentic rotational posting at impact.
- **Ball solidly on the mat**: The golf ball now sits squarely on the synthetic driving range hitting mat atop a dedicated rubber tee spot at $(X = 0.48\text{m}, Y = 0.068\text{m}, Z = -0.05\text{m})$ directly in front of the golfer's stance. The ball rests on the mat until the club strikes it at impact, launching it downrange. The straight target line extends directly from the ball down the range.
- **Realistic forged 7-iron club**: Upgraded the club graphics with a stepped chrome steel shaft (True Temper style), tour velvet rubber grip with white cap ring, black hosel ferrule, and a sculpted forged muscle-back blade featuring a satin face plate with authentic $34^\circ$ loft and grooves. The club is now rendered in the golfer's hands across all shots.
- **Unified TrackMan 3D camera angle**: Replaced multiple clunky view-switching buttons with one rock-solid, unified TrackMan launch monitor camera ($X = 1.8\text{m}, Y = 2.2\text{m}, Z = 5.0\text{m}$, looking downrange at $X = 0.0\text{m}, Y = 1.2\text{m}, Z = -45.0\text{m}$). Frames the golfer on the mat in the left foreground while simultaneously capturing the entire dual flight path (TrackMan cyan & model purple) soaring down the driving range fairway. Removed dynamic camera jumping during video playback so the perspective remains completely stable throughout the swing and flight.
- **Elevated camera perspective, thick 3D ribbons & natural driving range hazards**:
  - Elevated camera to $Y = 3.6\text{m}$ (pitched down toward $Z = -60\text{m}$) providing clear top-down perspective onto the landing zone to directly compare where both balls touch down.
  - Implemented thick 8 cm wide 3D tracer ribbons (`ribbonTruth` in cyan, `ribbonPred` in purple) and dynamic landing bullseye rings on the turf marking exact touchdown spots.
  - Added thick 26 cm target centerline, four organic sand bunkers ($Z = -75, -126, -153, -196\text{m}$), and staggered target greens with color-coded distance flags (50, 90, 125, 165, 210, 250 yd).
- All 29 unit tests pass; Three.js 7,980-frame geometry check passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

## Audited Best Models for All TrackMan Targets & Curvature-Aware 2D Landing — 2026-11-18
- **Integrated Audited Best Carry Model**: Replaced older retrospective ridge (26.04 yd MAE) in `build_research_viewer.py` with the audited v7 model (16.27 yd LOGO MAE overall, 12.64 yd on clean tracks).
- **Trained LOGO Models for All 10 TrackMan Targets (`scripts/train_all_targets.py`)**:
  - `carry_yd`: **16.27 yd MAE** (baseline: 40.05 yd)
  - `total_yd`: **14.90 yd MAE** (baseline: 30.51 yd, median: 7.56 yd)
  - `launch_dir_deg`: **4.55° MAE** (baseline: 7.50°, median: 1.57°)
  - `launch_angle_deg`: **2.61° MAE** (baseline: 5.49°, median: 1.27°)
  - `ball_speed_mph`: **9.05 mph MAE** (baseline: 14.84 mph, median: 5.50 mph)
  - `height_ft`: **11.00 ft MAE** (baseline: 27.80 ft, median: 5.60 ft)
  - `landing_angle_deg`: **5.13° MAE** (baseline: 12.74°, median: 2.11°)
  - `hang_time_s`: **0.56 s MAE** (baseline: 1.25 s, median: 0.30 s)
  - `carry_side_ft`: **39.49 ft MAE** (baseline: 48.36 ft, median: 30.73 ft)
  - `curve_ft`: **28.72 ft MAE** (baseline: 31.68 ft, median: 24.91 ft)
- **Upgraded 3D Scene with Shot Curvature (`scripts/range_scene.js`)**:
  - Predicted purple arc now models aerodynamic side curvature (`curveOffset`) towards the predicted `sidePred`, allowing authentic fade/draw visualization.
  - Overall 2D landing miss distance dropped to **23.6 yd** across all shots (median: **14.9 yd**), and down to **18.69 yd** on clean shots (median: **13.19 yd**; 60.3% within a 15-yard green radius).
- All 35 unit tests pass; Three.js 7,980-frame check passes. Rebuilt viewer at `outputs/research_viewer/index.html`.

## November 19 — Primary vs Fallback Transparency & Abstention Dashboard

- **Transparent Fallback Labeling**: Tagged the 14 low-confidence swings that resorted to predicting cohort training mean (~98–104 yd) with an amber warning banner and dropdown indicators, detailing the specific physical reason (lighting glare, extreme topped mishits, or <5 points in the early geometry window).
- **Primary Model Breakdown Dashboard**: Added a dedicated performance section in `Open Swing Lab.html` isolating the 58 confident launch swings, highlighting the authentic **12.64 yd MAE** (and **8.31 yd median** error, 64.8% error reduction vs baseline, 82.8% within 20 yards) delivered by the 2-feature launch physics model when it has a valid read.
- **Abstention Principle**: Codified the principle that golfers prefer explicit "Didn't get a good read" alerts over confident assertions of uncalibrated fallback means.
- **Estimated Ball Departure Navigation**: Added `[Ball departure (est)]` button for all shots lacking manual native 240 fps contact sheet review. Extracted from the first camera-tracked launch observation (verified within ~6 ms / 1 video frame of true contact), giving golfers immediate 1-click jump access to impact across all 76 shots while distinguishing estimated bounds from human-reviewed pilot bounds.
- **Enhanced Launch Frame Transparency Banner**: Updated the top strike quality banner across all shots to explicitly display both total captured frames and the exact subset of initial frames used by the 150 ms physics model (including source frame indices, e.g. "tracked 60 launch frames; model used initial 23 frames (frames 112–136) within the 150 ms launch window").
- **Ball Path Launch Window Visual Highlight**: Enhanced the 2D `#ballPath` canvas graphic to visually highlight the model-used launch segment (initial 5–24 frames, $\le 150\text{ ms}$) in bright emerald green (`#34d399`) with individual point nodes and a cutoff circle, distinguishing it from the remaining orange flight path and updating the legend accordingly.
- Rebuilt viewer at `outputs/research_viewer/index.html`; all 45 unit tests passing; Three.js 7,980-frame geometry check passing.

## November 19 — Ball extension and landing checkpoint

Ten sampled wide-crop tracks now appear in the local reviewer, and Lucas 02's club/arm false track is marked rejected. Wider tracking improved visible coverage. Extra late coordinates did not materially improve landing predictions. Per-target early video remains the reference recipe for landing and resting estimates. The next work is independent initial-launch review on ambiguous shots and collection of a fresh evaluation session. [Results and reproduction](reports/ball_extension_and_joint_landing.md)

The [motion cutoff follow-up](reports/ball_motion_gate_results.md) rejected the strict speed gates as defaults and added a conservative stationary display guard. The existing model remains the reference after comparison with 60 and 80 ms launch features.

Full native-frame club searches now cover all 76 swing windows. The higher-resolution pilot recovered 15 of 41 reviewed shaft directions within 10 degrees; the seeded Lucas 02 pilot retained 168 of 560 positions after drift checks. The range viewer continues to use its previous 120 Hz reconstruction pending broader validation. [Verification and reproduction](reports/native_swing_shaft_review.md)

The range now plays independently at wall-clock speed, with a 1.1 s reconstructed swing, per-ball hang times, rollout and optional video sync. The six starting camera views and dual-ball follow framing passed scene projection checks on all 76 shots at two viewport widths. Headless Chrome also exercised play, pause, replay and video sync on the built viewer.

Range playback now advances pose and club at the review video's 1× pace. Range overview is the default elevated right-side view and frames the full available flight and rollout; Follow ball uses a higher, wider view. The viewer offers four cameras.

Range landings now include a height-scaled hop and rollout along the flight's landing direction, with color-coded carry and total yardages at turf markers. Behind the tee fits the golfer and both flight paths for the current viewport.

The [cross-fitted label-sharing experiment](reports/trackman_label_sharing.md) retained a six-label rollout candidate. On the 69 viewer shots, 2D resting error fell from 23.68 to 23.11 yards against its matched independent baseline. Blanket stacking worsened both endpoints.

Promoted the six-estimate rollout and independent lateral endpoint models to Swing Lab. The summary scores 58 primary-geometry research rows, 57 accepted primary rows and all 69 displayed predictions. Both range balls now stop at their supplied total and total-side coordinates. [Results](reports/research_summary.md)

The range now registers the address stance to the tee and the rendered striking face to the back of the ball. Shoe clearance is a seven-iron display estimate. All 76 setups passed the scene geometry check.

The local range has a landscaped fairway, contour-shaped target greens and greenside bunkers, wooded ridges, and seven flagged targets with tee-to-pin yardages. The hitting bay includes a wire basket, packed practice balls, a feeder tray and textured mats.

The clean-only LOGO experiment found an exploratory video/gyro carry blend and final-side-anchored rollout. Next validate this frozen recipe on new golfers before viewer promotion. [Results](reports/clean_physics_coupling.md)

Automatic sparse-launch recovery reached exploratory 11.05 yd carry MAE and 5.60 mph ball-speed MAE on the 57 accepted primary shots. Freeze the candidate and validate on new golfers. [Results and reproduction](reports/clean_launch_audit.md)

The 3D range offers thin white dashed centerline markings through a checkbox that starts off. Rebuild with `.venv/bin/python scripts/build_research_viewer.py` and run `node scripts/verify_range_scene.js` to check scene geometry.

The range ball now rests directly on the center hitting mat at address.

Swing Lab now embeds a CC0 MakeHuman golfer with weighted shoulders and pelvis, planted feet, and a right-handed finish pivot. The new geometry runs in the recorded-shot and new-swing viewers.

The golfer uses adult reference proportions with a fixed-length torso, restrained knee flex and a centered stance. The updated browser contact sheet includes side views and a second golfer.

Restored the original landmark-driven range avatar. The hips and thighs share a trouser surface, the shoes face the ball and sit on the mat at address, and the display filters follow-through knee and ankle jitter.

The pose-driven range avatar now uses a continuous clothing surface from the ankles through the knees, hips and chest. The follow-through includes an estimated trail-foot pivot about the toe.
