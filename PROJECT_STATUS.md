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
