# Golf Shot Modeling

Executed research codebase for the specification in `prompt.md`. Resume with `AGENTS.md` and `PROJECT_STATUS.md`.

Current results: 76 reviewed swing identities, 72 labeled shots, 65 labeled club-covered shots; 76 pose clips; native ball/shaft diagnostics; 400 fixed-model and 26 nested carry comparisons; 360 learning-curve fits. See `reports/model_results.md`. This is an exploratory research system, not a validated launch monitor or polished app.

## Setup
Tested on Apple Silicon, Python 3.10.7, FFmpeg/ffprobe 4.3.2. Use an isolated venv; system Python packages on the original machine contain incompatible binaries.
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
`requirements-lock.txt` records the resolved environment. FFmpeg/ffprobe must be on PATH. Keep originals unchanged at project root: `Club Data/`, `Watch Data/`, `Video Data/`, `trackman_shot_labels.xlsx`. Teammates obtain recordings privately from the owner; no public data URL or remote is configured.

Download Google's versioned pose model:
```
mkdir -p models/pretrained
curl -L --fail https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task -o models/pretrained/pose_landmarker_full.task
```
Expected SHA-256: `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.
All inference is local. Method/equipment sources are in `reports/geometry_and_sources.md`.

## Reproduce on the supplied dataset
Run from project root using the venv:
```
python scripts/audit_dataset.py
python scripts/sample_videos.py
python scripts/detect_events.py
python scripts/check_event_alignment.py
python scripts/align_club.py
python scripts/process_pose.py
python scripts/make_review_sheets.py
python scripts/extract_features.py
python scripts/benchmark.py
python scripts/nested_carry.py
python scripts/analyze_benchmarks.py
python scripts/nested_uncertainty.py
python scripts/train_reference.py
python -m unittest discover -s tests -v
```
`annotations/shot_review.csv` contains the completed visual review for this exact dataset. Feature extraction checks its video/workbook hashes and candidate times; new or changed data requires a new review, not automatic acceptance. Re-running the audit resets the enriched manifest, so rerun event/alignment/feature stages afterward.

Optional diagnostics:
```
python scripts/ball_diagnostic.py
python scripts/track_ball_demo.py
python scripts/shaft_diagnostic.py
python scripts/shaft_diagnostic.py --motion
python scripts/head_diagnostic.py
python scripts/track_balls.py --representatives
python scripts/track_balls.py
python scripts/camera_events.py --session "Sid 1"
python scripts/verify_checkpoint.py
```
`process_pose.py --representatives` writes a separate six-shot feature table. Full pose processing caches per-shot landmarks/features. Ball proposals cache per-shot JSON and diagnostic images. Full camera-event scanning is expensive; its presence trace is cached. Do not confuse detector coverage with localization accuracy.

First source audit hashes ~12 GB. Hash/metadata caches live under `outputs/audit/cache/`. Delete only `hashes.json` there to force rehashing. Sparse preview files must be invalidated if sources change. `verify_checkpoint.py` intentionally rehashes every original and checks output consistency.

## Organization
- `configs/`: source/session mapping, equipment, pose, ball and benchmark settings; seed 396.
- `scripts/`: ingestion, synchronization, CV, features, evaluation and reporting.
- `annotations/`: source-bound shot review and unreviewed ball proposal queue.
- `data/manifests/`: original inventory, label extraction, event candidates, shot manifest and audit evidence.
- `data/processed/`: regenerable private feature/landmark caches, excluded from Git.
- `experiments/baseline_v1/`: fixed splits, configurations, metrics and all OOF predictions.
- `experiments/nested_carry_v1/`: nested family selections, metrics and predictions.
- `models/reference_ridge/`: 12 regenerated full-data reference fits, excluded from Git.
- `outputs/`: private overlays, traces, plots and caches, excluded from Git.
- `reports/`: data audit, results, collection plan, geometry sources and verification.

## Scientific boundaries
User confirms nth actual strike corresponds to nth Excel row within each named session, including unregistered strikes, with no practice swings. Club may stop early; never shift shot IDs to hide a missing event.

Current video regression uses watch-assisted clips and includes follow-through. It is not an autonomous camera-only or pre-impact/practice-swing result. Native ball features are not in benchmark inputs: 36/76 automatic proposals require independent point review. Pose visibility is not localization accuracy, especially at the wrists. Metric clubhead/ball speed, exact launch geometry, spin, gaze and calibrated 3D motion are unvalidated.

Within-session and leave-golfer-out folds are separate; all shot modalities stay in the same fold. Preprocessing fits inside training folds. Nested carry results are primary; fixed-grid minima are retrospective. There are five golfers and one collection day, no untouched external test set.

## Git and sharing
A project-local repository on `main` is initialized. Always verify `git rev-parse --show-toplevel`: the original machine also has an unrelated home-directory repository. Raw recordings, original workbook, generated media, models, environment and secrets are ignored. No remote upload or push is authorized. Share derived research outputs deliberately; regenerate excluded artifacts after obtaining the private sources.

## Latest research extensions and viewer
```
python scripts/review_ball_frames.py
python scripts/validate_ball_proposals.py
python scripts/early_swing.py
python scripts/phase_candidates.py
python scripts/track_balls.py --representatives --config configs/ball_tracking_v5.json
python scripts/build_research_viewer.py
python scripts/render_pose_3d.py --shot sid_1_01
python scripts/verify_research_delivery.py
```
Open `outputs/research_viewer/index.html` locally. The builder converts all overlays to browser-compatible H.264 and embeds estimated pose coordinates. Includes 76 shots, frame stepping, out-of-fold predictions, rotatable estimated skeletons and source-bound phase annotation export. Export marks before reload; they are not automatically saved. Browser UI verification could not complete because browser automation timed out; media/data checks passed. No service, upload or network library is used by the viewer.

Read `reports/research_summary.md` for delivery status. Early-window results are not validated pre-impact/practice-swing predictions. Point references were approximated from unmarked native frames by AI visual review with explicit uncertainty, and still need a second reviewer. V5 tracking failed to recover Andrew/Eduardo representative flights. Phase candidates are not accepted annotations.

## Contact quality and carry cutoffs
New executed studies: `reports/contact_quality_results.md` and `reports/carry_threshold_results.md`. Reproduce with:
```
python scripts/contact_features.py
python scripts/contact_benchmark.py
python scripts/contact_diagnostics.py
python scripts/contact_video_review.py
python scripts/reviewed_ball_features.py
python scripts/reviewed_ball_benchmark.py
python scripts/carry_threshold_experiment.py
python scripts/summarize_contact_studies.py
```
The reviewed-ball pilot is explicitly assisted, uses approximate source-bound AI annotations and still needs a second reviewer. Automatic ball proposals remain excluded. Conditional 100+ yd error must never be reported as end-to-end error: read the predicted acceptance, false accepts/rejects and conditional constant baseline alongside it. Contact flags predict short/low/directional outcomes, not validated chunk/thin/hosel mechanisms.

## Swing Lab product demo (2026-11-18)
Double-click **Open Swing Lab.html** in this project, or open `outputs/research_viewer/index.html` in your browser. All assets stay local; no server or internet is required. Keep the project folders together.

The demo includes all 76 clips, slow motion and frame stepping, a rotatable solid avatar from estimated world pose, approximate phase navigation, confidence-aware review prompts, held-out carry estimates, a six-error filter and local phase annotation storage/export. This is a recorded-session prototype: new uploads, calibrated flight reconstruction, validated phase detection and personalized coaching are not implemented. Browser automation is unavailable, so actual playback/interaction QA remains pending despite passing data/media and JavaScript syntax checks.

Rebuild with `.venv/bin/python scripts/build_research_viewer.py`; verify with `.venv/bin/python scripts/verify_research_delivery.py`. See `reports/misclassified_shot_audit.md` for the six errors and alignment limitations. Local browser storage may be unavailable under file URLs; export markers to preserve them.

## Local video analysis and native-frame review

Double-click **Start Swing Lab.command**, then open the loopback URL printed in Terminal. Keep that Terminal running. The page links the research demo and accepts a local MOV/MP4 (up to 2 GiB). Select the beginning of a 3.5-second window containing one complete swing. Processing and uploaded copies remain on this computer, under `outputs/new_swings/`; no cloud upload occurs. Stop the service with Ctrl+C. The original source is unchanged.

Alternatively:
```sh
.venv/bin/python scripts/local_app.py --port 8765
.venv/bin/python scripts/analyze_swing.py --video '/path/to/swing.mov' --start 3.0 --name my_swing
```
The new-video pipeline uses a fixed video-only ridge reference fit trained on the 72 labeled research shots. It does not use held-out predictions for new input, and its accuracy on new days/camera positions is unvalidated. Existing-recording smoke tests verify execution, not generalization. No calibrated speed or diagnosis of chunk/thin contact is produced. Keep the trained reference model files locally; reproduce them with `scripts/train_reference.py` if absent.

Research demo additions: synchronized image-plane head traces on all 76 swings; source-bound provisional departure review on 12 swings; native-cadence cropped ball/impact clips with timestamp-based frame stepping; experimental automatic trajectory plots. Automatic tracks remain excluded from prediction inputs. Source cadence is preserved in the native view; the separate pose replay stays at 30 fps. Browser visual/interaction QA remains pending, while local HTTP upload, processing, result delivery and media-range tests have passed.

Reproduction for this continuation (cached decoding is reused):
```sh
.venv/bin/python scripts/pose_observations.py
.venv/bin/python scripts/render_head_observations.py
.venv/bin/python scripts/ball_tracking_v6.py --version 6
.venv/bin/python scripts/link_ball_v6.py --version 6
.venv/bin/python scripts/ball_tracking_v6.py --version 7 --all
.venv/bin/python scripts/link_ball_v6.py --version 7 --reviewed-only
.venv/bin/python scripts/link_ball_v6.py --version 7 --link-version 8
.venv/bin/python scripts/impact_review_sheets.py
.venv/bin/python scripts/compile_impact_review.py
.venv/bin/python scripts/ball_timing_study.py
.venv/bin/python scripts/render_tracking_comparison.py --version 8
.venv/bin/python scripts/build_native_review.py
.venv/bin/python scripts/summarize_tracking_progress.py
.venv/bin/python scripts/build_research_viewer.py
.venv/bin/python scripts/verify_research_delivery.py
.venv/bin/python scripts/verify_local_app.py --process
```
The final command starts a temporary loopback server and runs an actual clip through the upload pipeline. It requires local socket permission; it sends nothing externally. Results and limitations are in `reports/tracking_timing_product_progress.md` and `reports/local_app_verification.json`.
