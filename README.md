# Golf Shot Modeling

Executed research codebase for the specification in `prompt.md`. Resume with `AGENTS.md` and `PROJECT_STATUS.md`.

Current results: 76 reviewed swing identities, 72 labeled shots, 65 labeled club-covered shots; 76 pose clips; native ball/shaft diagnostics; 400 fixed-model and 20 nested carry comparisons; 360 learning-curve fits. See `reports/model_results.md`. This is an exploratory research system, not a validated launch monitor or polished app.

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
