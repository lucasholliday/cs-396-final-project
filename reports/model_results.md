# Initial research results — 2026-11-17

This is an executed exploratory benchmark, not a validated launch-monitor replacement. No polished application is justified yet. Source recordings and TrackMan workbook remain unchanged.

## Data and protocol
76 user-ordered physical-shot candidates were matched to stable watch events and visually reviewed in three-frame swing contact sheets. 72 have TrackMan outcomes. 68 have usable club windows, of which 65 have labels. Four unlabeled strikes remain in the manifest. Eight missing club tails are retained for video/watch research. Exact impact timestamps and nine swing phases are not fully annotated.

Primary target: carry flat, yards. Secondary targets: ball speed mph, launch angle degrees, signed launch direction degrees. Clubhead speed and spin are absent from ground truth. All shots use a seven iron; five golfers and six sessions from one collection day. No independent-day test and no practice swings.

Inputs: nine projected body/motion features; ten whole-swing watch acceleration/gyro magnitude features; five club gyro features. Saturated club acceleration was excluded from predictors: all 68 club windows contain near-limit flags, median 7.96% of samples. Low-confidence wrist pose features were excluded. No TrackMan outcomes, golfer IDs, file names, timestamps or shot numbers are predictor columns.

**Camera-feature results use watch-assisted clip selection** and include follow-through. They are video-input regression on segmented swings, not an autonomous end-to-end camera-only product, pre-impact prediction, or practice-swing validation. No ball-flight features enter these baseline results. Pose clips are resampled to 30 fps for slow body motion; native frames are used separately for ball/shaft trials.

Two persisted five-fold protocols:
- Within-session contiguous blocks: held-out shots from already represented sessions/golfers. Nearby-shot correlation and same-day conditions remain; this does not estimate new-session performance.
- Leave-golfer-out: all shots of one golfer held out, including both Rohan sessions together. Only five golfers, so uncertainty is substantial and golfer/lighting/session effects are confounded.

All preprocessing/imputation/scaling fit inside training folds. Initial fixed settings: mean, median, ridge alpha 10, Extra Trees (150 trees, leaf size 5), SVR (C=1, epsilon=0.1 after target standardization). 400 target/cohort/protocol/feature/model evaluations ran, each with five external folds. No hyperparameters were optimized against external fold predictions.

For carry, a second experiment performs **nested family selection** among mean/ridge/Extra Trees/SVR in three grouped inner folds. Its 20 comparisons are the primary carry results below. Selecting the best feature set after viewing these results still introduces comparison uncertainty; there is no untouched final test cohort.

## Nested carry results: all 72 labeled shots
| Features | Known-session MAE / RMSE (yd) | Held-out golfer MAE / RMSE (yd) |
|---|---:|---:|
| Median baseline | 35.17 / 46.15 | 40.56 / 50.81 |
| Video features | 25.45 / 35.15 | 31.93 / 41.95 |
| Watch | 29.98 / 38.97 | 32.97 / 42.05 |
| Video + watch | 26.60 / 36.47 | 29.87 / 39.12 |

Video known-session R² = 0.351, median absolute error = 16.88 yd. Video + watch held-out-golfer R² = 0.196, median absolute error = 21.62 yd. These are modest results with large individual failures.

## Nested carry comparison: same 65 labeled club-covered shots
| Features | Known-session MAE (yd) | Held-out golfer MAE (yd) |
|---|---:|---:|
| Video | 27.25 | 31.15 |
| Watch | 30.10 | 35.93 |
| Club gyro | 34.23 | 41.60 |
| Video + watch | 25.86 | 29.99 |
| Video + club | 28.68 | 30.65 |
| Watch + club | 31.23 | 50.48 |
| All three | 27.79 | 28.58 |

Do not compare a 65-shot number directly against a 72-shot number to infer sensor benefit. Missing club tails change the cohort.

Paired held-out-golfer error differences, golfer-cluster bootstrap (2,000 resamples, five clusters), negative favors the added modality:
- Video+watch minus video: −1.16 yd, nominal 95% interval [−6.94, +3.28].
- All minus video+watch: −1.41 yd, [−4.93, +0.96].
- All minus video: −2.57 yd, [−7.31, −0.16].

The last interval is suggestive, but only five clusters, multiple comparisons, shared training folds and selection of the club-covered subset prevent a robust claim that every extra sensor helps. Fixed Extra Trees comparisons show even smaller/ambiguous gains. Preserve both experiments instead of cherry-picking the favorable one. Club gyro alone does not establish useful unseen-golfer carry prediction here.

## Secondary targets
These are **retrospective best fixed-model results**, not nested-selected target results:
| Target | Known-session best MAE | Held-out-golfer best MAE | Held-out median baseline |
|---|---:|---:|---:|
| Ball speed (mph) | 9.30 | 11.92 | 16.22 |
| Launch angle (degrees) | 4.74 | 5.09 | 5.32 |
| Launch direction (degrees) | 6.81 | 7.38 | 7.38 |

Angle/direction gains are weak; held-out launch-direction R² is negative. No reliable spin, clubhead speed or precise launch measurement is demonstrated.

## Error analysis
The known-session video+SVR model predicts normal carries for several very short shots: lucas_1_10 actual 16 yd / predicted 129.77; lucas_1_04 26 / 115.96; rohan_2_02 10 / 99.05; rohan_2_06 36 / 113.83; andrew_1_08 9 / 72.34. Retain these observations. Whole-body motion is insufficient to distinguish contact quality in these examples; do not remove them to improve scores. Investigate ball/club contact features and explicit mishit annotation next.

## Computer vision trials
- Pose: 7,980 frames across 76 clips, body detected in every frame. Mean landmark visibility fraction 0.843; mean both-wrists-valid fraction only 0.189 (range 0.095–0.438). Confidence/availability is not localization accuracy. Occlusion and estimated hand locations preclude precise wrist biomechanics claims.
- Head: saved confidence-masked/smoothed face-landmark centroid trace and overlaid path. It is a 2D image projection relative to clip start, not calibrated head height, annotated address, gaze, or a causal coaching diagnosis.
- Ball: bright-white blobs alone failed (2,876 candidates in 106 native ROI frames). Background differencing plus manual initialization followed sid_1_01 for 31 proposed locations over 0.179 playback seconds. One tiny component near background crossing is questionable. Multi-hypothesis automatic seed/departure tracking was then executed on all 76 shots: 36 candidate tracks passed internal gates (Sid 15/18, Rohan 1 2/5, Rohan 2 5/10, Lucas 8/12, Andrew 4/11, Eduardo 2/20). These counts are proposal coverage, not accuracy. Representative montages reveal motion blur, shaft confusion and background-tail risks. All proposals remain excluded from benchmark inputs pending independent point review. No metric speed, 3D launch or carry was derived.
- Club: unrestricted Hough lines produced misleading background proposals in all 106 frames. Temporal motion masking reduced proposals to nine partial shaft segments near impact, visually following the shaft. Endpoints are not verified clubhead/grip positions; full-swing tracking and 3D club path remain unimplemented.
- Estimated pose-world coordinates are cached but not validated motion capture. Human mesh/rigged avatar and coaching are deferred until tracking is trustworthy.

## Learning curves and next work
360 fixed-ridge learning-curve fits ran (15/25/40/50 training shots, three repeats, five external folds, two protocols, three feature sets). Known-session video MAE mean changes 30.95→28.78→28.42→28.49 yd, suggesting the present features plateau. Held-out video MAE changes 42.93→36.83→35.05→34.10, with large golfer variability. This supports better contact measurements and more diverse independent sessions; it does not justify extrapolating an exact sample size for a target accuracy.

See `data_collection_recommendations.md`. Highest-value work: validate automatic native ball trajectories across shots, improve shaft/clubhead localization, annotate actual impact and swing phases, and rerun separate pre-impact and post-impact benchmarks. A polished app, gaze feedback and precise metric coaching are premature.

## Reproduce and inspect
`scripts/benchmark.py`, `scripts/nested_carry.py`, `scripts/analyze_benchmarks.py` using `.venv/bin/python`.
`experiments/baseline_v1/`: configuration, fixed splits, 400 metric rows, every out-of-fold prediction, learning curves, paired uncertainty, per-golfer/per-shot errors.
`experiments/nested_carry_v1/`: 20 nested comparisons, inner model selections and predictions.
`scripts/nested_uncertainty.py` reproduces nested paired intervals; `scripts/train_reference.py` saves 12 fixed-ridge full-data fits for inspection, not independent performance evaluation. Model binaries are excluded from Git.
`outputs/plots/`: model comparison, learning curves, actual-versus-predicted plots (visually inspected).
`outputs/pose/`, `outputs/ball/`, `outputs/club/`, `outputs/head/`: private research visualizations. Read PROJECT_STATUS for latest additions after this checkpoint.

## Camera-only event experiment
On Sid's full session, a 20-fps compact bright-component disappearance detector proposed 29 events. All 18 distinct watch-reference swing neighborhoods had a candidate within ±1.5 s, with 11 extras. Native trajectory filtering retained 18 events: 15 matched distinct reference neighborhoods, three were extras, and three swings were missed (83.3% neighborhood precision/recall under this exploratory matching rule). Watch signals were used only for evaluation, not candidate detection or filtering. Thresholds/ROI were developed on this scene, so these are not held-out performance estimates or exact impact accuracy.
Visual inspection of events 05/12/19/27 showed plausible flight in 05/19, a missed visible ball in 12, and leg/background confusion in 27. Counts of internally accepted trajectories must not be interpreted as validated ball tracking. Results: reports/camera_events/; diagnostic montages: outputs/camera_events/tracks/.

## Early-window ablation (executed)
Six additional nested carry comparisons use the same 72 shots and persisted outer splits. Inputs are limited to watch-peak-relative [-2.0, -0.8] seconds, with confidence-masked head/shoulder/hip movement and watch gyro/acceleration summaries. Inner training folds select the model family. No ball or outcome inputs.

| Features | Within-session MAE (yd) | Held-out-golfer MAE (yd) |
|---|---:|---:|
| Early video | 30.54 | 40.05 |
| Early watch | 30.02 | 47.33 |
| Early video + watch | 30.24 | 43.26 |

These features generalize worse than the full-swing versions. The ablation changes both time window and feature representation, so the difference cannot be attributed solely to follow-through. This is an **offline early-window experiment**, not a validated pre-impact/practice-swing model: video/watch offsets vary by session, and individual impact exclusion has not been verified. In particular, Rohan 2's first detected departure is earlier than the assumed typical Sid offset. No practice-swing carry claim follows from these results. Full artifacts: experiments/early_swing_v1/.

## Point review, rejected improvements and phase proposals
Eighteen unmarked native panels were exported, three for the first strike of each session, selected at fixed offsets from the watch proxy without reference to predicted trajectories. Eight panels contained confidently identifiable airborne balls: three Sid, three Lucas, two Andrew. Approximate AI visual annotations are source-hash-bound and carry 6–12 px uncertainty; a second reviewer is still needed. The six Sid/Lucas points coincide with v4 proposals (errors 0.54–4.19 px, below annotation uncertainty). Andrew's two visible flight points have no proposed track. This tiny, selected spot check does not establish general tracking accuracy; it cannot certify entire trajectories or permit ball features into ML. Other panels show stationary balls or ambiguous flight, demonstrating that a universal watch-to-impact offset is inadequate.

A stationary-tail rule (three consecutive intervals below 200 px/s) reduces Sid's filtered camera events from 18 to 16: 15 reference neighborhoods and one extra leg-confusion event remain. It was developed on Sid, so this is a development result. Applying the frozen rule to all 76 cached shot proposals retains 35 of 36; counts are not accuracy. A separate v5 trial widens lateral search and allows shorter moving tracks to recover Andrew's leftward flight: still only four of six representative shots pass, with Andrew/Eduardo failing. Do not adopt it as an accuracy improvement.

Top-of-backswing proposals use visible wrist-height extrema with observations on both sides: 52 candidates, 24 abstentions. These are unvalidated projected extrema, with no automatic impact label. Source-bound phase review/export is available in the local viewer. Estimated 3D skeleton playback and an exported side-by-side clip illustrate the cached model coordinates; no mesh, calibrated kinematics or coaching claims are made.

## Subsequent contact and threshold studies (2026-11-18)
See contact_quality_results.md for 36 sensor-contact evaluations and 24 reviewed-ball pilot evaluations. On the same 31 reviewed shots, approximate visually reviewed ball features reduce carry MAE 37.36→24.78 yd relative to body/watch. This is assisted feasibility, not current automatic tracking. Club contact features improve some short-shot sensitivity but worsen distance regression and add false alarms.
See carry_threshold_results.md for twelve 80/100/120-yd threshold experiments. Known 100+ yd shots give ~12–14 yd MAE, but conditional training means already achieve ~12 yd. Operational classifier errors materially worsen accepted/all-shot results. No cutoff was selected as a validated winner.
