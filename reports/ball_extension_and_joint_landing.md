# Ball track extension and joint landing benchmark

November 19, 2026

The old tracking crop hid usable ball observations on several shots. The wider decode recovered a visible ball above that boundary in sampled native frames. Longer tracks improved replay coverage, while their late position features produced almost identical average landing error to the same features from the original tracks. Separate regressions for each outcome performed better than the tested joint regressions on the full cohort.

## Tracking and native review

The original detector used the source crop `[300, 650, 780, 1100]` and a 1.65-second decode. The experimental crop is `[180, 180, 900, 1570]` with a 2.10-second decode, measured on the video playback clock. The crop and decode duration changed together. Recovery within the original coordinates can come from the longer decode as well as the extension logic. All variants preserve the same proposed launch prefix, where available. The extension fits local motion using the last six points and requires three compatible detections before resuming. A compatible point must move with the recent direction and fit the expected position and component size. Ambiguous branches and isolated detections are dropped.

| Variant | Total proposed points | New proposed points | Shots extended | New points above old crop |
|---|---:|---:|---:|---:|
| Original tracks | 2,672 | 0 | 0 | 0 |
| Original crop, 35 ms | 2,954 | 282 | 22 | 0 |
| Wide crop, 35 ms | 3,326 | 654 | 31 | 342 |
| Wide crop, 70 ms | 3,416 | 744 | 31 | 366 |

All 76 windows were counted. There were 73 existing proposals, and this extension experiment did not initialize the other three shots. The 70 ms rule added 90 proposals over 35 ms across 12 shots. All four variants matched 43 of 78 existing approximate reference points within 16 pixels and 44 within 32 pixels, with a 6 ms timestamp tolerance. These existing points mainly check the original trajectory; they provide little coverage of the newly extended tail. The totals above measure detections and coverage. They cannot establish an all-frame false-positive rate.

I examined four added native frames per extended shot below, sampled across each new tail. The visible ball was consistent with the proposed position in those samples. Review sheets show points over the matching native decoded image.

| Shot | Before | After | Gap setting |
|---|---:|---:|---:|
| Sid 01 | 81 | 90 | 35 ms |
| Sid 05 | 84 | 96 | 35 ms |
| Sid 08 | 60 | 120 | 70 ms |
| Sid 12 | 82 | 97 | 70 ms |
| Sid 15 | 42 | 102 | 35 ms |
| Sid 16 | 77 | 113 | 70 ms |
| Rohan session 2, shot 01 | 40 | 103 | 35 ms |
| Rohan session 2, shot 05 | 76 | 100 | 35 ms |
| Lucas 05 | 62 | 71 | 70 ms |
| Eduardo 07 | 64 | 79 | 70 ms |

The local viewer uses timing-checked wide footage and extensions for these ten shots. It clears the active tracer after 35 ms without an observation and breaks lines at longer gaps. If the tracker subsequently reacquires the ball, the new segment appears on its recorded timestamps. The history graph retains a muted record of observations. View the [Sid 08 contact sheet](../outputs/ball_extension_review/sid_1_08_wide70.jpg) and [Rohan session 2, shot 01](../outputs/ball_extension_review/rohan_2_01_wide70.jpg).

**Lucas 02** was a real identity error. The five-point candidate moved onto the golfer's hand, shaft and clothing. The viewer now rejects that path and omits the video launch prediction. The existing automatic cache was preserved as a benchmark reference. None of the extension variants found a defensible replacement. **Lucas 09** retained ten rightward candidate points. **Rohan session 1, shot 02** retained ten candidate points. Neither gained an extension, and their sampled launch images remain difficult to resolve. The intervention addresses tail recovery after a valid launch. It cannot repair a false initial lock by extending it.

## Joint prediction experiment

Every outer fold holds out one golfer and both sessions for that golfer when applicable. Inner golfer groups select estimators using the training data. The all-label cohort has 72 shots. The matched club cohort has 65 shots with an actual synchronized club recording. Eight rows in the club feature file carry vision-derived stand-ins for missing club signals. The benchmark checks `has_club_imu` before including any row in a club comparison.

Landing error is the mean Euclidean miss between the estimated and recorded carry/side coordinates, with lateral feet converted to yards. Resting error uses total distance and total side. These are explicitly defined coordinate proxies from the spreadsheet. The joint models share a PLS representation or a multitask penalty across 11 standardized outcomes, including speed, direction, angle, height, hang time and curve. The structured alternatives train on the predicted landing point plus the change during rollout. Recorded TrackMan outcomes are training targets and scoring references. The per-target model selects a separate family for each of the four endpoint coordinates and can use the established video availability fallback for carry.

| Inputs and recipe, same 72 shots | Landing miss (yd) | Resting miss (yd) |
|---|---:|---:|
| Training-fold mean coordinates | 46.59 | 39.65 |
| Body pose + watch, per-target model selection | 35.57 | 35.09 |
| Early video, independent family selected on endpoint error | 29.48 | 28.41 |
| Early video, per-target model selection | 25.19 | 26.94 |
| Early video + watch, per-target model selection | 25.18 | 29.83 |
| Early video + watch, structured joint | 28.54 | 28.43 |
| Early video + original late positions, per-target | 24.63 | 31.04 |
| Early video + wide 35 ms late positions, per-target | 24.61 | 32.74 |
| Early video + wide 70 ms late positions, per-target | 25.77 | 34.06 |

The wide 35 ms variant improved mean landing error by 0.019 yards relative to the same late features from the original tracks. The difference is too small to justify changing the forecasting inputs, and resting error increased. The earliest two projection features still supplied the carry prediction in the per-target recipes, giving 16.27 yards carry MAE on all 72 shots. The wide features change the lateral and resting estimates. The early-video recipe reduced landing and resting error by roughly 46% and 32% respectively compared with training-fold means.

On the 65 matched club shots, the early-video per-target model scored **23.16 yards landing / 25.56 resting**. Adding the three proposed gyro descriptors produced **23.51 / 40.67**. The best tested joint method on this cohort did not beat the per-target early video landing result. On the full cohort, the structured video + watch joint model had lower lateral MAE than early video per-target (14.34 vs. 15.44 yards), but its carry MAE rose from 16.27 to 21.91 yards. This explains how joint training can help one outcome and worsen the endpoint.

The historical Gemini handoff mixes tracking-selected cohorts with matched club cohorts, and mixes an airborne endpoint with a resting endpoint under the term “2D error.” The saved joint comparisons use one explicit metric definition per endpoint and pair identical shot IDs across methods. Current tracks and benchmarks were developed repeatedly on these recordings. Model estimates use the pre-existing launch geometry cache for a controlled comparison, and retain all 72 labeled shots, including the viewer-rejected Lucas 02. New late features are exploratory candidate data. The near-tied landing results do not support a strong ranking between variants on new recordings.

## Reproduce

Run from the project root using the local virtual environment.

```sh
.venv/bin/python scripts/ball_tracking_v6.py --version 11 --all
.venv/bin/python scripts/extend_ball_tracks.py
.venv/bin/python scripts/review_ball_extensions.py sid_1_01 sid_1_05 sid_1_08 sid_1_12 sid_1_15 sid_1_16 rohan_2_01 rohan_2_05 lucas_1_02 lucas_1_05 lucas_1_09 rohan_1_02 eduardo_1_07
.venv/bin/python scripts/build_native_review.py --wide --shots sid_1_01 sid_1_05 sid_1_08 sid_1_12 sid_1_15 sid_1_16 rohan_2_01 rohan_2_05 lucas_1_05 eduardo_1_07
.venv/bin/python scripts/build_research_viewer.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_joint_landing.py --variants baseline narrow35 wide35 wide70 --out joint_landing_v2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_research_delivery.py
node scripts/verify_range_scene.js
```

Inputs, point matches and metrics are in [ball_extension_v1](../experiments/ball_extension_v1) and [joint_landing_v2](../experiments/joint_landing_v2). The next tracking priority is independent frame annotation of the initial launch and confusing club/ball overlaps, followed by a quality check before any extension starts. The next modeling priority is a fresh recording session and separate tests of lateral-curve features, using correctly tracked outcomes and paired evaluation.
