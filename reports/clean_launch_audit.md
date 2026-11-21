# Native launch review and clean-cohort validation

Five outer folds hold out all shots from one golfer. The 57 accepted primary shots and the equal-weight reference match the preceding experiment. Point substitutions compare frozen models. The automatic refit applies the same image processing to training and held-out shots; model preprocessing and estimation fit only training golfers. TrackMan speed and carry are supervised targets. All methods use post-impact video.

## Frame review

Three large video-model misses and one low-error control from each corresponding golfer were selected retrospectively. The six review strips omitted TrackMan and prediction values. The two largest misses were already known to the reviewer. One AI reviewer placed approximate ball centers at native frame times from enlarged crops, with original automatic candidates indicated by rings. These annotations are provisional and cannot serve as independent test labels. Last frame is the end of the reviewed identifiable segment, rather than a claim about disappearance from the full recording.

| Shot | Last reviewed frame | Overlapping automatic points | Median offset in original pixels | Observation |
|---|---:|---:|---:|---|
| Lucas 04 | 123 | 8 | 2.1 | The club rises to the left of the departing ball. Original points agree closely. |
| Andrew 04 | 122 | 3 | 121.2 | The ball moves left of the original track. The first common frame beyond 20 pixels is 111. |
| Eduardo 11 | 158 | 1 | 222.6 | Original point near the club at 156. The ball travels right, approaching the crop edge by 158. |
| Lucas 06 | 125 | 11 | 2.7 | Consistent with the visible ball. |
| Andrew 09 | 126 | 9 | 3.0 | Original locations agree. Refitting from hand points makes the carry estimate worse. |
| Eduardo 08 | 159 | 12 | 2.5 | Ball and club separate. Original points agree closely. |

Replacing six held-out inputs with hand-reviewed points changes overall carry MAE from 11.76 to 10.64 yd. Changed temporal coverage, sampled points and localization all contribute. A window-only control scored 11.66 using the original points, reverting to the full original fit on three reviewed shots with insufficient early support. This retrospective intervention is a diagnostic. The automatic comparison is independent of hand coordinates at inference.

## Automatic comparison

| Method | Carry MAE (yd) |
|---|---:|
| Fixed equal blend of video and video plus gyro | 11.76 |
| Inner-golfer fitted quality weights | 12.51 |
| Switch to alternate detector on low confidence | 22.47 |
| Recover a trajectory from cached candidates | 11.77 |
| Recover directly from image pixels, frozen regressors | 11.19 |
| Recover directly from image pixels, refit both regressors | 11.05 |
| Blend video model with stationary-ball-scale augmented model | 13.09 |

The quality gate uses early point count, same-frame agreement between two trackers and the alternate tracker's shaft conflict signal. It uses the video branch when club gyro is missing. The image recovery activates on fewer than six original launch points in the first 80 ms (5/57). It extracts positive brightness changes against a pre-impact background and scores trajectories originating at the stationary ball using each frame's timestamp. Candidates need at least five supported points, 30 ms of coverage and substantial displacement. The two regressors retain the existing projection descriptors. On frames with manual points the recovered median offset is 1.3 px across 12 frames for Andrew 04 and 2.7 px across six frames for Eduardo 11. Of the three additional recovered shots, frame inspection on Eduardo 05 shows the recovered track following the bright ball after separation even though carry fell from 114 to 60 yd against an actual 103. Thus coordinate improvement and carry improvement can diverge.

The refit improved Andrew (15.93 to 9.36 yd), Rohan (13.75 to 11.91) and Sid (7.80 to 5.57). Eduardo worsened (10.48 to 12.17) and Lucas worsened (16.13 to 18.88). Only three golfers contributed reviewed points and only two had sparse shots triggering recovery. The algorithm was developed after examining errors on this cohort. Freeze it before evaluating new players. The viewer continues to use its existing models.

## Camera calibration and speed

Six original MOVs have variable frame timestamps and QuickTime device metadata. Camera-to-ball pose, lens calibration at the recorded crop and an independent capture clock are unavailable. All 57 shots have a stationary-ball scale estimate. Multiplying initial image-plane velocity by nominal ball diameter divided by measured pixel diameter approximates a local transverse speed. It omits motion along the viewing ray. Adding that feature raised speed MAE from 6.44 to 8.02 mph. A learned ball-speed estimate trained on recovered projection descriptors scored 5.60 mph versus 14.13 mph for the training-set mean. The predictions CSV records a fold-specific error half-width estimated from the 80th percentile of absolute inner golfer residuals. This is empirical uncertainty for the supervised estimate, rather than a bound on physical speed.

For the next visit, lock the same camera, lens, zoom, stabilization setting, resolution and capture rate used for the swings. Before moving the phone, film a rigid checkerboard with measured square spacing at multiple tilts and depths. Then record it at the tee with a marked target-line direction, measure the camera center's height and offsets from the ball, and save an original clip of a timed flashing light to check the capture clock. Record the calibration again after any camera move. A synchronized second camera viewing from the side provides a depth measurement for early flight; a board at the tee alone does not determine ball depth in subsequent frames. Retain TrackMan ball speeds as held-out references. Compare reconstructed speeds and their sensitivity to point, timing and camera-pose errors on new golfers.

## Reproduce

Run these commands from the project directory, in order.

1. Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_launch_audit.py` using the existing caches and saved annotations.
2. Run `.venv/bin/python scripts/clean_launch_sheets.py` to regenerate the visual strips. Yellow indicates original detector points and cyan indicates the image-based recovery.

[Per-shot predictions](../experiments/clean_launch_audit/predictions.csv) · [Manual intervention](../experiments/clean_launch_audit/review_intervention.csv) · [Per-golfer metrics](../experiments/clean_launch_audit/per_golfer.csv) · [Review strips](../outputs/clean_launch_audit)
