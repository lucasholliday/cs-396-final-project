# Ball motion cutoff and early launch windows

I tested the proposed three-point speed cutoff both as an absolute floor of 1,000 source pixels per playback second and as a relative floor of 20% of the highest three-point speed seen so far during the first 80 ms. The combined rule stops at the first violation of either threshold. Speed is net source-image displacement from the first to third point divided by their recorded timestamp interval. Windows crossing a gap longer than 35 ms are skipped. The optional linker gate returns immediately after a violation. The reported comparison replays identical candidate paths so it isolates the effect of truncation.

## Effect on the tracker

The reference was the same set of 76 wide-crop, 35 ms gap windows from the previous extension experiment.

| Rule | Detections retained | Tracks shortened | Existing approximate reference points within 16 px |
|---|---:|---:|---:|
| Before, no speed cutoff | 3,326 | 0 | 43 / 78 |
| 1,000 px/s | 1,211 | 39 | 39 / 78 |
| 20% of launch peak | 899 | 39 | 30 / 78 |
| Either threshold | 882 | 39 | 30 / 78 |
| Three-point stall test, 180 px/s and all points within 2 px | 3,262 | 6 | 43 / 78 |
| Sustained display gate | 3,326 | 0 | 43 / 78 |

On Sid 08, using the same 35 ms association rule for all variants, the original 111-point track fell to 32 points with the absolute floor and 17 with the combined floor. The rejected native frames still show the ball moving steadily through the sky. The three-point stall test also removed visible ball motion on Sid 08, Rohan session 2 shot 01, and Rohan session 1 shot 01. Perspective shrinkage, a shallow apparent arc and pixel quantization can reduce image speed long before physical flight ends. Source pixel speed cannot distinguish a ball slowing in image space from a stationary background object on its own. [Sid 08 speed cutoff review](../outputs/ball_motion_review/sid_1_08_wide35_either.jpg)

## Effect on carry and endpoint prediction

All rows use the same 72 labeled shots and hold out one golfer at a time, with per-target estimator selection inside grouped training folds. The first row reproduces the earlier saved prediction recipe, including its older geometry cache. The wide-track rows recompute features from the current candidate paths and are the controlled comparison for changing time windows or the speed threshold. A smaller number of usable launch fits uses the established training-only fallbacks; the shots remain in the score.

| Model input | Usable launch fits | Carry MAE (yd) | Landing 2D miss (yd) | Resting 2D miss (yd) |
|---|---:|---:|---:|---:|
| Previous saved early geometry | 58 | 16.27 | 25.19 | 26.94 |
| Wide tracks, 150 ms, no cutoff | 66 | 21.91 | 28.51 | 28.78 |
| Wide tracks, 150 ms, 1,000 px/s | 66 | 21.61 | 28.28 | 28.65 |
| Wide tracks, 150 ms, either threshold | 66 | 21.75 | 28.41 | 28.67 |
| Wide tracks, 80 ms | 62 | 21.56 | 28.27 | 30.69 |
| Wide tracks, 60 ms | 47 | 31.84 | 37.29 | 33.15 |
| Wide tracks, robust linear motion over 80 ms | 67 | 28.26 | 33.06 | 32.05 |
| Wide tracks, robust linear motion over 60 ms | 65 | 28.15 | 32.71 | 33.63 |

The 80 ms window improved carry by 0.35 yd and landing by 0.24 yd versus the 150 ms fit on the same current tracks, while worsening the resting estimate by 1.91 yd. Carry on the 44 shots that actually exceeded 100 yd improved from 18.36 to 16.42 yd MAE in this comparison. The 60 ms window substantially degraded all-shot carry. None of these recipes improved the corresponding error of the frozen earlier model. The hard speed gate gave identical input features in both the 60 and 80 ms experiments because its first drop in this data occurred after 80 ms.

The timing windows are elapsed source playback time relative to a proposed launch seed, rather than fixed frame counts. Median usable detections among the 73 tracks were six in 60 ms and eight in 80 ms, reflecting actual cadence and missing detections. Native impact timing and physical capture time are still approximate. Early points can still follow the club instead of the ball, as the prior review of Lucas 02 demonstrated.

## Viewer behavior and verification

Kept the new speed-cutoff recipes in the benchmark. Open Swing Lab uses a display-only terminal check for at least six consecutive detections staying within 1.5 px of the initial point for at least 45 ms. If triggered, the active yellow tracer clears from the first dropped point. The prior 35 ms timeout on missing observations remains in place. The sustained stationary check triggered on zero of the 76 current viewer paths, so **2,970 displayed detections before and after**. Synthetic verification exercises the new stationary branch. On real frames it is preferable to the aggressive cutoff, which removed visible flight. This pass did not establish a current shot where a path sticks to a ceiling lamp.

All 45 Python tests passed. The rebuilt HTML passed JavaScript syntax and 76-shot delivery checks. Browser interactions were not exercised. Source recordings and labels were read only.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_ball_motion_gate.py
.venv/bin/python scripts/build_research_viewer.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_research_delivery.py
```

Per-shot decisions, sparse reference checks, forecasts and metrics are in [the experiment folder](../experiments/ball_motion_gate_v1).
