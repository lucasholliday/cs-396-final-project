# Clean-cohort carry and rollout experiment

All five outer folds train exclusively on other golfers among the 57 accepted primary shots. Both Rohan sessions stay together. The frozen primary-model and display masks identify shots without filtering by distance. Carry, total and the two endpoint coordinates follow the viewer's yardage convention. The baseline uses the training shots' coordinate means. Each second-stage training example receives first-stage estimates fitted without its golfer or the outer test golfer. Model inputs use post-launch video. The club feature is the 95th percentile of recorded gyro magnitude, not clipped acceleration or measured clubhead speed. Missing features are imputed on the training fold.

| Recipe | Carry MAE | Total MAE | Mean landing miss | Mean final miss | Lateral rollout MAE |
|---|---:|---:|---:|---:|---:|
| Training mean | 43.00 | 31.19 | 47.87 | 38.39 | 4.07 |
| Current recipe refitted | 12.33 | 10.20 | 18.75 | 18.14 | 3.03 |
| Slice penalty and full tangent from landing | 12.34 | 10.21 | 18.75 | 22.55 | 8.00 |
| Video and gyro blend, direct side estimates | 11.76 | 9.62 | 18.25 | 17.86 | 3.03 |
| Blend, final-side anchor and training-fit turf factor | 11.76 | 9.62 | 17.71 | 17.86 | 2.74 |

Errors are shot-weighted yards. The proposed drag is an empirical residual penalty per yard of predicted rightward bend for these right-handed golfers. Its nonnegative coefficient minimizes absolute error on inner held-out predictions. A symmetric bend penalty also gave 12.34. These tests cannot identify the causal effect of spin axis at equal launch conditions.

The blend equally weights the existing video SVR and a smoother SVR using the same video features plus club gyro. The retained six-estimate model supplies forward rollout. For horizontal carry C, rollout R, launch azimuth A and directly predicted resting side F, the alternative solves for landing side L = (F + k R tan A)/(1 + 2 k R/C). This follows from the horizontal tangent of a quadratic flight and the relation F − L = k R (2 L/C − tan A). Depth is floored at one yard in the slope denominator. The vertical TrackMan landing angle measures descent and cannot supply horizontal heading. The turf factor minimizes training absolute lateral-shift error on recorded endpoints, bounded between zero and one. Across folds it ranged from 0.98 to 1.00. This is an empirical geometry coefficient. The fixed 0.22 alternative gave slightly better landing error (17.49) but worse lateral rollout error (3.44).

Fourteen current predictions roll opposite their predicted arrival heading, six by over one yard. The anchored geometry has zero opposite-heading shifts by construction. Among 41 shots with more than one yard of measured lateral rollout, side-direction agreement increased from 29 to 32. TrackMan endpoints alone cannot verify the intermediate bounce or roll path.

Carry MAE by held-out golfer (current to blend): Andrew 18.62 to 15.93 (6 shots), Eduardo 14.18 to 10.48 (14), Lucas 14.43 to 16.13 (11), Rohan 10.16 to 13.75 (8), Sid 8.49 to 7.80 (18). Repeated use of these golfers makes the best fixed recipe exploratory. A broader inner-golfer selector scored 13.33 carry MAE. Independent testing should freeze the equal-weight blend and turf recipe before collecting new golfers. Historical viewer scores use broader training cohorts.

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_clean_coupling.py`. [All variants](../experiments/clean_coupling_v1/metrics.csv), [per-shot predictions](../experiments/clean_coupling_v1/predictions.csv), [per-golfer scores](../experiments/clean_coupling_v1/per_golfer.csv), [training-fold selections](../experiments/clean_coupling_v1/selections.csv).
