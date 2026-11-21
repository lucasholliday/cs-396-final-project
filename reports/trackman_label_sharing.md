# Sharing TrackMan predictions across models

Five outer folds hold out one golfer, including both of Rohan’s sessions together. Each first-stage model predicts one of eleven labels. Every training shot used by the second stage gets first-stage estimates made without its golfer or the outer test golfer. Actual TrackMan labels are targets during training and references for scoring. The viewer’s abstention mask defines the 69-shot paired comparison. An additional comparison scores all 72 complete label rows, including research fallback predictions on the other three.

The independent experimental baseline changes the lateral predictors, using early video geometry and available club gyro features with missing sensor values imputed inside each training fold. Its resting lateral coordinate is directly supervised. The viewer instead illustrates resting side by extending the landing tangent. For a common 2D definition, use total yardage and resting side in yards for both; using the viewer’s exact rendered downrange position gives 30.06 yd instead of 29.27 yd. The CSV keeps the rendered comparison separately.

| Recipe on 69 shots | Carry MAE | Total MAE | Landing 2D | Resting 2D |
|---|---:|---:|---:|---:|
| Existing viewer | 15.59 | 14.14 | 24.38 | 29.27 |
| Independent first-stage models | 15.59 | 14.14 | 23.22 | 23.68 |
| Every label updates every output | 17.28 | 15.09 | 25.19 | 25.58 |
| Independent, enforcing total ≥ carry | 15.59 | 14.03 | 23.22 | 23.60 |
| Predict both rollout coordinates using all estimated labels | 15.59 | 13.14 | 23.22 | 23.41 |
| Predict forward rollout from six flight estimates, directly estimate resting side | 15.59 | 13.44 | 23.22 | 23.11 |

Errors are in yards. The six inputs are predicted carry, ball speed, launch angle, apex height, landing angle and hang time. The rollout target is recorded total minus carry. The estimated forward rollout is floored at zero for this dataset. Training only on predicted carry to estimate rollout gave 15.03 yd total MAE and 24.09 yd resting error. The six-label candidate on all 72 rows gave 16.27 yd carry MAE, 14.26 yd total MAE, 24.14 yd landing and 24.29 yd resting error, versus 16.27, 14.90, 24.14 and 24.81 respectively for the independent comparison. Choosing mixture weights in additional inner golfer folds produced 23.89 yd resting error on the 69.

The six-label candidate improved resting error over the matched independent models for Andrew, Rohan and Sid; it increased error slightly for Eduardo and Lucas. The 23.11-yard recipe is an exploratory choice among tested configurations. Another golfer or independent session is needed to confirm the gain. The six-label rollout and direct lateral estimates now drive Swing Lab. The earlier viewer predictions are retained for comparison. [Deployed cohort metrics](../experiments/label_stack_v1/deployed_cohorts.csv)

Reproduce with `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_label_stack.py`. Results include [overall and per-golfer errors](../experiments/label_stack_v1/metrics.csv) and [per-shot predictions](../experiments/label_stack_v1/predictions.csv).
