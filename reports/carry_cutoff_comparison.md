# Carry errors above 70 and 100 yards — 2026-11-18

User requested the same six-method comparison for shots over 70 and over 100 yd. Cutoffs are strictly greater than, not narrow distance bands. Matched base cohort: 65 labeled shots with club data. Above 70: 50 shots/five golfers. Above 100: 39 shots/four golfers (Andrew has none). All external splits hold out entire golfers; filtering does not shift associations.

MAE is mean(abs(predicted carry − actual carry)), in yards. Overestimates and underestimates do not cancel. It is not a guaranteed error bound or confidence interval.

## A. Existing all-shot models, scored on the selected shots
Training and saved out-of-fold predictions are unchanged. The mean baseline is learned from all eligible training shots, including short shots.

| Method | All 65 | >70 yd (50) | >100 yd (39) |
|---|---:|---:|---:|
| Training-set mean | 41.60 | 31.05 | 35.58 |
| Watch | 35.93 | 28.35 | 31.76 |
| Body video | 31.15 | 26.80 | 30.42 |
| Video + watch | 29.98 | 23.29 | 26.57 |
| Video + watch + club gyro | 28.58 | 18.96 | 21.36 |
| Above + contact descriptors | 34.38 | 30.75 | 36.02 |

## B. Newly fitted models restricted to above-threshold shots
Both training and evaluation exclude actual below-threshold outcomes. The mean baseline now uses only above-threshold training shots. Model families (mean/ridge/Extra Trees/SVR) are selected in grouped inner training folds, preserving prior settings. No held-out golfer chooses its own model. At >100 yd only four golfers remain, so each outer training set has three golfers.

| Method | >70 yd (50) | >100 yd (39) |
|---|---:|---:|
| Training-set mean | 22.19 | 11.98 |
| Watch | 32.19 | 12.11 |
| Body video | 30.02 | 11.98 |
| Video + watch | 28.76 | 12.76 |
| Video + watch + club gyro | 31.87 | 12.80 |
| Above + contact descriptors | 25.21 | 13.71 |

The restricted training mean beats or matches every restricted-trained model. Above 100 yd, the video model selects the mean in every outer fold; it supplies no added within-category distance discrimination in this experiment. Nested selection can still perform worse than a fixed mean on held-out golfers: tiny training groups make model selection unstable. This is a measured failure, not a reason to choose outer-fold winners retrospectively.

The existing all-shot fusion model is better than the conditional mean on the >70 subset (18.96 vs 22.19), but worse on >100 (21.36 vs 11.98). These are different training strategies, not contradictory results. All subgroup evaluations use actual carry to decide eligibility. An application does not know that membership in advance; no automatic classification/gating improvement follows. These exploratory evaluations reuse the same data and do not establish an independently validated winning method. Cutoffs are not verified contact labels.

Reproduce: `.venv/bin/python scripts/carry_cutoff_comparison.py`. Metrics, restricted out-of-fold predictions, inner selections and configuration are saved in experiments/carry_cutoff_comparison_v1. Original model outputs remain unchanged.
