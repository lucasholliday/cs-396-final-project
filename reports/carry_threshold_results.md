# Carry thresholds and separate short-shot flags — 2026-11-18

Tested 80, 100 and 120 yards with leave-golfer-out evaluation. Four feature/cohort variants were evaluated at each cutoff. Model families are chosen using only the outer training golfers via inner grouped folds. The gate is also trained only on training golfers; its below-threshold probability is compared with a fixed 0.5 cutoff. All out-of-fold predictions and decisions are saved. These are exploratory tests on a repeatedly studied dataset, not a new untouched test set.

**A carry cutoff is not a contact-quality label.** A solid slower seven-iron can carry under 100 yards, and a badly directed or thin shot can exceed it. This study calls the outcomes above/below threshold; it does not infer confirmed chunks, thin strikes or hosel contact.

## What the columns mean
- `oracle_above_mae`: model trained on above-threshold training shots, scored only on test shots whose actual carry exceeds the cutoff. This assumes the qualifying category is already known.
- `oracle_above_constant_mae`: training-above-threshold mean on exactly the same qualifying test shots. Necessary because narrowing the range alone lowers MAE.
- `accepted_mae`: high-distance model error on the shots the classifier actually accepts, including mistakes. The system may flag rejected shots rather than output a distance.
- `gated_all_mae`: all-shot error if below/above-specific regressors are routed by the predicted gate. No test shot is removed. `allshot_mae` is the corresponding unfiltered regression comparator trained/evaluated on the same cohort.
- Reviewed-ball features use approximate AI visual point annotations and five coarse timing corrections; they estimate assisted-tracking potential, not current automatic-tracker accuracy. No unreviewed detector trajectories enter these models.

## Results
| features | threshold_yd | n | true_above_n | oracle_above_mae | oracle_above_constant_mae | accepted_n | accepted_mae | false_accepted_below_n | false_rejected_above_n | allshot_mae | gated_all_mae |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reviewed_ball_assisted | 80 | 31 | 21 | 21.36 | 19.37 | 19 | 24.59 | 1 | 3 | 24.78 | 23.61 |
| reviewed_ball_assisted | 100 | 31 | 17 | 13.07 | 12.15 | 17 | 20.0 | 3 | 3 | 24.78 | 25.65 |
| reviewed_ball_assisted | 120 | 31 | 12 | 10.6 | 8.98 | 9 | 27.19 | 6 | 9 | 24.78 | 34.41 |
| video | 80 | 72 | 53 | 27.92 | 18.7 | 45 | 21.83 | 4 | 12 | 31.93 | 29.65 |
| video | 100 | 72 | 44 | 11.98 | 11.98 | 21 | 28.93 | 5 | 28 | 31.93 | 48.28 |
| video | 120 | 72 | 31 | 10.26 | 10.23 | 17 | 24.14 | 7 | 21 | 31.93 | 34.9 |
| video_watch | 80 | 72 | 53 | 26.14 | 18.7 | 45 | 26.24 | 5 | 13 | 29.87 | 35.4 |
| video_watch | 100 | 72 | 44 | 12.86 | 11.98 | 24 | 31.74 | 7 | 27 | 29.87 | 48.61 |
| video_watch | 120 | 72 | 31 | 11.63 | 10.23 | 11 | 31.7 | 5 | 25 | 29.87 | 39.35 |
| video_watch_club_contact | 80 | 65 | 47 | 22.75 | 19.32 | 41 | 26.68 | 5 | 11 | 34.38 | 34.59 |
| video_watch_club_contact | 100 | 65 | 39 | 13.71 | 11.98 | 39 | 24.61 | 7 | 7 | 34.38 | 31.2 |
| video_watch_club_contact | 120 | 65 | 28 | 12.57 | 10.79 | 17 | 18.93 | 4 | 15 | 34.38 | 33.51 |

## Interpretation
At 100 yards, 44 of the 72 labeled shots qualify. Video gives 11.98 yd conditional MAE; video + watch 12.86. The conditional training-mean baseline is 11.98 yd: much of the apparently impressive accuracy comes from restricting the outcome range. The data do not show accurate within-category distance discrimination beyond that simple baseline.

Operational classification is substantially harder. With video + watch at 100 yards, only 24/72 shots are accepted; seven of those are actually below 100 and 27 qualifying shots are rejected. Accepted MAE is 31.74 yd. Hard routing across all shots worsens MAE from 29.87 to 48.61 yd.

On the separate 31-shot reviewed-ball cohort, 17 actually exceed 100 yards. The gate accepts 17: 14 correct accepts, three false accepts; it rejects three qualifying shots and correctly flags 11 shorter shots. Accepted MAE is 20.00 yd, compared with 13.07 yd if qualifying membership is known perfectly. All-shot hard routing is 25.65 yd versus 24.78 for direct reviewed-ball regression. Thus an automatic distance upgrade has not been demonstrated.

At 80 yards, assisted-ball hard routing is 23.61 yd versus 24.78 direct; video routing is 29.65 versus 31.93. These small exploratory differences should not determine the cutoff. At 120 yards, the qualifying sample shrinks, the constant baseline becomes even stronger, and several gates reject many legitimate qualifying shots. Group-cluster uncertainty is persisted; five golfers and multiple comparisons limit confidence.

A reasonable product experiment is to provide a short-shot flag and report distance only when the model has enough evidence, showing acceptance rate and false acceptance explicitly. Choose a cutoff based on the intended seven-iron use case and golfer population, then freeze it before new-day/new-golfer testing. Do not call every sub-100-yard shot a mishit or present 12-yard conditional error as end-to-end accuracy.
