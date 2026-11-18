# Six reviewed-ball classification errors — 2026-11-18

Scope: held-out-golfer predictions for the 31-shot reviewed-ball pilot, fixed 100-yard threshold. This is a short-carry proxy, not confirmed physical mishit classification. No labels were changed.

| Shot | Excel row | Carry yd | Ball mph | Launch ° | Classification error | Club coverage |
|---|---:|---:|---:|---:|---|---|
| Andrew 1 / 11 | 57 | 83 | 82 | 23.4 | Accepted as 100+ | Missing tail |
| Eduardo 1 / 19 | 76 | 83 | 73 | 22.1 | Accepted as 100+ | Available; residual 14.7 ms |
| Rohan 2 / 7 | 31 | 125 | 102 | 11.0 | Flagged under 100 | Available; residual 28.4 ms |
| Rohan 2 / 10 | 34 | 106 | 107 | 4.3 | Flagged under 100 | Missing tail |
| Sid 1 / 16 | 17 | 81 | 75 | 20.7 | Accepted as 100+ | Available; residual 28.2 ms |
| Sid 1 / 18 | 19 | 121 | 100 | 11.7 | Flagged under 100 | Missing tail |

The manifest retains accepted user-confirmed ordinal associations and prior visible-swing review for all six. Available club/watch gyro residuals are small and missing tail recordings agree with the reported SD-card failures. These residuals concern gyro events, not exact physical impact or proof of correct TrackMan transcription. Missing club data cannot directly explain errors in this ball-only-feature classifier.

The three false acceptances all have higher launch angles and lower ball speeds; the three false rejections have lower launch angles and higher ball speeds. This is consistent with early projected rise being confused with distance. It does not prove the model's mechanism, and short early tracks do not resolve full 3D launch speed. Sid 16's short-shot score is 0.477, just below the 0.5 decision boundary.

A real timing/localization issue was previously found for Eduardo 19: its original departure proposal was early (381.763 s); the pilot uses the reviewed 381.95 s correction. Reinspection of the corrected native sheet confirms the strike/flight sequence is in the window. Rohan 2 / 10's sheet also contains the strike/flight sequence; its recorded low launch is qualitatively consistent with the low-flight review. This does not independently verify their carry labels. The model errors persisted after the pilot timing correction.

No evidence from these checks justifies shifting any label to a neighboring strike. A new independent count of every original strike and independent transcription review were not performed in this continuation. Source label photographs remain unavailable; definitive exclusion of label errors is therefore impossible. Do not choose a label offset because it makes model predictions look better.

Reproduce the numerical rows from experiments/carry_thresholds_v1/predictions.csv (reviewed_ball_assisted, threshold_yd=100), joined by shot_id to the manifest and original extracted labels. Machine-readable joined audit: misclassified_shot_audit.csv. The demo's Six classification errors filter links each shot to its reviewed native ball frames.
