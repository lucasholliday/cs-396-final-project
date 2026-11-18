# Ball flight and club contact study — 2026-11-18

## Question and labels
The user asked whether visible early ball flight and specific club-sensor patterns can expose mishits that whole-body movement misses. The user confirms short-carry shots are useful broad mishit examples but cannot supply exact chunk/thin/hosel labels. We therefore test carry below 60 yd, launch angle below 10°, and (in the reviewed-ball pilot) absolute initial launch direction above 10°. These are observable outcome proxies, not contact-mechanism ground truth. Lateral launch is not the same as later curvature/slice.

## Club sensor: what was actually tested
Extracted 16 new descriptors from 68 recorded club events, of which 65 have valid outcomes. A fixed [-0.4,+0.4] s window around the club gyro peak includes:
- Gyro high-pass RMS/peak relative to swing amplitude, central versus outer-window energy, transient peak count and timing.
- Gyro derivative, deceleration, angular-axis direction change, and post/pre-peak motion ratios.
- Acceleration-limit fraction, run count, longest run, temporal centroid and post/pre imbalance. These describe censoring, not actual peak acceleration.
- Acceleration derivative only between adjacent **unclipped** samples, plus the fraction of such pairs. This can still have selection bias from clipping; it is a separate ablation, not a recovered impact-force measurement.

Signals are resampled at 180 Hz for a 20-Hz high-pass diagnostic. Native logging averages about 189 Hz and contains gaps; this does not recover missing high-frequency collision detail. The spike can include club movement, sensor/mount vibration and filtering effects. A gyro peak is not independently annotated impact. No claim is made that a spike proves ground-before-ball contact.

Executed 36 nested comparisons: six feature sets × two fold protocols × carry regression / short-carry classification / low-launch classification. Inner training folds choose the model; classification uses log loss and a fixed 0.5 decision threshold. All modalities share shot folds.

## Sensor results
On the same 65-shot held-out-golfer cohort:
- Video + watch carry MAE: **29.98 yd**. Adding the club baseline and all contact descriptors: **34.38 yd**. Contact descriptors did not improve distance regression.
- Sub-60-yard classification: video + watch detects **6/15** short shots with **7** false alarms (precision 46%). Adding club/contact detects **11/15**, with **10** false alarms (precision 52%). AUROC rises from 0.647 to 0.812, but probability calibration worsens (Brier 0.183→0.199), and precision-recall area falls (0.526→0.485). This is mixed evidence, not a universally better classifier.
- Club-only baseline and contact variants generalize poorly across golfers. Clipping alone is not a reliable chunk detector: every usable window clips, including normal shots.
- Low-launch classification from body/sensor features remains poor. Exact chunk/thin classification was not evaluated because those labels do not exist.

Paired waveform illustrations in outputs/contact_quality/waveform_pairs.png show why summary swing speed can fail: normal and short shots can have similar overall gyro curves with different brief transients. Such examples motivate the test but cannot establish a universal diagnostic rule.

## Reviewed-ball feasibility pilot
Selected **32 shots by a fixed ordinal rule**—every third shot starting at one, plus the final shot in each session—rather than choosing shots based on outcomes. Exported six unmarked native-frame panels for each, then inspected all sheets and difficult native crops. Recorded **78 approximate airborne-ball centers across 31 shots**, with 16-pixel annotation uncertainty and source hashes. One shot remained ambiguous. Five initial departure windows needed coarse visual correction; the original frames were archived. Later Eduardo mat-ball positions lie outside the original narrow seed ROI, exposing a concrete automatic-tracker failure.

Annotations are AI visual estimates from unmarked frames; the reviewer had prior dataset context, so this is not blinded external ground truth. A second reviewer is outstanding. Only the reviewed sparse points enter this separate benchmark—unreviewed automatic trajectories remain excluded. Three features describe normalized image rise/lateral displacement divided by estimated elapsed time, and elapsed time itself. These are projection features, **not meters/second or physical launch angles**. The earliest confidently visible point is used; no point is invented after the ball is lost.

Executed 24 nested held-out-golfer comparisons on the same respective cohorts. On all 31 reviewed shots:
- Carry MAE: body video + watch **37.36 yd**, reviewed ball **24.78 yd**, combined **27.92 yd**.
- Launch-angle MAE: body video + watch **8.65°**, reviewed ball **4.61°**.
- Sub-60-yard classification from reviewed ball: **9/10** short shots found, **3** false alarms; precision **75%**, recall **90%**, AUROC **0.900**.
- Low-launch classification: **6/7** found, **4** false alarms; precision **60%**, recall **86%**, AUROC **0.935**.

These results support investing in reliable ball localization. They do not establish current end-to-end automatic accuracy or validated physical flight extrapolation. The subset is small, visibility-selected and scene-specific. The matched-club subset has only 25 shots and gives different results: ball + club/contact improves short-shot detection from 6/9 to 8/9 but increases carry MAE from 24.46 to 33.75 yd. That instability reinforces the need for new data and independently reviewed points.

The user’s suggestion to use only longer shots is evaluated separately in reports/carry_threshold_results.md. It distinguishes conditional range restriction from a classifier that must identify the category first.

## Sources and physical limits
The [ST ISM330DHCX datasheet](https://www.st.com/resource/en/datasheet/ism330dhcx.pdf) describes the device's configurable output rates and filter chain. The actual collected configuration/log timing, not the maximum device capability, constrains this analysis. A higher-range accelerometer and faster, timestamped buffered acquisition could improve future impact measurements, but neither creates retrospective samples or guarantees chunk/thin discrimination.

A short 2D flight trace can improve empirical outcome prediction while still leaving depth, camera geometry, spin and aerodynamic parameters unresolved. Losing the ball is not observing landing. Distance here is learned from training-shot TrackMan labels and tested on held-out golfers; it is not a claimed physics reconstruction after disappearance.

## Reproduce
Run scripts/contact_features.py, contact_benchmark.py, contact_diagnostics.py, contact_video_review.py, reviewed_ball_features.py and reviewed_ball_benchmark.py using the project venv. Reviewed annotations and timing corrections are persisted under annotations/. Results and every OOF prediction are in experiments/contact_quality_v1/ and experiments/reviewed_ball_v1/. The original 72-shot benchmark remains intact.
