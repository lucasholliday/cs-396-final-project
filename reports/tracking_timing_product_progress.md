# Tracking, timing and product progress — 2026-11-18

## Executed tracking experiment

Compared the original v4 tracker with wide-field positive-contrast v6, dual-polarity v7, and v8 initialization constraints. The reason for v6 failure was visually confirmed: a white golf ball becomes a dark silhouette against the bright daytime sky. V7 detects positive and negative contrast separately. V8 requires a sufficiently large bright departure candidate near the mat before allowing either polarity in later flight.

| Method | Sparse point matches / 78 |
|---|---:|
| v4 final track | 41 |
| v6 final track | 24 |
| v7 final track | 49 |
| v8 final track | 52 |
| v7 candidate ceiling | 76 |

The 78 approximate AI-reviewed points cover 31 shots, selected from a deterministic 32-shot pilot. This is a repeatedly inspected development set, not independent validation. A match requires timestamp difference ≤6 ms and image error ≤32 pixels; references have stated 16-pixel uncertainty. No automatic trajectory was promoted to prediction input.

Candidate recall of 76/78 exposes the remaining selection problem: many candidate points belong to shafts, limbs, shadows or background. V7 has a median 143.5 candidates at reviewed frames. V8 improves linked-point recall to 52/78 but does not establish full-track precision. Even a track that matches all three sparse review points can contain wrong early or late points. Two Rohan 1 windows lack trustworthy initial seeds. Preserved overlays distinguish cyan reviewed points from orange linked proposals under outputs/ball_v7_review and outputs/ball_v8_review.

V7 candidate extraction ran on all 76 windows. V8 all-shot coverage is recorded in reports/tracking_progress/all_shot_coverage.csv. Track counts are proposal coverage, not validated successes. Candidate search still uses watch-assisted windows and a camera-specific region; it is not a general autonomous camera-only detector.

## Native timing review

Reviewed 12 native-cadence ball/club crops: first shot from each session plus six classification errors. Source-bound ball-departure intervals are saved in annotations/impact_intervals.csv, alongside review panel indices and notes. Bounds span 8.3–20.8 ms. A second reviewer is pending. Exposure blur and occlusion remain, and ball departure is not an exact ground-contact or club-ball-contact measurement. Original manifest impact fields and outcome labels remain unchanged.

The mapped wrist-gyro peak follows these departure midpoints by roughly 0.017–1.006 seconds, varying substantially by session. A universal wrist-peak-to-impact offset is therefore unsuitable. Automatic disappearance falls inside 11/12 reviewed intervals, a diagnostic observation on selected examples, not general timing accuracy.

Eduardo 19's earlier coarse 381.95-second reviewed departure is later than the new interval [381.9215, 381.9340]. This illustrates timing uncertainty in the original assisted features; it does not establish an incorrect shot identity. Prior model results are preserved, not silently recomputed with altered feature timing.

## Timing sensitivity and alternative flight features

Shifting the assumed departure 20 ms later nearly doubles the typical first-point projected rate (median factor 1.92). A 20-ms earlier departure reduces it to a median factor 0.68. These are feature-sensitivity calculations, not claimed prediction errors.

Tested departure-independent median pairwise slopes between airborne points on the 26 shots with at least two reviewed points. All feature sets use the same shots and nested held-out-golfer evaluation. Carry MAE: original first-point features 29.32 yd, inter-frame slope 38.55 yd, combination 32.65 yd. Launch-angle MAE: 4.95°, 6.26°, 5.39°. The alternative did not improve regression and is not promoted. The cohort contains only two Andrew and three Lucas shots, so comparisons are uncertain and visibility-selected. Full classification outputs and fold selections are in experiments/ball_timing_v1.

## Product changes and verification

All 76 swings now have confidence-masked head-centroid traces. Smoothing never bridges invalid runs, and the reference must be supported by the first ten review frames. Coordinates are image fractions; face rotation, perspective and landmark error can move the centroid. No gaze, physical head height or causal swing diagnosis is inferred.

The demo shows synchronized head traces, source-bound provisional departure navigation for the 12 reviewed swings, and jump buttons for saved user phase markers. Other phase buttons remain heuristic. The avatar remains a stylized rendering of estimated world landmarks.

Added isolated new-video analysis through scripts/analyze_swing.py and a loopback-only upload application via Start Swing Lab.command. The pipeline selects a user-specified 3.5-second window, estimates pose, produces H.264 replay and reference-model predictions, and saves output under outputs/new_swings. Research manifests and evaluation rows remain unchanged. Model files are trusted local artifacts checked against their saved hashes.

The new-video estimate uses a fixed ridge model trained on all 72 labeled research shots. It is not the existing held-out estimate and has no validated new-day accuracy. An existing recording was reused for pipeline smoke tests, which prove execution only, not generalization. HTTP upload, processing, result delivery, range requests, scope and request-validation checks passed. Browser interaction/visual QA remains unavailable and must not be reported as passed.

## Next checkpoints

1. Independently review departure intervals and full automatic tracks, especially wrong early/late segments.
2. Validate automatic seed detection for the Rohan 1 failures and measure track precision, not just sparse recall.
3. Evaluate richer trajectory hypotheses on a frozen, separately annotated set before promoting automatic ball features.
4. Verify the local app in a browser, then test genuinely new recordings and collect calibrated new-day/golfer data.
