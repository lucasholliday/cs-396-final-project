# Watch speed/acceleration versus carry — 2026-11-18

Executed descriptive analysis of all ten existing watch summaries against carry on 72 labeled shots. Whole-swing magnitudes are not impact-specific clubhead velocity; export units remain unverified. These are exploratory correlations on existing data, not additional held-out predictive validation.

| Feature | Pearson r, all shots | Spearman rank correlation | Within-golfer demeaned Pearson r |
|---|---:|---:|---:|
| Peak wrist rotation magnitude | -0.099 | -0.044 | -0.152 |
| 95th percentile rotation magnitude | -0.401 | -0.388 | -0.189 |
| Integrated rotation magnitude | -0.479 | -0.499 | -0.306 |
| Peak acceleration magnitude | -0.409 | -0.387 | -0.016 |
| 95th percentile acceleration magnitude | -0.079 | -0.094 | +0.052 |
| Acceleration peak width | +0.449 | +0.374 | +0.060 |

The apparent positive peak-width relationship largely disappears after subtracting each golfer's mean. None of these statistics establishes a robust faster-wrist→longer-carry rule. Negative correlations do not mean slowing down causes longer shots. Golfer/session differences, contact outcomes, whole-window timing, units and possible sensor limiting constrain interpretation. Within-golfer demeaning is descriptive, not a causal adjustment.

Among 44 shots already known to carry at least 100 yd, peak rotation rank correlation is +0.027 and peak acceleration -0.052. This selected cohort is not a confirmed well-struck label set or a deployable filtering rule.

Concrete examples (recorded magnitude units, not mph): Lucas shot 8 carries 158 yd with peak rotation 47.85 and peak acceleration 23.19; Lucas shot 10 carries 16 yd with respective peaks 49.80 and 24.01. Rohan session 2 shots 1 and 2 carry 132 and 10 yd with peak rotation 38.07 and 37.52. These examples illustrate ambiguity, not its proven cause or independent confirmation of exact temporal alignment.

A raw-file check across all six watch sessions finds per-axis rotation extrema around 32.44–35.35 recorded units, with values at/above absolute 34.89 on 13–54 rows per recording. This repeats the earlier potential watch-range limitation finding; no saturation flags or verified exporter/device range are available. It does not prove hard clipping. Club accelerometer saturation is separately established.

Next useful experiment: establish individual impact timing, then compare short-window pre-impact axis-specific wrist rotation/acceleration, change near contact, and relative tempo against these whole-window summaries using the same held-out-golfer protocol. Validate units and possible watch limiting. Combine verified club/ball observations to separate swing effort from realized ball outcome. Do not double-integrate uncorrected wrist acceleration and label it clubhead speed.

Reproduce: `.venv/bin/python scripts/watch_carry_relationships.py`. Full ten-feature and per-golfer outputs are in `experiments/watch_carry_relationships_v1/`.
