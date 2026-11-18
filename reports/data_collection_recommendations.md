# Next data collection — evidence-based batch plan

## Recommendation
Plan **100 additional ball strikes**, collected in two 50-shot sessions on separate days, plus **20 explicitly tagged practice swings**. This is a practical next experiment, not a statistically derived guarantee of a given prediction accuracy. Review the first 50 before committing the second 50 to the same recording protocol.

Use ten golfers: the existing five plus five new golfers. Each golfer contributes five strikes per day (ten total). Keep seven iron initially. Within each five-shot block, request three normal swings, one deliberately reduced-speed swing and one comfortable faster swing. Keep every mishit and unregistered strike, with an explicit event number. Avoid forcing dangerous or unnatural swings. Do not substitute repeated shots only from a single golfer for the planned diversity.

Record two practice swings per golfer, separately tagged. A nearby real strike can provide context but does not turn a practice swing into an observed carry measurement. Practice-swing claims require a separately defined validation protocol.

## Why this batch
- Current usable outcomes: 72; matched club comparisons: 65; only five golfers from one day.
- Nested carry error remains ~25.5 yd within known sessions and ~29.9 yd for video+watch on held-out golfers. Severe short shots are frequently predicted as normal carries.
- Video ridge learning curves are nearly flat within known sessions from 25 to 50 training shots (~28.8 to ~28.5 yd), while held-out-golfer curves remain variable and improve from ~42.9 at 15 to ~34.1 at 50. More identical swings alone may not fix feature limitations.
- Every usable club window contains near-acceleration-limit flags (median 7.96% of window samples), and eight strikes lack club data due to SD-card disconnection.
- Ball and shaft tracking have promising single-shot demonstrations but no validated multi-shot accuracy yet. Measurement improvements have high expected value.

## Before the range
1. Secure the SD card/enclosure mechanically and run a recording/shake test. Preserve a completion summary and actual sample timestamps. Add high-G acceleration if available; retain overlapping low-G/gyro signals. Do not assume the high-G sensor alone fixes timing or coordinate calibration.
2. Measure actual club playing length and sensor-chip/enclosure coordinates relative to grip end and clubhead. User confirms standard Wilson Staff CB seven iron and sensor about one inch below lower grip edge. Nominal 37.25-inch length is a prior, not exact sensor lever arm.
3. Mark a fixed camera location, height and orientation. Capture a calibration board or measured reference at the ball/swing plane. Include full club path and early ball flight; test shutter/exposure and ball visibility before collecting the batch. A second approximately orthogonal camera is the most direct route to reducing monocular depth ambiguity if feasible.
4. Choose and record the actual high-speed setting per clip; retain original files and timestamp metadata. Validate frame timestamps and unique frames after transfer. Fixed nominal 240 fps is not by itself evidence of 240 unique samples per second.
5. Record a visible/audible sync gesture before and after each session, visible to camera and detectable by both motion sensors. Say/show each shot ID. Save actual TrackMan export if possible, including clubhead speed and spin targets when available. Preserve source photos and capture timestamps for transcription review.

## At the range
Take one pilot strike per golfer and inspect the recording before proceeding. Keep mat/camera unchanged within a block. Record session/date/golfer/handedness/club/watch wrist/sensor orientation and any remounting. Keep the sensor configuration consistent across normal/reduced/faster swings. Do not delete embarrassing or unregistered shots.

## Checkpoint after 50 strikes
Proceed only after matching all actual strike IDs to video/watch and available club prefixes, verifying original video timing, measuring per-shot clipping and checking representative ball/club tracking overlays. If SD failures, invisible ball flight or missing label exports recur, fix acquisition before collecting the second half.

After 100, evaluate the new days separately, hold out the five new golfers, rerun matched-cohort sensor ablations and compare severe-mishit errors. Collect another 50 only if the latest two training-size increments show consistent improvement across held-out golfers/days or additional acquisition targets a documented failure mode. Do not promise a sub-10-yard system from these learning curves.

Detailed observed metrics and variability: `model_results.md`, `experiments/baseline_v1/learning_curves.csv`, `experiments/nested_carry_v1/metrics.csv`.
