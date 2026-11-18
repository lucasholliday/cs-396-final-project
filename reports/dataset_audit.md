# Initial dataset audit — 2026-11-17

79 source files / 11,976,204,529 bytes (excluding `.DS_Store`), SHA-256 inventoried. Six iPhone MOVs, six watch sessions, six club sessions, one workbook. Five named golfers. No byte-identical recording duplicates; duplicate metadata/manifests/tags only. Originals unchanged.

Follow-up checkpoint: all 79 original SHA-256 values were rechecked successfully. The 76 swing candidates now have accepted identity/order annotations based on user confirmation, stable watch event counts and visual contact-sheet review; 72 are eligible for whole-swing outcome benchmarks. Exact impacts and tracking localization remain unverified. See `model_results.md` for executed benchmarks and `checkpoint_verification.json` for integrity checks.

## Associations
User confirms nth actual strike in each named session corresponds to nth Excel row, including unregistered strikes. No practice swings. Devices started before first strike; club may stop early due to SD-card removal. All golfers right-handed, seven iron, watch on left lead wrist. Manifest records this rule but per-shot physical timestamps/counts remain unverified. Do not force detector counts or shift after missed events.

| Session | Rows | Video playback s | Watch span s | Club span s | Accel-limit rows |
|---|---:|---:|---:|---:|---:|
| Sid 1 | 18 | 424.110 | 407.488 | 368.828 | 452 |
| Rohan 1 | 5 | 126.238 | 136.414 | 91.158 | 83 |
| Rohan 2 | 10 | 266.893 | 233.875 | 206.618 | 170 |
| Lucas 1 | 12 | 261.353 | 230.650 | 242.191 | 319 |
| Andrew 1 | 11 | 257.043 | 278.956 | 246.114 | 296 |
| Eduardo 1 | 20 | 408.490 | 426.455 | 431.150 | 732 |

Durations are independent clocks. Exact paths are in `data/manifests/shot_manifest.csv`. Subsequent affine gyro alignment finds 68 club-covered events with residuals below 36 ms; eight missing tail events match the reported SD-card failures. 65 club-covered events have labels. This aligns swing neighborhoods, not exact impacts.

## Labels
All three workbook sheets read. 72 registered rows, three unregistered and one stale screen. Missing numeric outcomes: rohan_1_03 (Excel row 22), rohan_2_09 (33), andrew_1_05 (51), andrew_1_06 (52). Preserve candidates and exclude absent targets. Available: carry/total yards, ball speed mph, launch/landing angles, direction, curve/height/sides feet, from-pin yards and hang time seconds. Signed direction/side columns are right-positive. No clubhead speed, spin rate or spin axis. Source photographs (76 references) are absent, so independent transcription verification is unavailable. Preserve workbook supplemental corrections for sid_1_08, rohan_2_05, lucas_1_09 and original review statuses.

## Video
HEVC stored at 1920×1080 with rotation metadata yielding portrait 1080×1920 display, with two audio streams each. Nominal 240/1 fps, average playback ~159.0–159.6 fps. Packet duration uniformly 0.004167 s but median sorted PTS spacing ~0.008333 s, largest ~0.016667 s. User recalls mixed 120/240 capture, transferred through Drive. Neither frame index / 240 nor average FPS is established physical time. First/last swing video checks against watch epoch support approximately real-time PTS, while exact sub-frame synchronization is still uncalibrated.

Packet counts: Andrew 40,960; Eduardo 65,183; Lucas 41,654; Rohan 1 20,094; Rohan 2 42,446; Sid 67,548. Complete timestamp caches preserved. Sid original creation tag: 2026-11-15T16:03:06-0600; container creation: 2026-11-16T02:59:46Z; full-frame-rate-playback-intent=0. Container creation is not recording/impact time.

Decoded 18 windows (0.25 playback s at 10/50/90% of each clip), 38–40 frames each, zero exact adjacent duplicates. Six contact sheets visually inspected. Golfer/mat/club visible; background people/occlusion and lighting variation. No complete-strike count, ball trajectory, gaze, pose, calibration or physical speed validated. Sparse checks do not establish full-file integrity or absence of near-duplicates. Contact sheets `outputs/audit/*_contact.png` read left-to-right 10/50/90%; exact sample times in `sample_decode_results.json`.

## Sensors
All CSVs parse. WristMotion: epoch time, elapsed seconds, rotation rate, gravity, acceleration, quaternion and Euler angles. Increasing timestamps, ~99.23 Hz, no gaps >1.5× median in six wrist streams. Units/axes still technically unverified. Sid epoch corresponds to ~16:03 Chicago while folder says 22:03; do not treat folder text as local clock.

Club metadata: ISM330DHCX ±16g, ±4000 degrees/s range; m/s² acceleration includes gravity, gyro in rad/s, 200 Hz software polling, 208 Hz ODR, no FIFO, relative monotonic nanoseconds. No ADXL375 stream. Median interval ~5.005 ms; mean 188.57–188.74 Hz. Maximum gaps 28.75–39.76 ms. Gaps >1.5× median: 4,369 / 1,079 / 2,447 / 2,870 / 2,915 / 5,107 in table order. These are timing gaps, not missing-strike counts.

2,052 near-acceleration-limit rows, zero near-gyro-limit rows. First flagged times: Sid 7.621, Rohan 1 35.635, Rohan 2 30.673, Lucas 36.383, Andrew 24.346, Eduardo 1.406 seconds. User confirms range-limited acceleration. Quantify per shot after synchronization; flagged peaks are not trustworthy true maxima. Only Eduardo has summary complete=true. User confirms valid recorded prefixes for the other five, followed by SD-card disconnection.

Rohan 2 `WatchMicrophone.mp4` fails ffprobe (`moov atom not found`); motion remains readable. Other watch audio durations exceed wrist spans; audio-clock relationship unverified. Quarantine corrupt audio only.

## Reproduction and remaining work
`python3 scripts/audit_dataset.py`; `python3 scripts/sample_videos.py`; `python3 -m unittest discover -s tests -v` (three tests passed).

Evidence: `data/manifests/` initial seven CSV/JSON outputs plus event/alignment additions; `outputs/audit/` caches/samples. Follow-up event/feature/benchmark work is documented in PROJECT_STATUS and EXPERIMENT_LOG. Remaining audit tasks concern exact impacts, frame-level localization, physical clock/calibration and watch units. Do not confuse accepted whole-swing identity with validated metric kinematics.
