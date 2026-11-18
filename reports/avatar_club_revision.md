# Avatar, club evidence, and travel counter — 2026-11-18

The user confirms the range scene renders, but reports malformed head geometry and a club that does not follow the actual golfer. This is a reconstruction problem, not an established caching problem.

## Changes delivered

- Head center now follows the ear midpoint instead of the nose. Ear/nose landmarks form a head orientation basis; the neck connects to the tilted head. Smaller skull/cap, smooth material normals, and an oval torso replace several misleading geometric assumptions. This remains a stylized estimated body, not a recovered human mesh.
- Removed the phase-driven club-direction template. There is no default invented club on shots without reviewed evidence.
- Added an assisted visual pilot for `sid_1_01`: 23 approximate AI grip/head readings from unmarked 540×960, 30-fps review frames. A second reviewer is pending. Short gaps of at most five frames are interpolated for display; the visibly stationary setup interval 0–30 is a separate explicit assumption. Eight frames remain unavailable; 97/105 have an anchor or interpolation. Cyan video overlays indicate anchors; gold dashed overlays indicate interpolation.
- Club orientation uses an orthographic display projection estimated from the pose, with scale constrained by the longest visible shaft. Both depth signs reproduce the same image displacement; a switch exposes the ambiguity. The fixed 0.85 display-unit grip-to-head length is an avatar assumption, not a metric calibration or precise Wilson-club measurement. Visual grip corrections also shift the rendered hands. Scientific pose/features are unchanged.
- Added an always-visible **Estimated travel · N yd** badge. It follows the same progress parameter as the schematic ball animation, supports seeking/replay, and ends at the nonnegative model carry estimate. It reports horizontal animated carry progression, not observed ball position, physical elapsed flight time, or curved path length. Missing predictions show unavailable instead of inventing a yard count.

## Automatic tracking experiment

Pose-assisted Hough shaft detection ran on the first shot of each of six sessions. Thin-line contrast, body-mask exclusion and static-edge rejection reduce some arm/fence failures, but do not establish a reliable tracker. Final proposal counts: Sid 44, Rohan 1 49, Rohan 2 76, Lucas 64, Andrew 78, Eduardo 86 out of 105 frames each. These are proposal counts, **not correct detections**. Intermediate Sid panels showed arm confusion at the top and background confusion near impact. The stricter Sid panel removes those examples but also loses visible setup shafts. The unreviewed automatic outputs do not drive the production avatar or any prediction model.

## Executed checks and remaining work

26 unit tests pass. Actual Three.js scene-graph math ran across 7,980 frames, including finite head/neck transforms, persistent legs, absent unsupported clubs, and travel-counter bounds/reset/endpoints. Assisted club tests check occlusion and long-gap abstention, source-input immutability, and identical image projections for both depth alternatives. Near-zero internal projection residual is a construction check, **not tracking accuracy**. Embedded HTML JavaScript syntax checks pass. Source overlays and software scene projections were inspected; these are not browser screenshots. Browser/WebGL visual and interaction QA remains unavailable.

Next: review native-cadence club keypoints across golfers, compare predictions against independent annotations, improve wrist/hand estimation and temporal tracking, and verify the actual browser rendering. The assisted one-shot pilot does not satisfy full automatic club reconstruction across all 76 swings.

## Acceleration clipping interpretation

Clipping is a serious limitation for the club contact channel, but current experiments do not isolate it as the main cause of total prediction error. All 68 recorded club windows clip; no matched unclipped control exists. On 65 labeled club-covered shots, video/watch carry MAE was 29.98 yd and adding club/contact features worsened it to 34.38 yd. Contact features increased sub-60-yard detection from 6/15 to 11/15 while false alarms rose from 7 to 10. Ball localization, hand confidence, timing, sample size, logging rate/gaps, and missing club tails remain separate limitations. See `contact_quality_results.md`; no new prediction experiment was run for this visual update.
