# Native-frame club search across the recorded swing windows

Each of the 76 shot windows starts at its pose clip timestamp and spans 3.5 playback seconds. FFmpeg decodes every available frame in that interval from the original MOV, including frames between the existing 30 fps pose estimates. The detector uses the video timestamps and interpolates wrist positions only to center its shaft search. The median frame spacing is 8.33 ms and the 10th percentile is 4.16 ms. Capture speed and playback speed can differ.

The first detector scores straight edges near the hands. The second adds a body exclusion region and requires a segment extending into open space. Both require neighboring frames to support a direction. They run on 270 × 480 images. A proposed segment can cover only part of the shaft. Its end does not establish the position or speed of the clubhead.

## Direction checks

Sid 01 already had approximate grip and head markings on 23 review frames. They were matched to the original video timestamps with less than a millisecond discrepancy. Lucas 02 was sampled every 30 native frames starting at frame 10. I estimated shaft directions from the unmarked images before inspecting the detector output for that shot. One of 19 frames was too ambiguous to score. Six samples show almost the same address pose. Both sets are approximate AI visual readings. The accuracy criterion is an oriented grip-to-head direction within 10 degrees of the reading. A mismatch by 180 degrees counts as an error.

One failure is visible in [the Sid 01 image comparison](../experiments/native_swing_shaft_v2/sid_visual_check.jpg). At frames 2 and 311 the detected line follows the horizontal range background. A pilot check on Sid comparing the line against a median image of the same view rejected five of six wrong directions, retained all five close directions and left 12 frames without an initial match. This check was developed on that shot.

The higher-resolution pilot searches at 540 × 960, uses background images from the beginning and end of the window, and keeps several plausible lines per frame before checking neighboring directions. All angles and missing results for the two review swings are in [the paired comparison](../experiments/native_swing_shaft_v2/pilot_comparison.csv). The precision check evaluates the image-plane shaft direction. It does not evaluate depth, exact clubhead position, physical speed or distance prediction.

## Reproduction

From the project directory run the following. The full-window scans read the original local videos. The sheet and Sid background pilot use the existing native-frame review cache. The high-resolution pilot uses the first detector's frame timeline.

```sh
.venv/bin/python scripts/native_swing_shaft_v1.py
.venv/bin/python scripts/audit_native_swing_shaft.py v1
.venv/bin/python scripts/native_swing_shaft.py
.venv/bin/python scripts/audit_native_swing_shaft.py v2
.venv/bin/python scripts/shaft_review_sheet.py
.venv/bin/python scripts/shaft_background_check.py
.venv/bin/python scripts/native_shaft_highres.py sid_1_01 lucas_1_02
.venv/bin/python scripts/compare_native_shaft_pilots.py
```

## Comparison at the marked frames

Each cell gives the number within 10 degrees, the number with a detection and the number of scoreable frames, respectively.

| Automatic search | Sid 01 | Lucas 02 |
|---|---:|---:|
| Initial search | 0 / 5 / 23 | 1 / 13 / 18 |
| Body exclusion and temporal agreement | 5 / 11 / 23 | 3 / 11 / 18 |
| Higher resolution with background checks | 10 / 17 / 23 | 5 / 11 / 18 |

The higher-resolution pilot improved total close matches from 8 to 15 across the two swings. Three of its five close matches on Lucas 02 occurred during address. It still missed visible shafts on Lucas and selected an almost reversed direction at frame 190. The body-filter results for all shots are in [the per-shot coverage table](../experiments/native_swing_shaft_v2/coverage.csv).

Across all 76 windows the body-filter detector returned 19,507 proposals from 42,367 decoded frames (46.0%). Every shot produced proposals. Of those proposals, 11,524 occur in the first third, 4,990 in the second and 2,993 in the final third. Sustained address positions contribute many consecutive detections.

## Five marked clubheads on Lucas 02

I marked the visible clubhead on five frames (10, 250, 340, 430 and 550). Optical flow and template matching propagated each mark forward and backward, choosing the closest successful seed by frame distance. The resulting 560 samples cover every frame. Median error on 12 additional spot checks was 2.1 pixels at 540 × 960, with a maximum of 13.2 pixels. Several of those checks share the stationary address pose. Between checks the path jumped 416 pixels when it switched seeds. This prevents using that trajectory directly for the avatar or speed estimation.

A second pass retained points within 150 playback milliseconds of their seed or corroborated within 10 pixels by a seed on the other side. It retained 168/560 frames (30%) and reduced the largest step across consecutive retained frames to 29.4 pixels. None of the original 12 checks fell in the retained frames. I therefore sampled five additional frames, 15 native frames from their respective seeds, and marked their clubheads from the video. The filter returned four positions with errors of 0.1, 2.4, 8.1 and 2.7 pixels. It returned no point at frame 355. These checks are by the same AI reviewer on one swing. Adjacent-step size alone does not establish correctness.

Run ` .venv/bin/python scripts/shaft_seeded_lucas.py ` from the project root to reproduce the assisted tracks and [spot-check errors](../experiments/native_swing_shaft_v2/lucas_gated_spot_errors.csv).
