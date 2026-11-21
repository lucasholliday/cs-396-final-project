# Club motion in the range viewer

The earlier avatar sampled the body at 30 fps and rounded the assisted club frame to the nearest pose frame. On most shots its club orientation came from an arm and wrist heuristic. Independent orientations could reverse abruptly.

Open Swing Lab now interpolates a 120 Hz club animation against playback time. A temporal path solver penalizes angular jumps and directions that intersect approximate torso, head and pelvis volumes. It uses timestamped native-video shaft segments when neighboring frames agree and, on Sid 01, the existing assisted native clubhead observations. Depth remains estimated. Small grip adjustments ease the hands around the body, limited to 8 cm in scene coordinates. Both palms stay on one rigid handle and the forearms terminate at those palms.

Across 76 shots on the default depth branch, 657 native-frame cues contributed on 45 shots, including 397 on Sid 01. The other samples are reconstructed. A cue on a shot does not establish a verified whole-swing track.

| Scene-graph check, sampled every 1/120 playback second | Before | After |
|---|---:|---:|
| Samples | 31,692 | 31,692 |
| Maximum shaft direction change between samples | 172.52° | 13.42° |
| Changes exceeding 15° | 1,135 | 0 |
| Shaft samples intersecting approximated head, torso or pelvis | 1,480 | 12 |

The clipping check covers the shaft beyond the grip center and uses ellipsoids rather than exact clothing meshes. Some intersections persist, and arm or clothing contact can still be visible from particular views. Palm alignment and shaft-to-hosel continuity were also checked across all 7,980 source poses. All changes apply to visualization.

Rebuild the viewer once to supply the source payload, then run the following from the project directory. Source samples, rendered samples and per-shot comparisons live under `outputs/club_motion/`.

```sh
node scripts/sample_range_club.js
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/smooth_range_club.py
.venv/bin/python scripts/build_research_viewer.py
node scripts/sample_range_club.js --check
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/smooth_range_club.py --check
node scripts/verify_range_scene.js
```
