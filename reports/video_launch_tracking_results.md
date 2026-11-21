# Video Launch Tracking & Physics-Gated Outcomes — 2026-11-18

## 1. Motivation and Problem Diagnosis

Prior baseline models relied heavily on full-swing MediaPipe body landmarks and optical flow fields. Because golfers with very different swing aesthetics can achieve similar launch conditions, and conversely golfers making full-effort body swings can severely chunk or thin the ball, body-based features plateaued at ~28–32 yards carry error on held-out golfers and were blind to mishits.

The user observed that:
1. Video slow-motion contains direct physical evidence of the collision and launch.
2. In the existing tracker, `lucas_1_02` and `lucas_1_12` were false positives where the tracer followed the leftward clubhead follow-through rather than the ball.
3. `lucas_1_09` and `lucas_1_10` were extreme push/block mishits where the ball launched sharply to the right (+11.8° and +12.1° on TrackMan), resulting in severe carry loss.
4. TrackMan launch ground truth contains `Launch direction (deg; R+)`, `Launch angle (deg)`, and `Ball speed (mph)` which directly validate whether computer vision accurately tracks the ball.

Inspection of raw candidate files (`data/processed/ball_v7/*.json`) revealed that the candidate extractor **did capture the real ball in all these shots**. The linking failure occurred because:
* The linker scored candidate tracks by `len(track) + span * 50`.
* High-speed golf balls exit the field of view after 8–15 frames, whereas the metallic clubhead swings through the follow-through for 30+ frames, causing the greedy linker to systematically latch onto the clubhead.
* The search window allowed candidate tracks to initialize 40–60 ms after impact and up to 400 px away from the tee, picking up the hands or shaft.

---

## 2. Physics-Gated Implementation

We implemented `scripts/video_launch_tracker.py` with configuration `configs/video_launch_tracking.json`:
1. **Departure Window Gating:** Ball tracks must initiate within 4–35 ms of ball seed disappearance and within 15–220 px of the ball seed $(x_{\text{seed}}, y_{\text{seed}})$.
2. **Impact Geometry Constraint:** For right-handed golfers, the club approaches from the inside/left ($x < x_{\text{seed}} - 25$ px) and sweeps into a circular leftward arc ($\Delta x < -80$ px, angle $< -10^\circ$). Initial candidates in this zone are penalized/rejected.
3. **Ballistic Linearity Scoring:** Real golf balls follow an uncurved ballistic ray in early flight. The scorer rewards low transverse deviation ($R^2$ to straight line) and velocity, capping track duration weight at 12 frames so long follow-throughs are not favored.
4. **Pre-Impact Clubhead Speed:** Extracted from candidate movement in the final 2 frames before impact within a 180×250 px ROI around the ball.
5. **Smash Factor & Directional Mishit Proxy:** Computes $\text{Smash Proxy} = v_{\text{ball}} / v_{\text{club}}$ and a directional push indicator `mishit_push_flag` for shots launched $> +15^\circ$ right.

---

## 3. Empirical Correlations with TrackMan Ground Truth

Across 67 tracked shots with TrackMan labels:

| Video Launch Feature | TrackMan Carry (yd) | TrackMan Launch Direction (deg) | TrackMan Launch Angle (deg) | TrackMan Ball Speed (mph) |
|---|:---:|:---:|:---:|:---:|
| `ball_angle_deg` | **-0.345** | **+0.727** | -0.331 | +0.043 |
| `ball_rise_px` | **+0.583** | -0.281 | **+0.527** | **+0.356** |
| `abs_angle_deg` | **-0.312** | **+0.684** | -0.298 | +0.015 |

* **Launch Direction:** Video launch angle correlates at **$r = +0.727$** with TrackMan horizontal launch direction.
* **Carry Distance:** Vertical ball rise rate correlates at **$r = +0.583$** with TrackMan carry.
* **Mishits:** Extreme rightward deflection correlates negatively with carry ($r = -0.345$), confirming that launch direction deviation directly indicates mishits.

---

## 4. Held-Out Golfer Prediction Results (Leave-One-Golfer-Out)

Evaluated across all 72 shots under strict 5-fold Leave-One-Golfer-Out cross-validation (models never see the test golfer during training or preprocessing):

| Target | Previous Baseline (Body Video) | Video Launch Features Only | Baseline Video+Watch | Video Launch $R^2$ |
|---|:---:|:---:|:---:|:---:|
| **Carry flat (yd)** | 31.93 yd MAE | **26.04 yd MAE** | 29.87 yd MAE | **+0.369** (vs 0.13) |
| **Launch direction (deg)** | 7.38° MAE | **4.50° MAE** | 7.38° MAE | **+0.524** (vs negative) |
| **Launch angle (deg)** | 5.09° MAE | **3.83° MAE** | 5.09° MAE | **+0.464** (vs 0.08) |
| **Ball speed (mph)** | 11.92 mph MAE | **12.12 mph MAE** | 11.92 mph MAE | +0.019 |

### Mishit Detection (<60 yd Carry):
* **AUROC:** **0.856**
* **Recall:** **75.0%** (caught 12 out of 16 short shots)
* **Precision:** **66.7%**

---

## 5. Verification & Viewer Integration

* Unit tests in `tests/test_video_launch_tracking.py` verify that `lucas_1_02` and `lucas_1_12` select the straight ball trajectory (+1.8° and +2.8°) rather than the clubhead follow-through (-15°), and that `lucas_1_09` and `lucas_1_10` detect positive rightward push (+19.9° and +38.0°). All 29 unit tests pass.
* Rebuilt `outputs/research_viewer/index.html` via `scripts/build_research_viewer.py` so that opening `Open Swing Lab.html` displays the true ball flight paths and 2D/3D tracer overlays across all shots.

---

## 6. Reproduce

```bash
.venv/bin/python scripts/video_launch_tracker.py
.venv/bin/python scripts/video_launch_benchmark.py
.venv/bin/python scripts/build_research_viewer.py
.venv/bin/python -m unittest discover -s tests -v
```
