# GEMINI Agent Handoff & Technical Work Summary

**Project:** CS 396 ML + Sensing Final Project — Multi-Modal Golf Swing Sensing & Ball Flight Prediction  
**Workspace:** `/Users/lucasholliday/Downloads/CS 396 ML + Sensing/Final Project`  
**Primary Viewer:** [Open Swing Lab.html](file:///Users/lucasholliday/Downloads/CS%20396%20ML%20%2B%20Sensing/Final%20Project/Open%20Swing%20Lab.html) (redirects to `outputs/research_viewer/index.html`)  
**Environment:** `.venv/bin/python` (Python 3.10.7) & Node.js (v22.17.1)  
**Verification:** 38 unit tests passing (`tests/`); Three.js 7,980-frame geometry verification passing.

---

## 1. Executive Summary & Mission

The goal of this project is to predict and visualize full golf shot trajectories and outcomes using consumer-grade/wearable and vision sensors (high-speed camera at 240 fps, Apple Watch IMU, clubhead IMU) as a low-cost alternative to commercial Doppler radar launch monitors ($20,000+ TrackMan 4).

TrackMan radar labels (`trackman_shot_labels.xlsx`) serve strictly as ground-truth training/evaluation targets and benchmark comparison, **never as runtime inference inputs**. All predictions are evaluated strictly under **Leave-One-Golfer-Out (LOGO)** cross-validation to prevent identity leakage across players.

---

## 2. Core Sensing Modalities & Dataset

1. **High-Speed Camera Video (`Video Data/`)**:
   - 240 fps, $1080\times1920$ portrait, down-the-line / oblique angle.
   - Captures pre-impact clubhead delivery and post-impact early ball launch (first ~15–30 ms).
   - Raw folders: `lucas_1`, `lucas_2`, `sid_1`, `sid_2`, `edward_1`, `edward_2`.
2. **Apple Watch IMU (`Watch Data/`)**:
   - High-rate accelerometer and gyroscope data capturing wrist kinematics and swing phase transitions.
3. **Clubhead IMU (`Club Data/`)**:
   - Sensor affixed near the hosel/grip ($\sim 1\text{''}$ below grip edge, facing clubface). Sample rate is $\sim 188.6\,\text{Hz}$.
   - Gyroscope ($g_x, g_y, g_z$) captures high-frequency 3D rotational mechanics without clipping; raw accelerometer experiences high-g impact saturation and is excluded from initial linear predictors.
4. **TrackMan Ground Truth (`trackman_shot_labels.xlsx`)**:
   - Flat Carry (yd), Total Distance (yd), Ball Speed (mph), Launch Angle (deg), Launch Direction (deg), Apex Height (ft), Curve (ft), Carry Side (ft), Total Side (ft), Landing Angle (deg), Hang Time (s). *(Note: TrackMan spin rate was not captured or transcribed in this dataset).*

> **CRITICAL RULE**: Raw data directories (`Club Data/`, `Video Data/`, `Watch Data/`, `trackman_shot_labels.xlsx`) are strictly **immutable**. Never modify or overwrite them.

---

## 3. Key Systems & Features Built

### A. Video Ball Tracking & Pre-Impact Clubhead Velocity
- **Ball vs. Club Disambiguation**: Resolved tracking failures where visual trackers confused the shiny iron clubhead with the white ball during release (`lucas_1_02`, `lucas_1_09`, `lucas_1_10`, `lucas_1_12`).
- **Pipeline (`scripts/video_launch_tracker.py`, `scripts/separate_ball_club.py`)**:
  - Implemented high-pass temporal differencing and dual-polarity silhouette extraction.
  - Directional spatial gating: Separated downward/forward club path from upward/forward ball departure cone ($+10^\circ \text{ to } +35^\circ$).
  - Extracted authentic early ball launch velocity vectors and launch elevation angles.
  - Optical flow pre-impact clubhead speed estimation (`scripts/native_club_flow.py`) over the final 8 frames prior to impact.

### B. Autonomous Video Launch Prediction Models
- **Model Script**: `scripts/video_launch_benchmark.py`, results in `reports/video_launch_tracking_results.md`.
- **Methodology**: Leave-One-Golfer-Out (LOGO) cross-validation evaluating Ridge, Random Forest, and Gradient Boosted regressors.
- **Inputs at Inference**: Autonomous visual ball velocity vector, climb angle, and pre-impact clubhead speed (zero radar inputs).
- **Performance**:
  - **Carry Distance**: **26.04 yd MAE** across all unseen golfers (beating multi-modal baseline of 35.88 yd MAE).
  - **Launch Direction**: **4.50° MAE** across unseen golfers.
  - Validated by unit tests in `tests/test_video_launch_tracking.py`.

### C. Visual-Inertial Clubhead Arc Fusion & Gyro Face-to-Path Modeling
- **Fusion Engine Script**: `scripts/fuse_club_arc.py`, benchmark script `scripts/club_fusion_benchmark.py`, report in `reports/club_inertial_fusion_results.md`.
- **Methodology**:
  - **Grip Spatial Anchor**: Anchored the club grip to the golfer's 3D wrist positions $\vec{W}(t)$ extracted from camera video via MediaPipe.
  - **Rotational Orientation**: Integrated the high-rate clubhead gyroscope ($\mathbf{q}(t)$ at 188.6 Hz) to point the $37.25\text{''}$ ($0.946\,\text{m}$) steel shaft vector in 3D space:
    $$\vec{P}_{\text{head}}(t) = \vec{W}(t) + 0.946 \cdot \vec{v}_{\text{shaft}}(t)$$
  - **3D Delivery Arc & Swing Plane**: Traced continuous 3D delivery arcs and fitted downswing delivery planes via SVD across all 76 shots (`data/processed/club_fusion/*.json`).
  - **Face-to-Path Extraction**: Gyro roll axis ($G_z$) directly measures dynamic clubface angle at delivery, impact face closure rate ($d\text{face}/dt$), and delivery plane tilt.
- **Predictive Breakthrough under Leave-One-Golfer-Out (LOGO)**:
  - **Lateral Curve (`curve_ft`)**: Gyro Face-to-Path cut MAE from **30.21 ft to 21.53 ft** (a **28.7% error reduction**) and flipped held-out $R^2$ from **$-0.152$ to $+0.211$**, with median error dropping to **13.88 ft** across unseen golfers. On curving shots ($|\text{curve}| \ge 15\text{ ft}$), curvature miss was reduced by **9.5 ft**.
  - **Lateral Landing Dispersion (`carry_side_ft`)**: Fusing camera launch angle with gyro curvature reduced MAE to **37.09 ft** ($R^2 = +0.297$), beating vision-only (43.04 ft MAE).
  - **Physical Complementarity**: Camera resolves early departure angle (4.51° MAE) and vertical climb rate (26.04 yd carry MAE); club gyro resolves impact rotational mechanics and aerodynamic curvature.

### D. Aerodynamic Flight Continuation & Turf Bounce/Rollout Engine
- **Engine Script**: `scripts/ball_continuation.py` and `scripts/range_scene.js`.
- **Aerodynamics**: Implemented realistic golf ball flight mechanics with gravitational acceleration ($9.81\,\text{m/s}^2$), quadratic drag ($C_D \approx 0.22 - 0.28$), and Magnus lift ($C_L \approx 0.16 - 0.22$) generated by backspin.
- **Hang Time Calibration**: Fixed unrealistic quick-drop bug by extending simulation hang times to authentic 5.5s – 7.0s intervals.
- **Physical Turf Bounce & Rollout**: Post-impact timeline is partitioned into airborne carry flight (first 70%) followed by decaying vertical turf bounce hops ($0.28\text{m} \rightarrow 0.09\text{m} \rightarrow \text{turf}$) and forward deceleration to resting Total Distance.

### E. Anatomical 3D Golfer Avatar & Inverse Kinematics
- **Scene Script**: `scripts/range_scene.js`.
- **Anatomical Inverse Kinematics (`solveKnee`)**:
  - Replaced noisy MediaPipe landmark jitter with analytical two-bone IK using standard anthropometric femur (0.42m) and tibia (0.40m) lengths.
  - Constrained knee flexion strictly to the forward sagittal plane aligned with pelvis rotation. Completely eliminated backward knee hyperextension ("flamingo legs").
- **Realistic Forged Iron & Dynamic Blade Rotation**:
  - Stepped chrome steel shaft, Tour Velvet black rubber grip, black hosel ferrule, and forged muscle-back blade with authentic $34^\circ$ 7-iron loft.
  - Blade dynamically rotates around the shaft according to the measured `dynamic_face_angle_deg` (open, square, closed).
- **Ball on Tee Mat**:
  - Golf ball rests squarely on the synthetic driving range hitting mat tee spot at $(X = 0.48\,\text{m}, Y = 0.068\,\text{m}, Z = -0.05\,\text{m})$.

### F. Interactive 3D Driving Range & Research Reviewer UI
- **Dedicated Visual-Inertial Section**:
  - Prominent panel in [Open Swing Lab.html](file:///Users/lucasholliday/Downloads/CS%20396%20ML%20%2B%20Sensing/Final%20Project/Open%20Swing%20Lab.html): **Visual-Inertial Clubhead Arc Fusion** displaying dynamic face angle, swing plane tilt, closure rate, rotational delivery velocity, synchronization status badge, and full LOGO benchmark table.
- **Interactive 3D Range Controls**:
  - `[⚡ Club Arc: ON]`: Toggles glowing amber 3D clubhead delivery trail.
  - `[📐 Delivery Plane: OFF]`: Toggles translucent cyan delivery plane.
  - `[↺ Replay swing + flight]`: Restarts swing and flight playback.
- **6-Angle Camera Suite**:
  - **📐 Side 3/4 (Tight) [Default]**: Right flank perspective framing golfer in lower-left and full arc/landing separation down the fairway.
  - **📐 Side 3/4 (Wide)**: Elevated stadium side view.
  - **📐 Face-On Side**: Face-on perspective from the left flank.
  - **🎯 Landing Zone**: Front-row target green camera.
  - **🎥 Follow Ball**: Dynamic chase drone tracking behind ball in flight.
  - **⛳ Behind Bay**: Traditional elevated launcher perspective behind hitting bay.
- **Simultaneous Dual Flight Arcs & Collision-Free Badges**:
  - TrackMan Radar Truth (cyan `#38bdf8`) vs. Best Model Prediction (purple `#c084fc`).
  - Dynamic 2D collision separation prevents yardage badge text overlap.
  - Dual turf landing rings (Carry bullseye + Total distance resting ring).

### G. Ball Tracking Recovery for High-Speed Strikes & UI Strike Quality Labeling
- **`rohan_1_02` Investigation & High-Speed Launch Recovery**:
  - *Root Cause Analysis*: Shot `rohan_1_02` was a high-speed strike (111 mph ball speed, 143 yd carry). At 240 fps, departure velocity was $19,895\,\text{px/s}$. The legacy candidate tracker in `scripts/link_ball_v6.py` had a hardcoded upper speed gate of `distance / dt > 18000` (equivalent to ~100.7 mph), rejecting the ball candidate immediately. In addition, the initial departure step clamped $dt$ to 0.012s, underestimating launch velocity and penalizing early residual.
  - *Resolution*: Updated kinematic velocity ceiling to $28,000\,\text{px/s}$ (~155 mph ball speed), enforced early-departure polarity matching, and relaxed initial acceleration residual thresholds.
  - *Outcome*: Successfully recovered a 10-frame high-speed ball track for `rohan_1_02` (launch direction $-5.18^\circ$, rise 489.7 px) matching TrackMan's $-7.1^\circ$ pull. Recovered 5 missing shots across the dataset (`rohan_1_02`, `rohan_2_05`, `rohan_2_10`, `andrew_1_07`, `eduardo_1_01`), raising tracking coverage from 68/76 to 73/76 shots.
- **Simplified UI Tracking Status & Unverified Tracking Abstention**:
  - Removed all arbitrary 100-yard classification rules. Tracking success is strictly defined by physical camera ball departure capture.
  - Dropdown menu `#shot` displays clean shot names (`rohan 1 02 · Rohan`) with zero tags for normal shots; only abnormal strikes are tagged:
    - `rohan 1 03 · Rohan [TrackMan Missed Reading]` (radar missed reading)
    - `rohan 1 04 · Rohan [Camera Missed Ball]` (stationary tee lighting glare)
    - `rohan 1 05 · Rohan [Camera Missed Ball]`
  - Dedicated top banner `#strikeQualityBanner` dynamically adapts:
    - **Camera Launch Tracked** (Green): `✓ CAMERA LAUNCH TRACKED (${frames} FRAMES)` with autonomous video launch metrics.
    - **Camera Missed Ball** (Indigo): `⚠️ CAMERA MISSED BALL DEPARTURE (UNVERIFIED BALL TRACKING)`. Model abstains from prediction (`Prediction Abstained`), displays `—`, and suppresses 3D flight ribbons/badges.
    - **TrackMan Missed Reading** (Slate): `ℹ️ TRACKMAN RADAR MISSED READING`.
- **Multi-Modal Sensing Architecture & Benchmark Summary**:
  - Added dedicated dashboard section and responsive comparative error chart at the bottom of `Open Swing Lab.html` (`outputs/research_viewer/index.html`):
    - **Sensor Matrix**: Camera Video (Carry, Launch Angle, Ball Speed, Direction), Club Gyro (Lateral Curve Face-to-Path), Fusion (Carry Side Landing, 2D Landing Error), Apple Watch (Contact timing).
    - **Leave-One-Golfer-Out Performance (70 Tracked Shots)**: Flat Carry: **15.77 yd MAE** (-56.0% error vs baseline); Launch Angle: **2.58° MAE** (-56.9%); Ball Speed: **8.61 mph MAE** (-39.0%); Launch Direction: **4.23° MAE** (-55.5%); Lateral Curve: **21.53 ft MAE** (-28.7%); 2D Landing Error: **24.28 yd MAE** (-37.1%).

### H. Turf Bounce, Rollout Direction & TrackMan Empirical Dynamics
- **Problem Statement**: Should post-carry turf bounce and rollout proceed purely forward downrange along $Z$, or follow the arrival tangent angle (curvature + direction) when the ball hits the ground?
- **Empirical Analysis of TrackMan Ground Truth (`trackman_shot_labels.xlsx`)**:
  - TrackMan records both `Carry side (ft)` and `Total side (ft)` across all 72 shots.
  - Correlation between `Carry side` and `Total side` is **$r = 0.985$**.
  - Median lateral shift during rollout is only **6.00 ft** (2.0 yards), with empirical regression $\text{Total Side} \approx 1.22 \times \text{Carry Side}$.
  - In real golf, 7-iron turf contact has 5,000–7,000+ rpm backspin. Surface grass friction checks and kills sideways velocity upon impact; the ball rolls predominantly forward.
- **Rollout Model Comparison on 2D Distance Error (N = 70 Shots)**:
  - *Baseline (Pose + Watch)*: 35.15 yd MAE (Median: 30.89 yd)
  - *Undamped Flight Tangent* (100% descent angle): **29.24 yd MAE** (Median: 20.85 yd). An unconstrained tangent over-amplifies lateral movement into 15–30 ft of drift, multiplying prediction errors.
  - *Calibrated Turf Friction Tangent* ($k = 0.22$, matching TrackMan): **25.85 yd MAE** (Median: 20.12 yd).
  - *Straight Forward Rollout*: 25.40 yd MAE (Median: 19.26 yd).
- **Implementation in `scripts/range_scene.js`**:
  - Integrated the calibrated turf friction factor $k = 0.22$:
    $$\vec{P}_{\text{rollout}}(t) = \vec{P}_{\text{land}} + \begin{pmatrix} u_{\text{tangent}, x} \cdot \text{roll} \cdot 0.22 \\ \text{hop}(t) \\ u_{\text{tangent}, z} \cdot \text{roll} \end{pmatrix}$$
  - Fades and slices visibly hop and roll to the right; draws and hooks visibly hop and roll to the left, while maintaining high 2D landing distance accuracy.

---

## 4. Key Files & Directory Structure

| File / Directory | Purpose |
| :--- | :--- |
| `Open Swing Lab.html` | Root entrypoint to open the research review tool in any web browser. |
| `outputs/research_viewer/index.html` | Built standalone review viewer containing all 76 shots, 3D range scene, synced video, and metrics. |
| `scripts/range_scene.js` | Complete Three.js 3D driving range engine (avatar IK, club rendering, dual flight ribbons, landing rings, floating badges). |
| `scripts/viewer_template.html` | HTML/CSS/JS template for the research viewer UI. |
| `scripts/build_research_viewer.py` | Builder script that compiles manifest, features, and scene template into `index.html`. |
| `scripts/fuse_club_arc.py` | Visual-inertial clubhead arc fusion engine (wrist grip anchor + gyro orientation + SVD delivery plane). |
| `scripts/club_fusion_benchmark.py` | Leave-One-Golfer-Out regression benchmark comparing Vision-Only vs Club Gyro vs Fusion. |
| `data/processed/club_fusion/` | Fused 3D delivery arc, swing plane, and face orientation JSON files for all 76 shots. |
| `data/processed/club_fusion_features.csv` | Extracted dynamic face angle, closure rate, plane tilt, and delivery velocity features. |
| `experiments/club_fusion_v1/` | LOGO cross-validation predictions and metric tables for club fusion models. |
| `reports/club_inertial_fusion_results.md` | Formal research report on visual-inertial fusion benchmark and lateral outcomes. |
| `scripts/verify_range_scene.js` | Headless Node.js geometry verification script testing 76 shots and 7,980 animation frames. |
| `scripts/video_launch_tracker.py` | High-speed video ball launch extraction (velocity vector, launch angle, clubhead speed). |
| `scripts/video_launch_benchmark.py` | Leave-One-Golfer-Out regression models predicting carry distance and launch direction. |
| `scripts/ball_continuation.py` | Aerodynamic flight simulation model (gravity, drag, Magnus lift). |
| `scripts/native_club_flow.py` | Optical flow tracker for pre-impact clubhead delivery. |
| `scripts/separate_ball_club.py` | Disambiguation logic between ball departure and clubhead follow-through. |
| `reports/video_launch_tracking_results.md` | Formal report on video launch model benchmarks and cross-validation metrics. |
| `PROJECT_STATUS.md` | High-level status log and next tasks roadmap. |
| `EXPERIMENT_LOG.md` | Detailed chronological experiment and milestone log. |
| `tests/` | 38 passing unit tests covering audit, avatar motion, club display, club fusion, contact studies, research contract, tracking review, video launch tracking, and visual observations. |

---

## 5. How to Run, Test, and Rebuild

1. **Run Unit Tests**:
   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   ```
   *(All 38 tests must pass)*

2. **Verify 3D Range Scene Geometry**:
   ```bash
   node scripts/verify_range_scene.js
   ```
   *(Checks all 76 shots and 7,980 frames for finite coordinates, visible limbs, and grip alignment)*

3. **Rebuild the Research Viewer**:
   ```bash
   .venv/bin/python scripts/build_research_viewer.py
   ```
   *(Regenerates `outputs/research_viewer/index.html`)*

4. **Inspect in Browser**:
   Open [Open Swing Lab.html](file:///Users/lucasholliday/Downloads/CS%20396%20ML%20%2B%20Sensing/Final%20Project/Open%20Swing%20Lab.html) directly in Google Chrome or default browser.

---

## 6. Suggested Next Steps & Open Work for Future Agent (Codex)

1. **Multi-Modal Tri-Fusion Tuning (Push Carry MAE Below 14–15 yd)**:
   - Test whether fusing Apple Watch peak downswing angular acceleration and swing tempo (backswing-to-downswing duration ratio) with clubhead gyro delivery velocity and camera launch features can break below 15.0 yd MAE across unseen golfers under LOGO cross-validation.
2. **Joint Multi-Target Aerodynamic Consistency (Carry, Total, Apex Height, Landing Angle)**:
   - TrackMan recorded Flat Carry, Total Distance, Apex Height (ft), and Landing Angle (deg). Use joint multi-target modeling or physical flight envelope constraints (e.g., Total must be $\ge$ Carry, carry distance cannot exceed ballistic limits given ball speed and launch elevation) to reduce prediction variance on harder golfers like Lucas and Edward.
3. **Research Paper & Final Presentation Prep**:
   - Extract visual comparison figures and benchmark tables comparing Vision-Only vs. Club Gyro vs. Visual-Inertial Fusion vs. TrackMan Ground Truth for final project writeup and presentation slides.
   - Use the screenshots or headless snapshots from `outputs/range_scene_preview/` illustrating the 3D range, avatar IK, and dual trajectory comparison.
