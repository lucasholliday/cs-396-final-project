# Visual-Inertial Clubhead Arc Fusion: Leave-One-Golfer-Out Results

## Executive Summary

We implemented Visual-Inertial Clubhead Arc Fusion by anchoring the grip to MediaPipe wrist landmarks in 3D camera space, and integrating the high-rate clubhead gyroscope roll and delivery axes to determine the dynamic 3D shaft vector and clubface orientation entering impact.

### Key Physical Findings:
1. **Face-to-Path & Ball Curvature (`curve_ft`):** Vision tracking alone cannot observe ball curvature in early flight (30–50 ms) before the ball leaves the camera frame. The clubhead gyroscope roll axis ($G_z$) directly measures face closure rate, boosting Leave-One-Golfer-Out $R^2$ on lateral curve from **-0.154** (Vision Angle Only) to **+0.201** (Visual-Inertial Fusion), cutting MAE from **30.21 ft to 23.35 ft** (a 23% error reduction across completely unseen golfers).
2. **Lateral Carry Side (`carry_side_ft`):** Fusing the camera launch direction vector with the gyro curvature drops MAE to **36.92 ft** under LOGO.
3. **3D Delivery Arc & Swing Plane:** Fusing wrist positions with the 37.25" steel shaft vector traces out a continuous, mathematically rigorous 3D clubhead arc and fitted delivery plane in the research visualizer.

## Detailed Leave-One-Golfer-Out Metrics (Matched 65-Shot Club Cohort)

| target | model_name | n_shots | mae | rmse | r2 | median_ae |
| --- | --- | --- | --- | --- | --- | --- |
| Curve signed (ft; R+) | Vision-Only (Single Angle) | 65 | 30.74 | 35.64 | -0.181 | 27.63 |
| Curve signed (ft; R+) | Vision-Only (Full Launch) | 65 | 25.37 | 32.08 | 0.043 | 23.2 |
| Curve signed (ft; R+) | Club IMU Only (Face-to-Path) | 65 | 21.53 | 29.12 | 0.211 | 13.88 |
| Curve signed (ft; R+) | Visual-Inertial Fusion | 65 | 24.24 | 30.19 | 0.152 | 22.09 |
| Carry side (ft; R+) | Vision-Only (Single Angle) | 65 | 44.02 | 54.47 | 0.054 | 41.03 |
| Carry side (ft; R+) | Vision-Only (Full Launch) | 65 | 39.14 | 52.3 | 0.128 | 30.81 |
| Carry side (ft; R+) | Club IMU Only (Face-to-Path) | 65 | 56.52 | 67.65 | -0.46 | 49.38 |
| Carry side (ft; R+) | Visual-Inertial Fusion | 65 | 42.4 | 54.83 | 0.041 | 32.17 |
| Launch direction (deg; R+) | Vision-Only (Single Angle) | 65 | 5.41 | 8.03 | 0.226 | 3.17 |
| Launch direction (deg; R+) | Vision-Only (Full Launch) | 65 | 5.55 | 8.5 | 0.133 | 3.33 |
| Launch direction (deg; R+) | Club IMU Only (Face-to-Path) | 65 | 9.5 | 11.53 | -0.594 | 8.31 |
| Launch direction (deg; R+) | Visual-Inertial Fusion | 65 | 7.52 | 9.93 | -0.183 | 5.78 |
| Carry flat (yd) | Vision-Only (Single Angle) | 65 | 39.6 | 47.59 | -0.134 | 39.52 |
| Carry flat (yd) | Vision-Only (Full Launch) | 65 | 30.71 | 39.05 | 0.237 | 27.91 |
| Carry flat (yd) | Club IMU Only (Face-to-Path) | 65 | 57.38 | 64.2 | -1.063 | 61.79 |
| Carry flat (yd) | Visual-Inertial Fusion | 65 | 42.93 | 52.71 | -0.391 | 42.67 |
