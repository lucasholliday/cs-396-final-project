# Geometry and method sources

User confirms all right-handed golfers, lead/left watch wrist, standard-length Wilson Staff CB seven iron, club IMU approximately one inch below the grip's lower edge. Exact chip-to-head position is not yet measured. Record these separately from derived estimates in `configs/equipment.json`.

[Wilson's current Staff Model CB specifications](https://www.wilson.com/en-us/products/staff-model-cb-irons-wg1p0386) list the seven iron at 37.25 inches (0.94615 m), 34° loft and 62° lie. Retrieved 2026-11-17. The physical club's model year and exact chip/grip dimensions remain unspecified. Static loft is not dynamic launch angle; lie is not camera calibration.

For a rigid club, head velocity equals sensor translation velocity plus angular velocity crossed with the sensor-to-head position vector. Therefore multiplying gyro magnitude by total club length is not a validated clubhead speed: the lever arm differs from total playing length, the vector orientation matters, grip/sensor translate, and shaft flex can violate a rigid-body approximation. Likewise, a single camera sees projected motion with depth ambiguity. Metric speeds require geometry, trustworthy time and uncertainty validation. No such metric speed is currently reported.

[MediaPipe Pose Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) supplies the local video-mode API and image/world landmark interpretation. The full model is downloaded from Google's [versioned model bundle](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task). Thirty-three estimated landmarks are cached; model-world coordinates are predictions, not ground-truth 3D motion capture. Visibility is not an accuracy guarantee. No video is uploaded for inference.

Scientific scope: current body/flow regression uses whole swings including follow-through, selected with watch events. Native ball tracking is a distinct post-impact capability. Gaze, exact 3D pelvis/shoulder rotation, physical clubhead speed, spin and practice-swing generalization have not been established.
