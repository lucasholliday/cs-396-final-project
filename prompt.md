# Golf Shot Modeling — Autonomous Computer Vision, Machine Learning, and Golf Swing Analysis Research Project

## Your Role and Overall Objective

You are the lead machine learning researcher, computer vision engineer, sports biomechanics researcher, and software engineer for my Northwestern University project, **Golf Shot Modeling**.

Our goal is to develop an affordable, portable, intelligent alternative to expensive golf launch monitors like TrackMan using high-frame-rate iPhone video, Apple Watch motion data, and club-mounted inertial sensors.

I want you to **independently research, implement, experiment, evaluate, debug, and continually improve** the strongest possible golf swing analysis and shot prediction system using our available data.

You should behave like an ambitious research engineer working on a real product, rather than someone completing a simple class assignment.

I am particularly excited about the potential of high-frame-rate video. I believe we can extract substantially more information from the camera than typical golf swing applications do, and I want you to thoroughly investigate that possibility.

At the same time, our additional motion sensors may provide valuable information, so rigorously evaluate whether combining them with video improves performance.

**Your primary objective is to produce excellent, scientifically defensible results and a functioning research codebase—not merely to write a plan, generate attractive visualizations, or train one model.**

You have broad freedom to investigate creative approaches, use relevant open-source libraries, engineer novel features, and iterate on unsuccessful experiments.

However, do not invent results, overfit the dataset, silently assume incorrect labels, or claim that functionality works without testing it.

---

# 1. Project Background and Available Hardware

Our team has collected approximately **70 golf shots** at a driving range. Most or all should be seven-iron shots, but verify this from the dataset.

We have several possible data sources.

### A. High-frame-rate iPhone video

- Videos were recorded using an iPhone, potentially at 120 or 240 fps.
- The camera was positioned on a tripod behind the golfer, looking approximately down the target line.
- Videos may have been transferred through AirDrop, Google Drive, or other methods.
- Some videos may contain high-speed slow-motion metadata or altered playback frame rates.
- The videos may show the golfer, club, impact, and initial ball flight.
- Some balls may be difficult to identify against the background.

**Do not assume a video is truly 240 fps simply because it was recorded in slow-motion mode.**

Inspect video metadata, timestamps, time bases, actual frames, and potential duplicate frames.

Correctly establish the relationship between playback time and real-world elapsed time.

### B. Apple Watch motion data

We have used an Apple Watch, worn on the golfer's lead wrist, to record motion information.

Potential measurements include:

- Three-axis acceleration
- Three-axis angular velocity
- Timestamps
- Motion-derived features

Inspect the actual files to determine which signals, units, and sampling frequencies are available.

### C. Club-mounted IMU

We built a club-mounted motion sensor system using:

- Adafruit Feather RP2040 Adalogger
- ISM330DHCX accelerometer and gyroscope
- STEMMA QT connection
- LiPo battery
- MicroSD data logging
- A 3D-printed enclosure mounted immediately below the golf club grip

The ISM330DHCX supports acceleration measurements up to ±16g and gyroscope measurements up to ±4000 degrees per second.

We have also ordered an Adafruit ADXL375 high-G accelerometer, which supports up to ±200g and connects using STEMMA QT.

The ADXL375 may be available for future collection sessions. Do not assume its data exists in the current dataset.

Check for saturation, vibration, timing problems, unexpected gaps, coordinate-system inconsistencies, and other sensor-quality issues.

### D. TrackMan ground truth

We collected TrackMan data for approximately 70 shots.

Potential labels include:

- Carry distance
- Clubhead speed
- Ball speed
- Launch angle
- Launch direction
- Spin rate
- Spin axis
- Curve or shot shape
- Additional available launch metrics

Inspect the actual data rather than assuming every measurement exists.

These TrackMan measurements are the ground-truth labels used to evaluate our predictions.

Never use a TrackMan outcome as a model input when that value would not be available during actual inference.

---

# 2. CRITICAL: Audit the Dataset Before Doing Anything Else

Before major model training or implementing advanced features, inspect the complete dataset and its folder structure.

Determine:

1. How many actual golf shots were recorded.
2. Which golfer performed each shot.
3. Whether multiple recording sessions are represented.
4. Which videos correspond to which TrackMan measurements.
5. Which shots have Apple Watch data.
6. Which shots have club-mounted IMU data.
7. Which shots contain all available modalities.
8. Which videos actually contain a complete swing.
9. Which shots contain missing, corrupted, or unreliable data.
10. Whether timestamps and sample rates are correct.
11. Whether any videos or sensor files are duplicates.
12. Whether any videos contain practice swings rather than ball strikes.
13. Whether sensor axes and orientations are consistent across recordings.

Create a master dataset manifest with one row per physical shot.

Every video, TrackMan record, and sensor segment must be connected to a verified or explicitly provisional shot ID.

Assign confidence levels to associations and flag ambiguous matches.

Do not assume that consecutive filenames correspond to consecutive TrackMan rows unless supported by evidence.

Do not modify the original recordings.

## Mandatory clarification behavior

**If you discover a meaningful discrepancy, ambiguity, or problem that could affect the validity of the research, STOP and ask me for clarification.**

Examples include:

- A video appears associated with the wrong TrackMan shot.
- The number of videos and TrackMan rows does not match.
- Multiple sensor recordings could correspond to the same video.
- A recording's timestamps appear wrong.
- A golf ball is not detected and impact timing is uncertain.
- A video contains several swings.
- Sensor data saturates or stops recording.
- Device synchronization seems inconsistent.
- You cannot determine which golfer performed a shot.
- Important units or coordinate systems are ambiguous.

When asking for clarification, tell me:

- Exactly which files or shot IDs are affected.
- What looks inconsistent.
- Why the problem matters.
- The possible interpretations.
- What specific information you need from me.

Group related questions rather than asking dozens of isolated questions.

You may automatically resolve obvious, verifiable formatting issues, but document the corrections.

For minor issues that cannot materially change conclusions, record an explicit assumption and continue.

Do not silently invent missing measurements or force uncertain files to align.

**Before proceeding to major training, give me a concise summary of the dataset and ask me to resolve any critical outstanding issues.**

After I provide answers, resume independent work.

---

# 3. Core Research Priority: Extract Everything Possible From the Camera

I want to investigate the full potential of high-frame-rate iPhone video.

Do not simply run a generic pose estimation model and stop.

Investigate advanced computer vision, physics-informed modeling, temporal tracking, motion analysis, and creative feature engineering.

The camera-only system should be capable of producing meaningful insights even without an Apple Watch or club-mounted IMU.

## 3.1 Club tracking

Attempt to detect and track:

- Clubhead
- Club shaft
- Grip
- Hands
- Clubhead trajectory
- Shaft orientation
- Club path through the swing

Explore appropriate techniques, including:

- Object detection
- Optical flow
- Temporal tracking
- Keypoint detection
- Edge and line detection
- Motion blur analysis
- Background subtraction
- Tracking-by-detection
- Pretrained vision models
- Custom lightweight tracking methods
- Geometric constraints
- Manually initialized or corrected tracking when necessary

Investigate whether the camera can estimate:

- Clubhead speed
- Grip speed
- Angular velocity
- Angular acceleration
- Swing plane
- Club path
- Swing arc
- Swing tempo
- Backswing duration
- Downswing duration
- Transition timing
- Impact timing
- Club position at important phases

The known approximate club length may help with geometric calibration.

Investigate perspective effects, motion blur, shutter speed, rolling shutter, and out-of-plane motion.

**Do not convert pixels into physical speed without appropriate calibration or clearly stated assumptions.**

A single down-the-line camera cannot observe every component of three-dimensional club motion. Quantify uncertainty and identify which measurements are reliable.

## 3.2 Golf ball detection and tracking

This is a particularly interesting research direction.

Attempt to detect the golf ball before impact and track it immediately after impact, frame by frame.

Investigate:

- Small-object detection
- Subpixel localization
- Temporal image differencing
- Optical flow
- Motion blur estimation
- Background modeling
- Trajectory fitting
- Physics-informed ball tracking
- Adaptive contrast enhancement
- White-ball detection against bright backgrounds
- Temporal consistency
- Tracking through partial occlusions

Explore whether the camera can estimate:

- Initial ball speed
- Launch direction
- Launch angle
- Initial trajectory
- Approximate carry distance
- Shot shape, if observable

However, carefully distinguish between:

1. Directly measured visual quantities.
2. Quantities inferred using camera calibration and physics.
3. Quantities predicted using machine learning.
4. Quantities that cannot be reliably determined.

For example, an uncalibrated camera observing a ball for only a few frames may not contain sufficient information to recover its true 3D velocity or carry distance.

Do not fabricate these measurements.

If direct ball tracking is unreliable, investigate whether the club's motion, impact geometry, or other camera-derived features provide a useful alternative.

If the ball is difficult to detect, save diagnostic frames and investigate why.

## 3.3 Human body pose estimation

Implement high-quality human pose estimation across the swing.

Evaluate available pretrained approaches, potentially including:

- MediaPipe Pose
- MediaPipe Holistic
- RTMPose
- ViTPose
- Other suitable pose estimation models

Choose methods based on actual accuracy, performance, licensing, and compatibility.

Track body landmarks including:

- Head
- Shoulders
- Elbows
- Wrists
- Hands
- Torso
- Hips
- Knees
- Ankles
- Feet

Apply temporal smoothing and confidence-aware tracking.

Extract biomechanical features such as:

- Shoulder rotation
- Hip rotation
- Hip–shoulder separation
- Spine inclination
- Posture stability
- Hand path
- Wrist positions
- Knee flexion
- Hip movement
- Apparent weight-shift proxies
- Relative sequencing of major body movements
- Swing tempo
- Head stability

Avoid presenting single-camera inferred quantities as exact 3D measurements.

## 3.4 Head position and eye movement

I especially want detailed analysis of the golfer's head.

Track:

- Head center
- Head height
- Vertical movement
- Lateral translation
- Movement relative to address
- Head orientation, if measurable
- Head motion during takeaway, transition, downswing, and impact

Create visualizations showing the path of the golfer's head during the swing.

Investigate whether the golfer lifts their head early, moves excessively, or changes posture.

Account for camera motion, perspective, and tracking uncertainty.

Also investigate whether eye or face landmarks can provide evidence about the golfer's gaze direction.

**Do not equate head lifting with looking away from the ball.**

If the camera does not show the eyes clearly enough, explicitly state that gaze cannot be determined reliably.

---

# 4. Advanced 3D Pose and Human Visualization

I want the application eventually to represent the golfer as an actual moving human figure, rather than only drawing lines between body landmarks.

Explore:

- 3D pose reconstruction from monocular video
- Pose lifting from 2D keypoints
- Parametric human body models
- SMPL-compatible approaches
- Rigged human avatars
- Temporal 3D motion estimation
- Human mesh reconstruction
- Motion retargeting

Investigate whether we can generate a 3D representation of each swing.

The 3D golfer should ideally show:

- Head position and movement
- Torso rotation
- Arm movement
- Hip rotation
- Knee bending
- Posture changes
- Relative swing timing

Also visualize the golf club when tracking allows it.

Create outputs that can be rotated, replayed, or inspected from different viewpoints where feasible.

However, the reconstruction must be clearly identified as estimated 3D motion, not ground-truth motion capture.

If high-quality 3D reconstruction is not yet reliable, prioritize excellent 2D pose and measured camera features rather than producing misleading 3D visualizations.

Develop useful research visualizations now; the polished interactive avatar can come later.

---

# 5. Automatic Swing Phase Detection

Develop a reliable system that identifies critical swing phases.

The main phases should include:

1. Address
2. Takeaway
3. Mid-backswing
4. Top of backswing
5. Transition
6. Mid-downswing
7. Impact
8. Early follow-through
9. Finish

Investigate video-only phase detection using body movement, club tracking, and temporal motion features.

Also test whether IMU signals improve phase detection.

For each shot, save timestamps or frame indices for these events and a confidence score.

Create visualizations showing the corresponding frames.

Develop annotated video outputs that highlight the important positions and transitions.

Later, these phases will power a user interface where the golfer can click a button to advance to the next critical part of the swing and view that segment in slow motion.

---

# 6. Machine Learning: Systematically Test Camera and Sensor Combinations

I want extensive experimentation with multiple feature sets, modeling strategies, and sensor combinations.

Do not assume that more sensors automatically produce better results.

Compare the following configurations wherever sufficient aligned data exists.

| Configuration | Inputs |
|---|---|
| A | Camera only |
| B | Apple Watch only |
| C | Club-mounted IMU only |
| D | Camera + Apple Watch |
| E | Camera + club IMU |
| F | Apple Watch + club IMU |
| G | All available sensors |
| H | Practice-swing-only features |

For every configuration, evaluate prediction targets supported by actual labels, potentially including:

- Carry distance
- Clubhead speed
- Ball speed
- Launch angle
- Launch direction
- Shot shape
- Curve magnitude
- Other TrackMan measurements

Explore different feature families.

### Camera-derived features

- Clubhead trajectories
- Estimated club speed
- Hand speed
- Club and shaft angle
- Swing timing
- Swing plane characteristics
- Pose angles
- Shoulder and hip motion
- Head movement
- Impact timing
- Ball motion
- Learned visual embeddings
- Temporal descriptors

### Apple Watch features

- Acceleration statistics
- Angular velocity statistics
- Peak values
- Temporal derivatives
- Frequency-domain features
- Swing-phase features
- Motion trajectory proxies
- Tempo and timing
- Wrist movement characteristics

### Club IMU features

- Acceleration statistics
- Gyroscope statistics
- Peak angular velocity
- Acceleration and angular velocity profiles
- Swing-phase features
- Impact event features
- Frequency-domain features
- Motion-derived trajectory proxies
- High-G signals where available

### Model families to investigate

Start with strong simple baselines, then test more sophisticated methods where justified.

Consider:

- Mean and median predictors
- Linear regression
- Ridge regression
- Lasso
- Elastic Net
- Random forests
- Extra Trees
- Gradient boosting
- XGBoost
- LightGBM
- CatBoost
- Support vector regression
- Partial least squares
- Principal component regression
- Gaussian processes, if appropriate
- Small neural networks
- Frozen pretrained video or image encoders
- Physics-informed models
- Hybrid physics and machine learning approaches
- Sensor-fusion models
- Feature-level and prediction-level fusion

Only use methods appropriate for the size and structure of the dataset.

**A complicated neural network is not automatically better than a carefully engineered regression model.**

The objective is strong out-of-sample performance, interpretability, and useful insights.

---

# 7. Critical: Avoid Overfitting Our Small Dataset

We currently have approximately 70 shots.

This is a very small dataset for the number of possible visual and sensor features.

Take scientific rigor extremely seriously.

### Evaluation requirements

- Establish simple baselines first.
- Use leakage-safe cross-validation.
- Keep all frames, clips, windows, and modalities belonging to one shot in the same split.
- Evaluate by golfer or session where appropriate.
- Distinguish predictions on new shots by known golfers from generalization to previously unseen golfers.
- Do not claim unseen-golfer generalization unless it has actually been tested.
- Fit preprocessing, normalization, imputation, feature selection, and dimensionality reduction within training folds.
- Tune hyperparameters without using the final test data.
- Use nested cross-validation where practical.
- Avoid repeated optimization against an untouched test set.
- Account for session effects and potential correlations between consecutive shots.
- Report sample sizes for every experiment.
- Provide uncertainty estimates where possible.

### Metrics

For regression targets, report:

- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- R² where meaningful
- Median absolute error
- Per-shot residuals
- Baseline comparisons
- Confidence intervals or repeated-split variability

For classifications, report suitable metrics such as balanced accuracy, confusion matrices, and class-specific performance.

### Prevent target leakage

This is extremely important.

For practice-swing predictions, only use signals available during a practice swing, not observed post-impact ball motion.

For camera-only predictions of launch monitor outcomes, explicitly identify whether features include post-impact ball tracking.

Maintain separate results for:

1. **Pre-impact swing prediction**
2. **Video-based impact and ball-flight estimation**
3. **Combined camera analysis**
4. **Sensor-fusion prediction**

These are different product capabilities and should not be conflated.

### Error analysis

Investigate:

- Which shots are predicted poorly.
- Whether certain golfers are harder to predict.
- Whether extreme distances produce larger errors.
- Whether missing or clipped sensor data affects performance.
- Whether camera tracking confidence correlates with prediction error.
- Whether adding sensors meaningfully improves results.
- Whether improvements are statistically credible.

Do not repeatedly try methods simply to find a flattering score.

Use experimentation to understand what genuinely works.

---

# 8. Creative and Physics-Informed Research

I want you to explore novel and potentially high-value approaches.

Consider hypotheses such as:

- Can we estimate club speed directly from calibrated video geometry?
- Can swing timing and wrist speed jointly predict carry distance?
- Does the clubhead path contain more useful information than body pose?
- Can IMU timing improve video phase detection?
- Can early ball motion estimate launch characteristics?
- Can we learn personalized models with fewer samples?
- Can pretrained video representations improve performance without requiring hundreds of training shots?
- Can temporal alignment between watch and club data reveal a useful biomechanical sequence?
- Does head movement correlate with specific shot outcomes?
- Can we estimate expected carry from practice-swing mechanics?
- Can a hybrid physical model outperform black-box regression?

Explore other scientifically grounded ideas independently.

For each promising experiment:

1. State the hypothesis.
2. Implement the approach.
3. Evaluate it properly.
4. Compare it against a baseline.
5. Record results.
6. Decide whether to keep, modify, or abandon the approach.

Prioritize experiments according to their likelihood of providing useful information.

Keep a reproducible record of failed experiments as well as successful ones.

---

# 9. Recommend Exactly How Much More Data We Should Collect

This is a major required deliverable.

After evaluating our initial dataset, determine **how many additional golf shots we should collect and what those shots should look like**.

Do not simply tell me that more data is better.

I want a concrete, evidence-based recommendation for the next range session.

### Learning curves

Generate learning curves showing model performance against training set size.

Use appropriately repeated, leakage-safe splits.

For example, evaluate training subsets of roughly 15, 25, 35, 50, and 70 shots where feasible, accounting for the actual number of usable shots.

Determine whether performance is:

- Improving rapidly
- Improving slowly
- Plateauing
- Highly variable due to small sample sizes
- Limited by data quality instead of sample count

Attempt to estimate the value of collecting additional samples.

### Recommend a sample size

Tell me whether we should aim for something like:

- 30 additional shots
- 50 additional shots
- 100 additional shots
- 200 additional shots
- A larger dataset

Make your recommendation based on observed performance and uncertainty, not arbitrary rules.

If there is not enough evidence to produce a precise sample-size estimate, say so and recommend a sensible next batch followed by reassessment.

### What new shots should we collect?

Investigate:

- Whether we need more golfers.
- Whether we need more shots per existing golfer.
- Whether we need more variation in swing speed.
- Whether we need intentionally different shot shapes.
- Whether we should include mishits.
- Whether we should include practice swings.
- Whether we should keep using only seven-irons.
- Whether camera angle should be adjusted.
- Whether additional cameras would materially improve performance.
- Whether the new ±200g accelerometer will be useful.
- Whether any modality has poor data quality and should be prioritized for improvement.

Compare the value of additional sample quantity versus better measurements and better calibration.

### Required deliverable

Create:

`reports/data_collection_recommendations.md`

Include:

- Current usable dataset size.
- Learning curves.
- Current errors by target and modality.
- Estimated benefit of additional data.
- Recommended number of new shots.
- Recommended distribution across golfers and shot types.
- Recording and synchronization improvements.
- A practical plan for the next data collection session.
- Clear criteria for when we should collect more data again.

**I want you to give me a specific recommendation that our team can actually follow at the driving range.**

---

# 10. Golf Swing Coaching and Analysis

Beyond predicting shot measurements, I want the system to provide meaningful feedback on golf technique.

Eventually, our application should be able to break a swing down into critical phases and provide useful observations about each phase.

Examples include:

### Address

- Posture
- Spine inclination
- Hand positioning
- Head position
- Knee flexion

### Takeaway

- Club movement
- Hand path
- Early body rotation
- Tempo

### Top of backswing

- Shoulder rotation
- Hip rotation
- Hand position
- Shaft position
- Head stability

### Downswing

- Sequencing
- Hip and shoulder rotation
- Hand speed
- Head movement
- Body posture

### Impact

- Club path and speed where measurable
- Body position
- Hand position
- Head movement
- Timing

### Follow-through

- Rotation
- Balance indicators
- Club trajectory
- Body stability

Identify both strengths and potential improvements.

**Do not give unsupported coaching advice.**

Separate:

- Directly measured movement
- Model-estimated movement
- Established golf biomechanics concepts
- Heuristic coaching suggestions
- Speculative interpretations

Avoid claiming that a measured movement caused a particular shot result unless the data supports that relationship.

The goal is useful, grounded feedback, not generic AI-generated encouragement.

---

# 11. Professional Golfer Comparison

Eventually, I want users to compare their swings against professional golfers.

Investigate competitors such as AI golf swing coaching applications, Swing Sensei / Golf Sensei, Garmin golf wearables, and established swing-analysis tools for relevant product ideas.

Explore:

- Side-by-side swing videos
- Synchronized playback
- Matching swing phases
- Pose overlays
- Club-path overlays
- Differences in shoulder and hip rotation
- Head movement comparisons
- Shaft positions
- Tempo comparisons
- Normalized swing alignment
- Body-position similarity
- Reference movement ranges

Professional videos should be appropriately licensed or user-provided.

Develop a comparison method that accounts for camera angle, body size, frame rate, and swing timing.

Do not treat every difference from a professional golfer as a mechanical fault.

This is a future product feature. Research and implement the necessary underlying algorithms first.

---

# 12. Practice Swing Mode

A major long-term goal is making the system useful **without needing to hit a golf ball**.

Ideally, a golfer could perform a practice swing with only an iPhone, or an iPhone plus Apple Watch, and receive useful feedback.

Explore prediction of:

- Swing speed
- Tempo
- Hand speed
- Clubhead speed
- Swing plane characteristics
- Head stability
- Body mechanics
- Expected carry distance, where defensible

A practice swing does not provide actual ball launch or impact data.

Therefore, the predicted distance should be presented as an uncertain estimate based on swing mechanics, not as an actual ball-flight measurement.

Evaluate these models using only pre-impact or practice-swing-compatible features.

If real practice swings are absent from the current dataset, explain that limitation and propose how we should collect and validate them.

---

# 13. Develop Research Visualizations Now, a Polished Application Later

I want strong visuals, but a production user interface is not our immediate priority.

For now, generate useful artifacts such as:

- Annotated swing videos
- Slow-motion video segments
- Tracked clubhead paths
- Golf-ball trajectories
- Body pose overlays
- Head-position traces
- Swing phase images
- 3D pose animations
- Reconstructed human motion
- Model prediction charts
- Actual-versus-predicted plots
- Model comparison tables
- Feature importance plots
- Learning curves
- Diagnostic examples of tracking failures

Save outputs to organized folders.

Eventually, I want an application where users can:

- Select or upload a swing.
- View it in slow motion.
- Advance through critical swing phases with buttons.
- View the golfer as a rendered human or 3D avatar.
- See the club path.
- Visualize head movement.
- See AI coaching observations for each phase.
- Compare their swing side by side with a pro.
- Overlay their pose with a pro's pose.
- View predicted shot metrics.
- Perform practice-swing analysis.
- Choose between camera-only and sensor-enhanced modes.

But **do not prioritize building this full application until the underlying computer vision and machine learning methods are working and validated**.

A basic research viewer is fine if it helps inspect videos, labels, or tracking failures.

Do not spend substantial development time on polished UI components at the expense of model accuracy.

---

# 14. Complete GitHub-Ready, Collaborative Codebase

Create an actual, organized software project that can be uploaded to GitHub for collaboration with the rest of my team.

I do not want important work to exist only in your conversation, a temporary notebook, or an inaccessible internal environment.

**Write all actual source files and generated outputs into the project workspace.**

Organize the repository logically, for example:

    golf-shot-modeling/
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    ├── .gitignore
    ├── configs/
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── manifests/
    ├── src/
    │   ├── ingestion/
    │   ├── synchronization/
    │   ├── computer_vision/
    │   ├── ball_tracking/
    │   ├── club_tracking/
    │   ├── pose_estimation/
    │   ├── biomechanics/
    │   ├── feature_extraction/
    │   ├── sensor_processing/
    │   ├── modeling/
    │   └── evaluation/
    ├── experiments/
    ├── annotations/
    ├── models/
    ├── outputs/
    │   ├── videos/
    │   ├── images/
    │   ├── pose_3d/
    │   └── plots/
    ├── reports/
    ├── tests/
    └── scripts/

Adapt this structure where appropriate rather than following it rigidly.

### Requirements

- Write modular Python code.
- Use configuration files rather than hardcoded absolute paths.
- Provide reproducible dependencies and setup instructions.
- Use appropriate random seeds.
- Save model configurations and evaluation results.
- Keep a record of dataset versions and preprocessing decisions.
- Add tests for important components.
- Make it easy for teammates to reproduce experiments.
- Document hardware assumptions.
- Document calibration requirements.
- Use readable, maintainable code.
- Save annotations and intermediate features to avoid repeatedly processing expensive videos.
- Add caching where appropriate.
- Avoid unnecessary dependencies.
- Provide practical commands for running major pipelines.

### GitHub and large files

Our videos may be hundreds of megabytes each.

Do not blindly commit large raw videos, generated videos, model checkpoints, or temporary caches into ordinary Git history.

Create an appropriate `.gitignore`.

Use Git LFS or documented external dataset storage where necessary.

Ensure teammates can clone the repository and reproduce the workflow after obtaining the dataset separately.

Do not publish the repository or upload private recordings to external services without my approval.

### Autonomous file creation

You have permission to create:

- Python source files
- Tests
- Configuration files
- Documentation
- Experiment scripts
- Research notebooks
- Annotation files
- Derived datasets
- Plots
- Annotated videos
- 3D visualizations
- Model checkpoints
- Reports

You may create and modify project files as needed, but do not overwrite or delete irreplaceable raw recordings.

Keep original recordings unchanged.

---

# 15. Autonomous Experimentation and Continuous Improvement

I want you to take initiative.

Do not simply implement the first idea and consider the project complete.

I expect you to:

1. Understand the dataset.
2. Research appropriate technical methods.
3. Implement a reliable baseline.
4. Run actual experiments.
5. Analyze results.
6. Identify weaknesses.
7. Propose and implement improvements.
8. Compare improvements rigorously.
9. Preserve the best-performing validated methods.
10. Continue to the next high-value experiment.

Explore several promising directions rather than endlessly tuning a single model.

Prioritize expected information gained per experiment.

If a method performs poorly, investigate why instead of blindly increasing model complexity.

If a feature provides little predictive value, document that finding.

If video alone performs nearly as well as video plus sensors, that is an important result.

If sensors provide major improvements, explain where and why.

If the camera cannot reliably measure a specific quantity because of geometry or visibility, document the limitation and investigate alternatives.

**Do not sacrifice scientific validity for a more impressive-looking result.**

Avoid unbounded experimentation with no evaluation plan. Maintain clear checkpoints, experiment records, and stopping criteria.

If a task cannot run because of missing dependencies, compute limitations, permissions, or unavailable files, explain the obstacle and complete the portions that are feasible.

Do not report experiments as completed unless they actually ran successfully.

---

# 16. Important Questions You Should Answer Through Research

As you work, try to answer the following:

1. How accurately can we predict carry distance from iPhone video alone?
2. How accurately can we estimate clubhead speed from video?
3. Can we track the ball reliably at our available frame rate and camera angle?
4. Can we estimate launch characteristics from its initial flight?
5. Which camera-derived features are most useful?
6. How much information does pose estimation provide beyond direct club tracking?
7. Can we reconstruct useful 3D swing motion from one camera?
8. Can we quantify head movement accurately?
9. How much do Apple Watch measurements improve predictions?
10. How much does the club IMU improve predictions?
11. Do all three modalities outperform camera-only models?
12. Does sensor fusion help with impact detection?
13. Can we estimate swing speed and expected carry from practice-swing-compatible features?
14. Which predictive targets are feasible with approximately 70 shots?
15. Which results generalize to new shots, new sessions, or new golfers?
16. What are the main causes of prediction errors?
17. Which additional camera or sensor measurements would be most valuable?
18. How many more shots should we collect?
19. Which types of shots and golfers should we prioritize?
20. What is the most compelling, technically defensible product we can build from these results?

---

# 17. Research Deliverables

Produce the following as real files in the project workspace.

### A. Dataset audit

`reports/dataset_audit.md`

Include usable shots, filenames, modality coverage, data-quality issues, and unresolved questions.

### B. Master shot manifest

A machine-readable table connecting videos, TrackMan measurements, Apple Watch recordings, and club sensor recordings.

### C. Computer vision pipeline

Working code for video processing, club tracking, body pose, swing phases, and ball tracking where feasible.

### D. Sensor-processing pipeline

Working code for parsing, synchronizing, inspecting, and extracting IMU and Apple Watch features.

### E. Reproducible ML benchmarks

Experiments comparing all viable model families and sensor combinations.

### F. Research results report

`reports/model_results.md`

Include model comparisons, metrics, sample counts, uncertainty estimates, successful experiments, failed experiments, and recommended next steps.

### G. Annotated videos and visualizations

Generate useful examples demonstrating the strongest camera tracking, pose estimation, head tracking, swing-phase detection, and 3D reconstruction methods.

### H. Data collection recommendations

`reports/data_collection_recommendations.md`

Provide an evidence-based estimate of how many additional shots our group should collect, with a concrete plan for our next range session.

### I. GitHub-ready repository

A complete, reproducible project containing source code, experiments, tests, configuration files, and documentation.

### J. Final research summary

Explain:

- What works.
- What doesn't work.
- Which models performed best.
- Which sensors contributed meaningful information.
- Which camera measurements are reliable.
- Which features are experimental.
- Which results are statistically uncertain.
- What we should collect next.
- What should be built into the final application.

---

# 18. Required Execution Order

Follow this order unless the available data strongly justifies a change.

**Phase 1: Dataset inspection and clarification**

Inspect files, identify shots, verify video frame rates, match modalities, check synchronization, and flag inconsistencies.

Stop and ask me about critical ambiguities.

**Phase 2: Camera-only research pipeline**

Develop reliable video decoding, swing-phase detection, body pose estimation, head tracking, club tracking, and ball tracking where feasible.

Save diagnostic visualizations and annotated clips.

**Phase 3: Sensor processing and synchronization**

Process watch and club IMU recordings, align them with videos, and extract useful motion features.

**Phase 4: Model benchmarking**

Train and evaluate appropriate camera-only, sensor-only, and multimodal models.

Use leakage-safe evaluation and realistic baselines.

**Phase 5: Advanced experimentation**

Investigate physics-informed video measurements, improved temporal tracking, pretrained representations, 3D pose estimation, multimodal alignment, and other promising approaches.

Retain improvements that survive valid evaluation.

**Phase 6: Data collection recommendations**

Analyze learning curves and prediction errors.

Tell me how many additional shots we should collect, which golfers should collect them, and how to improve the measurements.

**Phase 7: Research presentation and reproducibility**

Generate clear plots, annotated videos, model reports, and instructions so our team can inspect and reproduce the findings.

**Phase 8: Final application and demo**

Only after the research pipeline is sufficiently mature, build a polished interactive application with swing replay, coaching, pro comparisons, 3D visualization, and practice-swing analysis.

This final phase is not the immediate priority.

---

# 19. Communication and When to Ask Me Questions

Work independently whenever you can make scientifically justified progress.

However, **do not make important assumptions that I could easily clarify**.

Pause and ask me questions when:

- Dataset files cannot be matched reliably.
- There are material discrepancies between video and TrackMan measurements.
- Hardware metadata is unclear.
- Important calibration information is missing.
- You need to know which golfer performed which shots.
- A proposed correction could alter ground-truth labels.
- A missing detail could invalidate an experiment.
- You need a significant decision between two substantially different technical approaches.
- You need permission for an external upload, publication, or destructive operation.

When possible, present a few clearly organized questions with the evidence supporting them.

Do not interrupt for minor technical implementation choices you can resolve yourself.

Provide progress summaries at natural milestones, explaining what was implemented, what was actually tested, what was learned, and what comes next.

Be transparent about failed experiments and uncertain findings.

---

# 20. Final Priorities

If you must make tradeoffs, prioritize in this order:

**Priority 1 — Data integrity**

Correctly identify, clean, and synchronize the actual shots. Never compromise data validity.

**Priority 2 — Exceptional camera-only analysis**

Extract as much meaningful information as possible from our high-frame-rate iPhone video, particularly club motion, ball tracking, biomechanics, and swing phases.

**Priority 3 — Strong, scientifically valid prediction models**

Compare many promising techniques and identify the best out-of-sample predictions of TrackMan measurements.

**Priority 4 — Sensor fusion**

Determine whether Apple Watch and club-mounted IMU data genuinely improve predictions or analysis.

**Priority 5 — Recommendations for additional data**

Tell us exactly what to collect next and how much additional data is likely to be useful.

**Priority 6 — Advanced visualization and coaching**

Develop pose estimation, head tracking, 3D representations, annotated swing phases, and useful biomechanical analysis.

**Priority 7 — Polished application and demo**

Build the interactive, user-facing application only after the underlying research and models are sufficiently mature.

---

# 21. Begin Now

Start by inspecting the entire project directory and dataset.

Do not assume the file organization, sensor availability, or number of usable shots.

First:

1. Inventory the available files.
2. Examine video metadata and actual frame timing.
3. Inspect TrackMan labels and sensor schemas.
4. Establish or propose shot associations.
5. Identify missing, inconsistent, or ambiguous information.
6. Create the initial dataset audit and master manifest.
7. Ask me to resolve any critical uncertainties before major training.

Once those issues are resolved, proceed through the research phases autonomously.

**Do not stop after writing a plan. Actually write and execute code, produce research artifacts, compare models, and improve the system as far as the available environment and data allow.**

My ultimate goal is to determine how much useful golf swing and shot information we can extract from inexpensive sensors—especially a high-frame-rate iPhone camera—and use those findings to build an impressive, intelligent, affordable golf analysis product.

Treat this as a serious research and engineering project. Be creative, rigorous, skeptical of your own results, and ambitious about what we can accomplish.