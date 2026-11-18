# Research summary and delivery status

The available-data research system is implemented and has been run. It preserves the original recordings, matches 76 strikes, evaluates 72 labeled outcomes, and exposes failures as well as successful examples. It is not yet a validated replacement for a launch monitor. The production application remains gated by tracking and external validation, as requested in the project brief.

## Findings
- Full-swing video features predict carry with nested MAE 25.45 yd within the recorded sessions and 31.93 yd on held-out golfers. Video + watch changes those to 26.60 and 29.87 yd. Video clip selection still uses watch timestamps.
- On the same 65 labeled shots with club coverage, all sensors achieve 28.58 yd held-out-golfer MAE, versus 31.15 for video and 29.99 for video + watch. Only five golfers contribute; incremental sensor gains are uncertain.
- Early-window models perform worse on new golfers (video 40.05 yd, video + watch 43.26). Individual pre-impact exclusion is unverified. These results do not support practice-swing carry prediction.
- Body pose and head projections are available; precise wrist measurements are unreliable. The camera can visibly capture early ball flight, but automatic tracking is inconsistent. Eight independently viewed airborne-ball panels show six localized points in Sid/Lucas and two misses in Andrew. Approximate annotations need a second reviewer.
- Native shaft tracking produces plausible short segments, not a verified full clubhead path. Club acceleration is saturated. Neither camera nor sensors currently support defensible metric clubhead speed, spin or calibrated launch measurements.
- Severe mishits are often predicted as normal shots. This is the central modeling failure, not a reason to discard short shots.

## Deliverables
| Requested component | Delivered artifact | Status |
|---|---|---|
| Dataset audit and association | reports/dataset_audit.md; data/manifests/shot_manifest.csv | Executed; source-label photographs unavailable |
| Sensor pipeline | scripts/detect_events.py, align_club.py, extract_features.py | Executed; missing tails retained as missing |
| Video/body/head pipeline | scripts/process_pose.py; outputs/pose/ and outputs/head/ | Executed; projected movement only |
| Ball and shaft tracking | scripts/track_balls.py, shaft_diagnostic.py; reports/ball_validation/ | Experimental, failures documented |
| Swing phases | scripts/phase_candidates.py; annotations/phase_candidates.csv | 52 top proposals / 24 abstentions; impacts unannotated |
| Reproducible predictions | experiments/baseline_v1/, nested_carry_v1/, early_swing_v1/ | 400 fixed + 26 nested comparisons; OOF results saved |
| Data quantity recommendation | reports/data_collection_recommendations.md | 100 new strikes in two reviewed batches, plus 20 tagged practice swings |
| 3D visualization | scripts/render_pose_3d.py; outputs/pose_3d/ | Estimated skeleton illustration, not validated reconstruction |
| Local research viewer | outputs/research_viewer/index.html; scripts/build_research_viewer.py | Generated for all 76 shots; browser automation unavailable, UI verification incomplete |
| Team reproducibility | README.md, configs/, tests/, requirements-lock.txt | Local Git checkpoints; nothing published |
| Coaching, human mesh, pro comparisons, polished product | No unsupported demo claims | Deferred until measurements and references support them |

## Use the research viewer
Open outputs/research_viewer/index.html on the laptop. It references local H.264 clips and contains all 76 shot records. Select a strike, step through review frames, inspect the 2D overlay and rotate the estimated skeleton. Predictions are held-out-golfer estimates, not training fits. Mark phases and export annotations as JSON; review resolution is 30 fps, so these marks are not precise native-frame impact measurements. Annotations are held in memory until exported; reload discards unsaved marks. No file is uploaded or modified by viewing.

## Why the remaining product claims are gated
Metric motion requires measured camera geometry, exact sensor-to-head geometry and reliable timing; standard club length alone is insufficient. Spin and clubhead-speed labels are absent. Practice swings are absent. Ground-truth body motion, clubhead points and exact impacts are absent. A professional reference video was not supplied and no unlicensed footage was acquired. Automatically manufacturing any of these would create misleading validation.

The next dataset should secure the SD card, preserve label exports/photos, identify every event, include camera calibration and start/end synchronization gestures, and add new golfers/days. First independently review native ball/club points and impact frames across golfers, then evaluate contact-sensitive features. Do not train on the unreviewed ball proposals or expand the polished interface ahead of this gate.

## Completion boundary
This delivery completes a reproducible initial research study and review tooling with the available recordings. It does not complete the long-term product objective. More tuning of these 72 outcomes is not an untouched validation set. Browser playback still requires a successful local UI check; that attempt timed out and was not repeated. The documented next work is measurement validation and additional acquisition, not a claim that current experimental tracking is production-ready.
