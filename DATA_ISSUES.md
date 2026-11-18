# Data issues

## User clarifications — 2026-11-17
- Nth actual strike within each named session maps to nth Excel row, including unregistered strikes. No practice swings. Devices started before strikes, not necessarily simultaneously.
- All golfers right handed, all seven iron, watch always left/lead wrist.
- User later specifies Wilson Staff CB 7 iron, sensor on shaft immediately below rubber grip. Nominal current Wilson reference length 37.25 inches (0.94615 m), loft 34°, lie 62°; actual year/custom length and chip offset unknown. `configs/equipment.json` preserves distinction. Overall club length alone does not establish exact sensor-to-head or sensor-to-body location.
- Follow-up resolves custom length: standard length confirmed. Sensor sits approximately one inch below lower/distal grip end. User cannot measure exact chip position now and can later. Model year/grip length/chip offset remain unknown; do not re-ask until precise metric calibration is needed.
- Five club sessions ended when SD card came loose; recorded prefix remains valid. Eduardo alone has completion summary. Missing tail must remain absent; retain video/watch-only shots.
- Acceleration saturation is expected from sensor range. Do not treat clipped peaks as true peaks.
- Videos transferred through Google Drive; user recalls mixed 120/240 fps capture. Exact per-file capture rate unknown. Use measured timestamps; validate real-time relationship using sensor events before reporting physical speed.

## Open audit issues
- Six MOVs nominal 240 fps but average playback ~159 fps and median PTS interval ~1/120 s. Capture clock still requires technical verification; no additional user question currently pending.
- Actual strike counts/timestamps remain unverified. Must reconcile 18/5/10/12/11/20 without shifting labels after missed detections.
- Missing labels: rohan_1_03 (Excel row 22), rohan_2_09 (33), andrew_1_05 (51, stale screen), andrew_1_06 (52). Preserve strike candidates; exclude missing targets.
- Source photos referenced by all 76 rows are absent. User-supplied labels retained; cannot independently check transcription. Preserve supplemental corrections for sid_1_08, rohan_2_05, lucas_1_09.
- Rohan 2 WatchMicrophone.mp4 fails ffprobe with `moov atom not found`; motion CSV is readable. Other watch audio durations differ from wrist timelines; audio alignment unverified.
- Watch folder times appear UTC despite America/Chicago metadata. Use epoch timestamps and verify events. Club time is monotonic nanoseconds since start, not UTC.
- Watch signal units/axis conventions need technical verification. Club units/ranges declared in metadata.
- No clubhead speed, spin or spin-axis ground truth. No real practice swings.
- Git resolves to unrelated home repository (master, no commits). Do not stage there.
- System Pillow binary incompatible architecture. Audit uses FFmpeg instead.

## Subsequent resolutions and limitations
- All 76 identity/order associations accepted for whole-swing analysis after contact-sheet review. Exact impact not annotated. 72 labeled shots benchmarked, 65 on matched club cohort.
- Club tails missing: sid_1_18; rohan_1_04–05; rohan_2_08–10; lucas_1_12; andrew_1_11. Valid preceding data retained. All 68 club windows contain near-acceleration-limit samples; club acceleration excluded from predictors.
- Portrait rotation metadata must be honored. Initial sparse audit thumbnails were squashed for contact-sheet display; actual pose inputs preserve aspect ratio and later review sheets are portrait-correct.
- Pose wrist confidence is poor (mean 18.9% of clip frames pass the both-wrists threshold), even though body/head detection is available. Do not claim precise hand/club biomechanics from these landmarks.
- Ball batch produced 36/76 gated proposals, not 36 validated tracks. Motion blur, club confusion and stationary-background tails remain. `annotations/ball_review_queue.csv` requires point review before ball features enter models.
- Watch axis extrema cluster near ~35 recorded rotation-rate units; possible sensor range limiting is not established. No watch saturation flags or independently verified export units. Initial regression uses recorded units and does not interpret peaks as physical angular speed.
- Local Git repository on main now initialized with approval; system-package problems resolved by isolated .venv. Earlier home-repository/system-binary findings are historical. No remote or push.

- Camera-only Sid filtering retained 18 proposals, of which 15 match reference neighborhoods and three are extras. Event 27 visibly tracks a leg/background; event 12 misses visible flight. No exact impact or point-tracking accuracy established.

- A universal impact offset from watch gyro peak is unsupported: first-shot native review panels capture different stages across sessions. Early-window experiments must not be called verified pre-impact. Existing whole-swing identity remains accepted.
- New point references are approximate AI annotations of unmarked frames with stated uncertainty, not external or second-reviewer ground truth. Entire ball tracks remain unvalidated.
- Browser automation unavailable/timed out. Local viewer media/data and JS checks passed, actual browser interaction verification remains unperformed.

## Contact and threshold studies — 2026-11-18
User cannot identify exact chunk/thin shot IDs; confirms short carry is a useful broad mishit indicator and lateral outcomes can indicate directional misses. These do not establish ground-first/face-edge/hosel labels. Carry cutoffs 80/100/120 yd are explicit outcome proxies, not contact-quality truth.
Later Eduardo ball positions fall outside the original narrow seed ROI. Five candidate departure windows were visually corrected only for the separate reviewed-ball pilot; original generated panels archived. 78 approximate AI-reviewed ball points across 31/32 deterministically selected shots are distinct from unreviewed detector outputs, with source hashes and 16 px uncertainty; second reviewer pending. They support an assisted feasibility study, not autonomous accuracy.
Acceleration clipping descriptors are newly evaluated as censoring patterns in a separate ablation; saturated amplitudes remain excluded as physical measurements. Exact impact times and actual high-frequency collision waveforms are unverified.

- Six-error audit finds no evidence sufficient for a label shift. Three false acceptances have 20.7–23.4° launch and 73–82 mph ball speeds; three false rejections have 4.3–11.7° launch and 100–107 mph. Possible early-flight feature ambiguity remains distinct from association errors. Small gyro alignment residuals do not prove exact impact or label correctness. See reports/misclassified_shot_audit.md.
- Demo phase navigation uses top proposals and illustrative offsets; setup/finish are window boundaries. Solid avatar is a rendering of uncertain monocular landmarks, not validated anatomy. Browser host unavailable on 2026-11-18; UI verification remains pending.

## Tracking/timing/application continuation — 2026-11-18
- Daytime balls can be dark silhouettes against bright sky. Positive-only difference detection is insufficient; dual-polarity candidates recover more reviewed points but also many distractors. V8 matches 52/78 sparse points; no claim of 52 correct full trajectories. Rohan 1 shots 2–5 abstain in the all-shot run.
- Twelve native-cadence visual departure intervals are provisional AI review, not independent exact impacts. Eduardo 19 bounds [381.9215, 381.9340] precede the older coarse 381.95-second override. Previous assisted results retain their original timing; no silent correction of benchmark features or labels.
- New-video pipeline model fits include all 72 labeled research shots. Existing-recording smoke tests are not independent prediction evaluation. Manually selected clip timing and new camera positions are unvalidated. Poor pose availability triggers prediction abstention.
- Native H.264 export initially rounded timestamps and lost a terminal decoded frame due to encoding/edit-list behavior. Final export uses source encoder time base, 60 kHz container time scale, no B frames/edit list, and explicit frame count. Every retained frame count and inter-frame cadence is checked against the source candidate window. Generated copies are lossy; originals unchanged.
- Browser host unavailable; Safari native fallback timed out. HTTP functional checks passed, browser layout and interactive controls remain unverified. Do not conflate backend checks with browser QA.
