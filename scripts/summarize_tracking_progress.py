"""Persist comparisons, timing sensitivity, and scientific figures for this checkpoint."""
import json,os
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__=='__main__':
 old=pd.read_csv(ROOT/'experiments/ball_tracking_v6/reference_errors.csv');rows=[dict(version='v4 final track',matches=int(old.v4_track_within_32px.sum()),references=len(old),kind='linked track')]
 for version in [6,7,8]:
  p=pd.read_csv(ROOT/f'experiments/ball_tracking_v{version}/linked_reference_errors.csv');rows.append(dict(version=f'v{version} final track',matches=int(p.within_32px.sum()),references=len(p),kind='linked track'))
 p=pd.read_csv(ROOT/'experiments/ball_tracking_v7/reference_errors.csv');rows.append(dict(version='v7 candidate ceiling',matches=int(p.v7_within_32px.sum()),references=len(p),kind='candidate recall only'))
 out=ROOT/'reports/tracking_progress';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4.5));bars=ax.barh([r['version'] for r in rows],[r['matches'] for r in rows],color=['#687a85','#9e6a63','#4d8279','#2b7551','#c4a66c']);ax.set(xlim=(0,85),xlabel='Reviewed reference points matched within 32 native pixels (out of 78)',title='Ball tracking development comparison: 31 reviewed swings');ax.invert_yaxis()
 for bar,r in zip(bars,rows):ax.text(r['matches']+1,bar.get_y()+bar.get_height()/2,str(r['matches']),va='center')
 fig.text(.02,.015,'Approximate AI references; repeated development set. Candidate recall is not track precision.',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'comparison.png',dpi=160)
 intervals=pd.read_csv(ROOT/'annotations/impact_intervals.csv');sensitivity=pd.read_csv(ROOT/'experiments/ball_timing_v1/departure_sensitivity.csv');tracking=[]
 for path in sorted((ROOT/'data/processed/ball_v8_tracks').glob('*.json')):
  d=json.loads(path.read_text());o=d['observations'];tracking.append(dict(shot_id=path.stem,points=len(o),status=d['status'],span_s=o[-1]['playback_s']-o[0]['playback_s'] if o else 0))
 pd.DataFrame(tracking).to_csv(out/'all_shot_coverage.csv',index=False)
 text=f'''# Tracking, timing and product progress — 2026-11-18

## Executed tracking experiment

Compared the original v4 tracker with wide-field positive-contrast v6, dual-polarity v7, and v8 initialization constraints. The reason for v6 failure was visually confirmed: a white golf ball becomes a dark silhouette against the bright daytime sky. V7 detects positive and negative contrast separately. V8 requires a sufficiently large bright departure candidate near the mat before allowing either polarity in later flight.

| Method | Sparse point matches / 78 |
|---|---:|
'''
 for r in rows:text+=f"| {r['version']} | {r['matches']} |\n"
 text+='''
The 78 approximate AI-reviewed points cover 31 shots, selected from a deterministic 32-shot pilot. This is a repeatedly inspected development set, not independent validation. A match requires timestamp difference ≤6 ms and image error ≤32 pixels; references have stated 16-pixel uncertainty. No automatic trajectory was promoted to prediction input.

Candidate recall of 76/78 exposes the remaining selection problem: many candidate points belong to shafts, limbs, shadows or background. V7 has a median 143.5 candidates at reviewed frames. V8 improves linked-point recall to 52/78 but does not establish full-track precision. Even a track that matches all three sparse review points can contain wrong early or late points. Two Rohan 1 windows lack trustworthy initial seeds. Preserved overlays distinguish cyan reviewed points from orange linked proposals under outputs/ball_v7_review and outputs/ball_v8_review.

V7 candidate extraction ran on all 76 windows. V8 all-shot coverage is recorded in reports/tracking_progress/all_shot_coverage.csv. Track counts are proposal coverage, not validated successes. Candidate search still uses watch-assisted windows and a camera-specific region; it is not a general autonomous camera-only detector.

## Native timing review

Reviewed 12 native-cadence ball/club crops: first shot from each session plus six classification errors. Source-bound ball-departure intervals are saved in annotations/impact_intervals.csv, alongside review panel indices and notes. Bounds span 8.3–20.8 ms. A second reviewer is pending. Exposure blur and occlusion remain, and ball departure is not an exact ground-contact or club-ball-contact measurement. Original manifest impact fields and outcome labels remain unchanged.

The mapped wrist-gyro peak follows these departure midpoints by roughly 0.017–1.006 seconds, varying substantially by session. A universal wrist-peak-to-impact offset is therefore unsuitable. Automatic disappearance falls inside 11/12 reviewed intervals, a diagnostic observation on selected examples, not general timing accuracy.

Eduardo 19's earlier coarse 381.95-second reviewed departure is later than the new interval [381.9215, 381.9340]. This illustrates timing uncertainty in the original assisted features; it does not establish an incorrect shot identity. Prior model results are preserved, not silently recomputed with altered feature timing.

## Timing sensitivity and alternative flight features

Shifting the assumed departure 20 ms later nearly doubles the typical first-point projected rate (median factor 1.92). A 20-ms earlier departure reduces it to a median factor 0.68. These are feature-sensitivity calculations, not claimed prediction errors.

Tested departure-independent median pairwise slopes between airborne points on the 26 shots with at least two reviewed points. All feature sets use the same shots and nested held-out-golfer evaluation. Carry MAE: original first-point features 29.32 yd, inter-frame slope 38.55 yd, combination 32.65 yd. Launch-angle MAE: 4.95°, 6.26°, 5.39°. The alternative did not improve regression and is not promoted. The cohort contains only two Andrew and three Lucas shots, so comparisons are uncertain and visibility-selected. Full classification outputs and fold selections are in experiments/ball_timing_v1.

## Product changes and verification

All 76 swings now have confidence-masked head-centroid traces. Smoothing never bridges invalid runs, and the reference must be supported by the first ten review frames. Coordinates are image fractions; face rotation, perspective and landmark error can move the centroid. No gaze, physical head height or causal swing diagnosis is inferred.

The demo shows synchronized head traces, source-bound provisional departure navigation for the 12 reviewed swings, and jump buttons for saved user phase markers. Other phase buttons remain heuristic. The avatar remains a stylized rendering of estimated world landmarks.

Added isolated new-video analysis through scripts/analyze_swing.py and a loopback-only upload application via Start Swing Lab.command. The pipeline selects a user-specified 3.5-second window, estimates pose, produces H.264 replay and reference-model predictions, and saves output under outputs/new_swings. Research manifests and evaluation rows remain unchanged. Model files are trusted local artifacts checked against their saved hashes.

The new-video estimate uses a fixed ridge model trained on all 72 labeled research shots. It is not the existing held-out estimate and has no validated new-day accuracy. An existing recording was reused for pipeline smoke tests, which prove execution only, not generalization. HTTP upload, processing, result delivery, range requests, scope and request-validation checks passed. Browser interaction/visual QA remains unavailable and must not be reported as passed.

## Next checkpoints

1. Independently review departure intervals and full automatic tracks, especially wrong early/late segments.
2. Validate automatic seed detection for the Rohan 1 failures and measure track precision, not just sparse recall.
3. Evaluate richer trajectory hypotheses on a frozen, separately annotated set before promoting automatic ball features.
4. Verify the local app in a browser, then test genuinely new recordings and collect calibrated new-day/golfer data.
'''
 (ROOT/'reports/tracking_timing_product_progress.md').write_text(text)
 print(pd.DataFrame(rows).to_string(index=False));print('all-shot track cache rows',len(tracking))
