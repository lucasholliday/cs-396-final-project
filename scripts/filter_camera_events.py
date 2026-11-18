"""Camera-only candidate filtering via native post-departure trajectory proposals."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from track_balls import track

ROOT=Path(__file__).resolve().parents[1]
session='Sid 1';video='Video Data/sid_session_1.MOV';out=ROOT/'outputs/camera_events'
events=pd.read_csv(out/'sid_session_1_events.csv')
cfg=json.loads((ROOT/'configs/ball_tracking.json').read_text())
cfg.update(output_dir='camera_events/tracks',cache_dir='camera_event_ball')
results=[]
for i,t in enumerate(events.camera_disappearance_s):
    # Set search start to disappearance -0.75s. No watch signal enters proposal or filtering.
    row=SimpleNamespace(shot_id=f'sid_camera_event_{i+1:02d}',video_candidate=video,video_peak_candidate_s=t+.5)
    f=track(row,cfg);f['camera_disappearance_s']=t;results.append(f)
    print(row.shot_id,f['ball_track_available'],flush=True)
d=pd.DataFrame(results);d.to_csv(out/'sid_session_1_filtered.csv',index=False)
references=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
times=references[references.session==session].video_peak_candidate_s.to_numpy()
accepted=d[d.ball_track_available==1]
near=[int(np.argmin(abs(times-t))) for t in accepted.camera_disappearance_s if np.min(abs(times-t))<=1.5]
summary=dict(session=session,camera_candidates=len(d),trajectory_passed=len(accepted),
    near_reference_candidates=len(near),unique_reference_swings_matched=len(set(near)),reference_swings=len(times),
    interpretation='Event localization proposals, compared with reference windows at +/-1.5s; not exact impact accuracy or validated point tracking. No watch used in detection/filtering.')
(out/'sid_session_1_filtered_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
