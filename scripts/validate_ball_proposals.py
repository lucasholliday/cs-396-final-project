"""Independent visual spot checks plus a frozen stationary-tail rejection diagnostic.
Annotations were made from unmarked frames, not generated trajectory overlays.
They are approximate single-analyst observations, not external ground truth.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def moving_prefix(observations):
    if len(observations)<2:return observations
    p=np.array([[o['x'],o['y']] for o in observations]);t=np.array([o['playback_s'] for o in observations])
    speed=np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(t)
    # End at first sustained near-stationary interval. Freeze before other-session assessment.
    low=speed<200
    for i in range(len(low)-2):
        if low[i:i+3].all():return observations[:i+1]
    return observations

if __name__=='__main__':
    refs=pd.read_csv(ROOT/'annotations/ball_reference_points.csv');rows=[]
    for r in refs.itertuples():
        if r.visibility!='flight_visible':continue
        data=json.loads((ROOT/f'data/processed/ball/{r.shot_id}.json').read_text());obs=data['observations']
        row=dict(shot_id=r.shot_id,panel=r.panel,reference_s=r.playback_s,reference_x=r.x,reference_y=r.y,annotation_uncertainty_px=r.uncertainty_px,matched=False,error_px=np.nan)
        if obs:
            times=np.array([p['playback_s'] for p in obs]);i=int(np.argmin(abs(times-r.playback_s)))
            if abs(times[i]-r.playback_s)<.006:
                row.update(matched=True,time_error_s=float(times[i]-r.playback_s),error_px=float(np.hypot(obs[i]['x']-r.x,obs[i]['y']-r.y)))
        rows.append(row)
    out=ROOT/'reports/ball_validation';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'point_errors.csv',index=False)
    results=[]
    for path in sorted((ROOT/'data/processed/camera_event_ball').glob('*.json')):
        d=json.loads(path.read_text());a=d['observations'];trim=moving_prefix(a)
        passed=len(trim)>=20 and trim[-1]['playback_s']-trim[0]['playback_s']>=.1 and trim[0]['y']-trim[-1]['y']>150
        results.append(dict(shot_id=d['shot_id'],before_points=len(a),moving_points=len(trim),retained=passed))
    pd.DataFrame(results).to_csv(out/'stationary_tail_filter.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(results).to_string(index=False))

    all_rows=[]
    for path in sorted((ROOT/'data/processed/ball').glob('*.json')):
        d=json.loads(path.read_text());a=d['observations'];t=moving_prefix(a)
        keep=len(t)>=20 and t[-1]['playback_s']-t[0]['playback_s']>=.1 and t[0]['y']-t[-1]['y']>150
        all_rows.append(dict(shot_id=d['shot_id'],original_points=len(a),moving_points=len(t),retained=keep))
    pd.DataFrame(all_rows).to_csv(out/'all_shot_tail_filter.csv',index=False)
