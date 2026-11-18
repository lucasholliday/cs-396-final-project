"""Confidence-masked image-plane observations for review, not coaching diagnoses."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def masked_smooth(values, valid, radius=2):
    """Smooth within contiguous valid runs; never bridge an occlusion."""
    result=np.full_like(values,np.nan,dtype=float)
    ids=np.flatnonzero(valid)
    if not len(ids):return result
    for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
        for j,i in enumerate(run):result[i]=np.median(values[run[max(0,j-radius):j+radius+1]],axis=0)
    return result

def observations(p):
    a=np.asarray(p['landmarks'],float);n=len(a)
    valid=np.isfinite(a[:,[0,7,8],:2]).all(axis=(1,2)) & (a[:,[0,7,8],3]>=.6).all(axis=1)
    xy=a[:,[0,7,8],:2].mean(axis=1)
    head=masked_smooth(xy,valid)
    # Reference must be in first 0.33 seconds; do not silently call later finish address.
    early=np.flatnonzero(valid & (np.arange(n)<10))
    reference=np.median(head[early],axis=0) if len(early)>=3 else np.array([np.nan,np.nan])
    delta=(head-reference)*[100,100]
    baseline='median visible head centroid in first ten review frames' if len(early)>=3 else 'unavailable: fewer than three valid opening frames'
    rows=[dict(frame=i,playback_s=p['clip_start_playback_s']+i/p['fps'],head_valid=bool(valid[i]),head_x=float(head[i,0]),head_y=float(head[i,1]),head_dx_image_pct=float(delta[i,0]),head_dy_image_pct=float(delta[i,1]),wrists_valid=bool((a[i,[15,16],3]>=.6).all())) for i in range(n)]
    return dict(reference=baseline,semantics='Estimated face centroid in image coordinates. Positive x is screen right; positive y is screen down. Image percent is not physical displacement, head rotation or gaze. Window opening is not verified address.',frames=rows,head_valid_fraction=float(valid.mean()),wrists_valid_fraction=float((a[:,[15,16],3]>=.6).all(axis=1).mean()))

if __name__=='__main__':
    out=ROOT/'data/processed/pose_observations';out.mkdir(exist_ok=True);summary=[]
    for path in sorted((ROOT/'data/processed/pose').glob('*.json')):
        p=json.loads(path.read_text());d=observations(p);d.update(shot_id=path.stem,pose_signature=p['signature'])
        (out/path.name).write_text(json.dumps(d).replace('NaN','null'))
        summary.append(dict(shot_id=path.stem,head_valid_fraction=d['head_valid_fraction'],wrists_valid_fraction=d['wrists_valid_fraction'],reference=d['reference']))
    pd.DataFrame(summary).to_csv(out/'coverage.csv',index=False);print(pd.DataFrame(summary).head().to_string(index=False));print('Saved',len(summary),'confidence-masked traces')
