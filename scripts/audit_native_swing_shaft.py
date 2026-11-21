"""Compare the automatic shaft directions with the existing Sid visual anchors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import sys
version=sys.argv[1] if len(sys.argv)>1 else 'v2'
src=ROOT/('data/processed/native_swing_shaft'+('_v2' if version=='v2' else ''))
out=ROOT/f'experiments/native_swing_shaft_{version}'
out.mkdir(exist_ok=True)
rows=[]
for path in sorted(src.glob('*.json')):
    d=json.loads(path.read_text())
    pts=np.asarray(d['times'])
    obs=d['candidates']
    phases=np.array_split(np.arange(len(pts)),3)
    observed={p['frame'] for p in obs}
    rows.append(dict(shot_id=path.stem,frames=len(pts),candidates=len(obs),
                     nominal_window_fps=(len(pts)-1)/np.ptp(pts),
                     **{f'third_{k+1}':len(observed.intersection(ids)) for k,ids in enumerate(phases)}))
pd.DataFrame(rows).to_csv(out/'coverage.csv',index=False)
d=json.loads((src/'sid_1_01.json').read_text())
marks=json.loads((ROOT/'annotations/club_keypoints_sid_1_01.json').read_text())['points']
mapping=json.loads((ROOT/'outputs/native_club/sid_1_01.json').read_text())['review_frame_map']
by_frame={p['frame']:p for p in d['candidates']}
checks=[]
for m in marks:
    timestamp=mapping[m['frame']]['playback_s']-d['start']
    ix=int(np.argmin(abs(np.array(d['times'])-timestamp)))
    assert abs(d['times'][ix]-timestamp)<.001
    v=np.subtract(m['head'],m['grip'])
    p=by_frame.get(ix)
    err=np.degrees(np.arccos(np.clip(np.dot(v,p['v'])/np.linalg.norm(v),-1,1))) if p else np.nan
    checks.append(dict(review_frame=m['frame'],source_frame=ix,dir_error_deg=err))
q=pd.DataFrame(checks)
q.to_csv(out/'sid_anchor_errors.csv',index=False)
print(pd.DataFrame(rows).sum(numeric_only=True))
print('anchors',len(q),'matched',q.dir_error_deg.count(),'within 10 degrees',(q.dir_error_deg<=10).sum())
