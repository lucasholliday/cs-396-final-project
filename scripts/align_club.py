"""Robust provisional gyro peak alignment; excludes unmatched tails, never fills missing data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/audit.json').read_text())
manifest=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
reports=[]
for session,(_,club,watch) in cfg['sessions'].items():
    w=manifest[manifest.session==session].watch_peak_candidate_s.to_numpy()
    df=pd.read_csv(ROOT/club/'imu.csv')
    t=df.t_ns.to_numpy()*1e-9
    norm=np.linalg.norm(df[['gx_rad_s','gy_rad_s','gz_rad_s']],axis=1)
    idx,_=find_peaks(norm,height=15,distance=round(3/np.median(np.diff(t))))
    c=t[idx]
    candidates=[]
    for scale in np.linspace(.98,1.02,81):
        for offset in (w[:,None]-scale*c).ravel():
            err=np.abs(w[:,None]-(scale*c+offset))
            nearest=err.argmin(axis=1)
            residual=err[np.arange(len(w)),nearest]
            valid=residual<.4
            # Three-second candidate spacing makes duplicate matches impossible at .4 s tolerance.
            score=(int(valid.sum()),-float(np.median(residual[valid])))
            candidates.append((score,scale,offset,nearest,valid))
    candidates.sort(key=lambda x:x[0],reverse=True)
    _,scale,offset,nearest,valid=candidates[0]
    if valid.sum()>=3:
        scale,offset=np.polyfit(c[nearest[valid]],w[valid],1)
        err=np.abs(w[:,None]-(scale*c+offset));nearest=err.argmin(axis=1)
        residual=err[np.arange(len(w)),nearest];valid=residual<.4
    else:
        residual=np.full(len(w),np.nan)
    rows=manifest.index[manifest.session==session]
    for i,row in enumerate(rows):
        if valid[i]:
            manifest.loc[row,'club_peak_candidate_s']=c[nearest[i]]
            manifest.loc[row,'club_watch_residual_s']=residual[i]
    report=dict(session=session,watch_equals_scale_times_club_plus_offset=dict(scale=float(scale),offset_s=float(offset)),
        matches=int(valid.sum()),watch_events=len(w),club_candidates=len(c),
        residual_median_s=float(np.median(residual[valid])),residual_max_s=float(np.max(residual[valid])),
        unmatched_shots=manifest.loc[rows[~valid],'shot_id'].tolist(),
        status='provisional_gyro_alignment_not_verified_impact',
        club_end_watch_s=float(scale*t[-1]+offset))
    reports.append(report)
manifest.to_csv(ROOT/'data/manifests/shot_manifest.csv',index=False)
(ROOT/'data/manifests/club_alignment.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
