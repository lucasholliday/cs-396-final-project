"""Whole-swing sensor features with provenance; saturation diagnostics kept separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/benchmark.json').read_text())
m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
review=pd.read_csv(ROOT/'annotations/shot_review.csv').set_index('shot_id')
inventory=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path').sha256
features=[];quality=[];cache={}
for row in m.itertuples():
    r=review.loc[row.shot_id]
    assert r.video_sha256==inventory[row.video_candidate]
    assert r.workbook_sha256==inventory['trackman_shot_labels.xlsx']
    assert abs(r.video_peak_candidate_s-row.video_peak_candidate_s)<1e-6
    f={'shot_id':row.shot_id}
    for modality,path,peak,axes,acol,timecol,scale in [
        ('watch',row.watch_candidate,row.watch_peak_candidate_s,['rotationRateX','rotationRateY','rotationRateZ'],['accelerationX','accelerationY','accelerationZ'],'seconds_elapsed',1),
        ('club',row.club_candidate,row.club_peak_candidate_s,['gx_rad_s','gy_rad_s','gz_rad_s'],['ax_m_s2','ay_m_s2','az_m_s2'],'t_ns',1e-9)]:
        if pd.isna(peak):continue
        if path not in cache:cache[path]=pd.read_csv(ROOT/path)
        df=cache[path];times=df[timecol].to_numpy()*scale
        low,high=peak-cfg['sensor_window_before_peak_s'],peak+cfg['sensor_window_after_peak_s']
        if low<times[0] or high>times[-1]:continue
        keep=(times>=low)&(times<=high);x=df.loc[keep];t=times[keep]
        for signal,cols in [('gyro',axes),('accel',acol)]:
            v=np.linalg.norm(x[cols],axis=1)
            if modality=='club' and signal=='accel':continue # Saturated signal excluded from initial predictors.
            f.update({f'{modality}_{signal}_{stat}':float(val) for stat,val in [
                ('max',v.max()),('p95',np.quantile(v,.95)),('rms',np.sqrt(np.mean(v*v))),
                ('area',np.trapezoid(v,t)),('peak_width_s',np.sum(v>.5*v.max())*np.median(np.diff(t))) ]})
        quality.append(dict(shot_id=row.shot_id,modality=modality,samples=len(x),start_s=low,end_s=high,
            max_gap_s=float(np.diff(t).max()),accel_limit_fraction=float(x.near_accel_limit.mean()) if modality=='club' else None,
            gyro_limit_fraction=float(x.near_gyro_limit.mean()) if modality=='club' else None))
    features.append(f)
out=ROOT/'data/processed';out.mkdir(exist_ok=True)
pd.DataFrame(features).to_csv(out/'sensor_features.csv',index=False)
pd.DataFrame(quality).to_csv(out/'sensor_window_quality.csv',index=False)
m['physical_shot_verified']=True
m['association_status']='accepted_user_order_plus_visible_swing'
m['training_eligible']=m.trackman_status=='registered'
m.to_csv(ROOT/'data/manifests/shot_manifest.csv',index=False)
print('Extracted',len(features),'shot feature rows;',m.training_eligible.sum(),'label-eligible; precise impacts unverified.')
