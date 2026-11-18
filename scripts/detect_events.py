"""Generate provisional gyro event candidates; never infer verified impacts from peaks."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT/'configs/audit.json').read_text())
media = {r['path']:r for r in json.loads((ROOT/'data/manifests/media_audit.json').read_text())}
manifest = pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
events, sensitivity = [], []
for session, (video, club, watch) in config['sessions'].items():
    video_start = datetime.strptime(media[video]['metadata']['format']['tags']['com.apple.quicktime.creationdate'], '%Y-%m-%dT%H:%M:%S%z').timestamp()
    for modality, path, axes, timecol, scale, thresholds in [
        ('watch',watch+'/WristMotion.csv',['rotationRateX','rotationRateY','rotationRateZ'],'seconds_elapsed',1,[20,25,30,35]),
        ('club',club+'/imu.csv',['gx_rad_s','gy_rad_s','gz_rad_s'],'t_ns',1e-9,[15,20,25,30])]:
        df = pd.read_csv(ROOT/path)
        times = df[timecol].to_numpy()*scale
        norm = np.linalg.norm(df[axes].to_numpy(),axis=1)
        distance = int(round(3/np.median(np.diff(times))))
        for threshold in thresholds:
            idx, props = find_peaks(norm,height=threshold,distance=distance)
            sensitivity.append(dict(session=session,modality=modality,threshold=threshold,events=len(idx),minimum_separation_s=3))
            if threshold != 25:
                continue
            for ordinal, ix in enumerate(idx,1):
                epoch = int(df.time.iloc[ix])*1e-9 if modality=='watch' else None
                events.append(dict(session=session,modality=modality,ordinal=ordinal,source=path,
                    sample_index=int(ix),elapsed_s=float(times[ix]),gyro_norm=float(norm[ix]),epoch_s=epoch,
                    video_time_from_metadata_s=epoch-video_start if epoch else None,
                    status='gyro_peak_candidate_not_verified_impact'))
        if modality=='watch':
            idx,_ = find_peaks(norm,height=25,distance=distance)
            expected = manifest.index[manifest.session==session]
            if len(idx)==len(expected):
                for row,ix in zip(expected,idx):
                    manifest.loc[row,'watch_peak_candidate_s'] = times[ix]
                    manifest.loc[row,'video_peak_candidate_s'] = int(df.time.iloc[ix])*1e-9-video_start
            # Never truncate or pad a mismatched detection count to match labels.
pd.DataFrame(events).to_csv(ROOT/'data/manifests/event_candidates.csv',index=False)
pd.DataFrame(sensitivity).to_csv(ROOT/'data/manifests/event_sensitivity.csv',index=False)
manifest.to_csv(ROOT/'data/manifests/shot_manifest.csv',index=False)
print(pd.DataFrame(sensitivity).to_string(index=False))
