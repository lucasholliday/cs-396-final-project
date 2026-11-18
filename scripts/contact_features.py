"""Detailed gyro transients and acceleration-censoring descriptors; no inferred impact truth."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
ROOT=Path(__file__).resolve().parents[1]

def extract_window(df,peak,cfg):
 t=df.t_ns.to_numpy()*1e-9;lo,hi=np.array(cfg['window_relative_to_club_gyro_peak_s'])+peak
 if lo<t[0] or hi>t[-1]:return None
 keep=(t>=lo)&(t<=hi);w=df.loc[keep];t=t[keep]-peak
 g=w[['gx_rad_s','gy_rad_s','gz_rad_s']].to_numpy();a=w[['ax_m_s2','ay_m_s2','az_m_s2']].to_numpy()
 clip=w.near_accel_limit.to_numpy().astype(bool);norm=np.linalg.norm(g,axis=1);dt=np.diff(t)
 grid=np.arange(t[0],t[-1],1/cfg['resample_hz']);gi=np.column_stack([np.interp(grid,t,g[:,i]) for i in range(3)])
 hp=sosfiltfilt(butter(3,cfg['gyro_highpass_hz'],btype='highpass',fs=cfg['resample_hz'],output='sos'),gi,axis=0)
 magnitude=np.linalg.norm(hp,axis=1);derivative=np.diff(g,axis=0)/dt[:,None];dmag=np.linalg.norm(derivative,axis=1)
 before=(t>=-.25)&(t<-.05);after=(t>.05)&(t<=.25)
 eps=1e-6;scale=np.sqrt(np.mean(norm**2))+eps
 pre=np.sqrt(np.mean(norm[before]**2));post=np.sqrt(np.mean(norm[after]**2))
 central=(grid>=-.15)&(grid<=.15);outer=(abs(grid)>=.2)
 peaks,_=find_peaks(magnitude,prominence=max(.05,float(np.quantile(magnitude,.75))),distance=round(.02*cfg['resample_hz']))
 derivative_norm=np.diff(norm)/dt
 direction=g/(norm[:,None]+eps);turn=np.linalg.norm(np.diff(direction,axis=0),axis=1)/dt
 unclipped=~(clip[:-1]|clip[1:]);ad=np.linalg.norm(np.diff(a,axis=0)/dt[:,None],axis=1)
 runs=np.diff(np.r_[False,clip,False].astype(int));starts=np.flatnonzero(runs==1);ends=np.flatnonzero(runs==-1)
 f=dict(contact_gyro_hp_rms_ratio=float(np.sqrt(np.mean(magnitude**2))/scale),
  contact_gyro_hp_peak_ratio=float(np.max(magnitude)/scale),contact_gyro_hp_center_ratio=float(np.mean(magnitude[central]**2)/(np.mean(magnitude[outer]**2)+eps)),
  contact_gyro_derivative_p95_ratio=float(np.quantile(dmag,.95)/scale),contact_gyro_post_pre_ratio=float(post/(pre+eps)),
  contact_gyro_deceleration_ratio=float(-np.quantile(derivative_norm,.05)/scale),contact_gyro_direction_change_p95=float(np.quantile(turn,.95)),
  contact_gyro_hp_peak_count=int(len(peaks)),contact_gyro_hp_peak_offset_s=float(grid[np.argmax(magnitude)]),
  contact_clip_fraction=float(clip.mean()),contact_clip_run_count=int(len(starts)),
  contact_clip_longest_run_s=float(max((t[min(b,len(t)-1)]-t[a] for a,b in zip(starts,ends)),default=0)),
  contact_clip_centroid_offset_s=float(t[clip].mean()) if clip.any() else 0.,
  contact_clip_post_minus_pre=float(clip[after].mean()-clip[before].mean()),
  contact_accel_unclipped_jerk_p95=float(np.quantile(ad[unclipped],.95)) if unclipped.sum()>5 else np.nan,
  contact_accel_unclipped_pair_fraction=float(unclipped.mean()))
 quality=dict(samples=len(w),max_gap_s=float(dt.max()),resample_hz=cfg['resample_hz'],gyro_clipped=int(w.near_gyro_limit.sum()),accel_clipped=int(clip.sum()))
 return f,quality

if __name__=='__main__':
 cfg=json.loads((ROOT/'configs/contact_quality.json').read_text());m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');cache={};rows=[];quality=[]
 for r in m.itertuples():
  if pd.isna(r.club_peak_candidate_s):continue
  if r.club_candidate not in cache:cache[r.club_candidate]=pd.read_csv(ROOT/r.club_candidate)
  result=extract_window(cache[r.club_candidate],r.club_peak_candidate_s,cfg)
  if result:
   f,q=result;rows.append(dict(shot_id=r.shot_id,**f));quality.append(dict(shot_id=r.shot_id,**q))
 pd.DataFrame(rows).to_csv(ROOT/'data/processed/contact_features.csv',index=False)
 out=ROOT/'experiments/contact_quality_v1';out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(quality).to_csv(out/'window_quality.csv',index=False)
 print('Extracted contact descriptors:',len(rows))
