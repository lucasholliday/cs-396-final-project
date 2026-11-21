"""Outcome-blind launch features in image coordinates and source playback time.
No physical clubhead speed, smash factor, launch elevation or spin is inferred.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def features(track,seed,body_scale):
 if not track or not seed:return {}
 t=np.array([p['playback_s']-seed['playback_s'] for p in track]);xy=np.array([[p['x']-seed['x'],seed['y']-p['y']] for p in track]);order=np.argsort(t);t=t[order];xy=xy[order];valid=(t>0)&(t<.7);t=t[valid];xy=xy[valid]
 if len(t)<3:return {}
 f={'tracked':1.,'track_duration':float(t[-1]-t[0]),'first_delay':float(t[0]),'last_delay':float(t[-1]),'count':float(len(t))}
 # Interpolate only across actual short observed gaps; never extrapolate a missing window.
 for time in [.02,.04,.06,.08,.12]:
  ix=np.searchsorted(t,time)
  if 0<ix<len(t) and t[ix]-t[ix-1]<.025:
   point=xy[ix-1]+(xy[ix]-xy[ix-1])*(time-t[ix-1])/(t[ix]-t[ix-1]);tag=f'{round(time*1000):03d}'
   f['dx_'+tag]=float(point[0]/1080);f['rise_'+tag]=float(point[1]/1920);f['body_dx_'+tag]=float(point[0]/body_scale);f['body_rise_'+tag]=float(point[1]/body_scale)
 early=t<=.06
 if early.sum()>=3:
  tt=t[early];q=xy[early];design=np.c_[tt,np.ones(len(tt))];co=np.linalg.lstsq(design,q,rcond=None)[0];res=np.sqrt(np.mean((design@co-q)**2));f.update(vx_early=float(co[0,0]/1080),vy_early=float(co[0,1]/1920),body_vx_early=float(co[0,0]/body_scale),body_vy_early=float(co[0,1]/body_scale),early_fit_error=float(res/1920),ray_angle=float(np.degrees(np.arctan2(co[0,0],co[0,1]))))
 # Constant-velocity pinhole proxy: displacement = a*t/(1+b*t).
 # b is an apparent perspective parameter, NOT deceleration or physical speed.
 use=t<=.15
 if use.sum()>=4 and np.ptp(t[use])>=.025:
  tt=t[use];q=xy[use];best=None
  for b in [0,1,2,4,8,16,32,64]:
   z=tt/(1+b*tt);a=(z[:,None]*q).sum(0)/np.dot(z,z);loss=np.mean(np.minimum(np.linalg.norm(z[:,None]*a-q,axis=1),40)**2)
   if best is None or loss<best[0]:best=(loss,b,a)
  loss,b,a=best;f.update(perspective_b=float(b),perspective_vx=float(a[0]/1080),perspective_vy=float(a[1]/1920),perspective_residual=float(np.sqrt(loss)/1920))
 f['size_early']=float(np.median([o.get('area',np.nan) for o in track[:min(4,len(track))]]))
 return f

if __name__=='__main__':
 rows=[]
 for p in sorted((ROOT/'data/processed/ball_launch_tracks').glob('*.json')):
  d=json.loads(p.read_text());sid=p.stem;pose=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text());lm=np.array(pose['landmarks']);height=np.median((lm[:20,[27,28],1].mean(1)-lm[:20,[0,7,8],1].mean(1))*1920);scale=max(300,float(abs(height)));f=features(d['observations'],d['seed'],scale);rows.append(dict(shot_id=sid,**f))
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');df=m[['shot_id']].merge(pd.DataFrame(rows),on='shot_id',how='left');df['tracked']=df.tracked.fillna(0);df.to_csv(ROOT/'data/processed/audited_launch_features.csv',index=False);print(df.shape)
