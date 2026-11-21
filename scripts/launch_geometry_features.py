"""Fit image-projection descriptors, not metric launch physics, without reading outcomes."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]

def fit(track,seed,max_elapsed_s=.15):
 if not track or not seed:return {}
 t=np.array([o['playback_s']-seed['playback_s'] for o in track]);xy=np.array([[o['x']-seed['x'],seed['y']-o['y']] for o in track]);mask=(t>0)&(t<=max_elapsed_s);tt=t[mask];q=xy[mask];out={}
 if len(tt)<5 or np.ptp(tt)<.03:return out
 # d(t) = a*(t+offset)/(1+b*(t+offset)); no camera-distance assumption.
 init=q[0]/max(.005,tt[0]);lower=-min(.012,float(tt.min())*.8)
 def residual(v):
  z=(tt+v[3])/(1+v[2]*(tt+v[3]));return ((z[:,None]*v[:2]-q)/10).ravel()
 result=least_squares(residual,[float(np.clip(init[0],-29999,29999)),float(np.clip(init[1],50,29999)),10,0],bounds=([-30000,-1000,0,lower],[30000,30000,150,.012]),loss='soft_l1',max_nfev=150)
 ax,ay,b,shift=result.x;out.update(proj_vx=ax/1080,proj_vy=ay/1920,proj_depth_rate=b,proj_shift=shift,proj_fit_rmse=float(np.sqrt(np.mean(residual(result.x)**2))*10/1920),proj_angle=float(np.degrees(np.arctan2(ax,ay))),proj_asymptote_y=float(ay/max(1,b)/1920))
 for horizon in [h for h in [.03,.05,.1] if h<=max_elapsed_s]:
  z=(horizon+shift)/(1+b*(horizon+shift));tag=f'{round(horizon*1000):03d}';out['fit_dx_'+tag]=float(ax*z/1080);out['fit_rise_'+tag]=float(ay*z/1920)
 selected=[o for o in track if 0<o['playback_s']-seed['playback_s']<min(.08,max_elapsed_s) and o.get('area',0)>=10 and max(o.get('width',1),o.get('height',1))/max(1,min(o.get('width',1),o.get('height',1)))<3]
 if len(selected)>=3:
  time=np.array([o['playback_s']-seed['playback_s'] for o in selected]);r=np.sqrt([o['area'] for o in selected]);out['area_early_median']=float(np.median(r));out['area_log_slope']=float(np.polyfit(time,np.log(r),1)[0]);out['area_cv']=float(np.std(r)/np.mean(r))
 return out

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--tracks',default='ball_launch_tracks');parser.add_argument('--output',default='launch_geometry_features.csv');args=parser.parse_args();rows=[]
 for p in sorted((ROOT/'data/processed'/args.tracks).glob('*.json')):
  d=json.loads(p.read_text());rows.append(dict(shot_id=p.stem,**fit(d['observations'],d['seed'])))
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');df=m[['shot_id']].merge(pd.DataFrame(rows),on='shot_id',how='left');df.to_csv(ROOT/'data/processed'/args.output,index=False);print(df.shape)
