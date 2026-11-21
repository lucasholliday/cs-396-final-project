"""Experimental club-delivery candidate paths: NOT validated clubhead tracks.
Visual review found clothing/shaft/head identity switching even among survivors.
Do not train on these path speeds. Exact contact and physical speed are unverified."""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CFG=dict(before_s=.065,after_s=.004,min_points=4,min_displacement_px=65,max_finish_distance_px=90,max_step_gap_s=.018,min_speed_px_s=700,max_speed_px_s=28000,maximum_fit_error_px=14,minimum_shaft_support_frames=2)

def extract(sid):
 path=ROOT/f'data/processed/ball_launch_tracks/{sid}.json'
 if not path.exists():return dict(shot_id=sid,status='abstain_no_seed',observations=[])
 launch=json.loads(path.read_text());seed=launch['seed'];origin=np.array([seed['x'],seed['y']]);raw=json.loads((ROOT/f'data/processed/ball_v7/{sid}.json').read_text());shaftpath=ROOT/f'data/processed/ball_club_v2/{sid}.json';shafts=json.loads(shaftpath.read_text()) if shaftpath.exists() else {};lines={f['frame']:f['candidates'] for f in shafts.get('club_frames',[])};beam=[];finished=[]
 for frame in raw['frames']:
  dt=frame['playback_s']-seed['playback_s']
  if not -CFG['before_s']<=dt<=CFG['after_s']:continue
  nodes=[]
  for c in frame['candidates']:
   xy=np.array([c['x'],c['y']]);delta=xy-origin
   if c['area']<15 or not(abs(delta[0])<400 and -500<delta[1]<80) or np.linalg.norm(delta)<12:continue
   support=any(np.linalg.norm(xy-np.array(l['b']))<45 for l in lines.get(frame['frame'],[]))
   nodes.append(dict(**c,frame=frame['frame'],playback_s=frame['playback_s'],shaft_support=bool(support)))
  next_beam=[]
  for node in nodes:
   point=np.array([node['x'],node['y']]);best_paths=[]
   for path in beam:
    last=path[-1];gap=node['playback_s']-last['playback_s']
    if not 0<gap<=CFG['max_step_gap_s']:continue
    previous=np.array([last['x'],last['y']]);v=point-previous;speed=np.linalg.norm(v)/gap
    if not CFG['min_speed_px_s']<speed<CFG['max_speed_px_s'] or np.linalg.norm(point-origin)>np.linalg.norm(previous-origin)+10:continue
    if node['area']/last['area']>5 or last['area']/node['area']>5:continue
    if len(path)>1:
     before=np.array([path[-2]['x'],path[-2]['y']]);v0=previous-before
     if np.dot(v,v0)/max(1,np.linalg.norm(v)*np.linalg.norm(v0))<.8:continue
    best_paths.append(path+[node])
   best_paths.sort(key=lambda p:(-len(p),-sum(o['shaft_support'] for o in p)));next_beam.extend(best_paths[:2]);next_beam.append([node])
  beam=[p for p in beam if frame['playback_s']-p[-1]['playback_s']<CFG['max_step_gap_s']]+next_beam;beam=sorted(beam,key=lambda p:-len(p))[:250]
  for path in next_beam:
   if len(path)<CFG['min_points']:continue
   xy=np.array([[o['x'],o['y']] for o in path]);t=np.array([o['playback_s'] for o in path]);distance=float(np.linalg.norm(xy[-1]-origin));support=sum(o['shaft_support'] for o in path)
   if distance>CFG['max_finish_distance_px'] or np.linalg.norm(xy[-1]-xy[0])<CFG['min_displacement_px'] or support<CFG['minimum_shaft_support_frames']:continue
   tail=xy[-5:];tt=t[-5:]-t[-1];co=np.linalg.lstsq(np.c_[tt,np.ones(len(tt))],tail,rcond=None)[0];error=float(np.sqrt(np.mean((np.c_[tt,np.ones(len(tt))]@co-tail)**2)))
   if error>CFG['maximum_fit_error_px']:continue
   speed=float(np.linalg.norm(co[0]));score=len(path)*3+support*4-distance*.1-error;finished.append(dict(score=score,observations=path,delivery_image_speed=speed,delivery_fit_error=error,delivery_shaft_support=support,delivery_span=float(t[-1]-t[0])))
 if not finished:return dict(shot_id=sid,status='abstain_no_supported_identity',observations=[])
 best=max(finished,key=lambda q:q['score']);return dict(shot_id=sid,status='unreviewed_delivery_candidate',seed=seed,configuration=CFG,**best)

if __name__=='__main__':
 out=ROOT/'data/processed/club_delivery_candidates';out.mkdir(exist_ok=True);rows=[]
 for sid in pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').shot_id:
  r=extract(sid);(out/f'{sid}.json').write_text(json.dumps(r));rows.append(dict(shot_id=sid,delivery_available=float(bool(r['observations'])),**{k:v for k,v in r.items() if k.startswith('delivery_')}));print(sid,r['status'],len(r['observations']),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'data/processed/club_delivery_features.csv',index=False)
