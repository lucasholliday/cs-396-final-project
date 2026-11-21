"""Direction-symmetric early-launch proposals with shaft/body conflict penalties.
No physical clubhead speed or label-based push/mishit classification.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd,cv2
from link_ball_v6 import follow
ROOT=Path(__file__).resolve().parents[1]
CFG=dict(version=2,initial_elapsed_min=.004,initial_elapsed_max=.035,initial_radius_min=15,initial_radius_max=220,min_area=10,min_points=4,min_rise=40,min_duration=.015,shaft_distance=14,shaft_penalty=45,body_penalty=40,contact_grace=.025)

def line_distance(p,a,b):
 v=b-a;u=np.clip(np.dot(p-a,v)/max(1e-8,np.dot(v,v)),0,1);return np.linalg.norm(p-a-u*v)

def extract(sid):
 d=json.loads((ROOT/f'data/processed/ball_v7/{sid}.json').read_text());pose=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text());lm=np.array(pose['landmarks']);shaft_path=ROOT/f'data/processed/ball_club_v2/{sid}.json';shaft=json.loads(shaft_path.read_text()) if shaft_path.exists() else {};byframe={f['frame']:f['candidates'] for f in shaft.get('club_frames',[])};tracks=[]
 for seed in d['seeds'][:2]:
  if seed['score']<40:continue
  for f in d['frames']:
   dt=f['playback_s']-seed['playback_s']
   if not CFG['initial_elapsed_min']<=dt<=CFG['initial_elapsed_max']:continue
   for c in f['candidates']:
    if c.get('polarity')!=1 or c['area']<CFG['min_area']:continue
    distance=np.hypot(c['x']-seed['x'],c['y']-seed['y'])
    if not CFG['initial_radius_min']<distance<CFG['initial_radius_max'] or c['y']>=seed['y']:continue
    track=follow(d['frames'],f['frame'],c,seed)
    if len(track)<CFG['min_points']:continue
    xy=np.array([[o['x'],o['y']] for o in track]);t=np.array([o['playback_s'] for o in track]);span=t[-1]-t[0]
    if span<CFG['min_duration'] or xy[0,1]-xy[-1,1]<CFG['min_rise']:continue
    early=t<=seed['playback_s']+.08;q=xy[early]
    if len(q)<3:continue
    center=q.mean(0);_,_,vh=np.linalg.svd(q-center,full_matrices=False);perpendicular=float(np.median(abs((q-center)@vh[-1])));shaft_hits=body_hits=0
    for o in track[:16]:
     if o['playback_s']-seed['playback_s']<CFG['contact_grace']:continue
     point=np.array([o['x'],o['y']]);lines=byframe.get(o['frame'],[])
     shaft_hits+=any(line_distance(point,np.array(l['a']),np.array(l['b']))<CFG['shaft_distance'] for l in lines)
     pi=int(np.clip(round((o['playback_s']-pose['clip_start_playback_s'])*pose['fps']),0,len(lm)-1));hull=cv2.convexHull((lm[pi,[11,12,13,14,15,16,23,24],:2]*[1080,1920]).astype(np.float32));body_hits+=cv2.pointPolygonTest(hull,tuple(point),True)>8
    speed=np.linalg.norm(xy[-1]-xy[0])/span;score=min(len(track),12)*2+min(speed/1000,12)*1.5-perpendicular*.8-CFG['shaft_penalty']*min(1,shaft_hits/2)-CFG['body_penalty']*min(1,body_hits/2)
    tracks.append(dict(score=float(score),observations=track,seed=seed,shaft_conflicts=int(shaft_hits),body_conflicts=int(body_hits),ray_residual_px=perpendicular))
 if not tracks:return dict(shot_id=sid,status='abstain',observations=[],seed=None)
 tracks.sort(key=lambda r:-r['score']);best=tracks[0]
 if best['score']<0:return dict(shot_id=sid,status='abstain_conflicts',observations=[],seed=None)
 return dict(shot_id=sid,status='unreviewed_direction_symmetric_proposal',**best,score_margin=float(best['score']-tracks[1]['score']) if len(tracks)>1 else None,configuration=CFG)

if __name__=='__main__':
 out=ROOT/'data/processed/audited_ball_tracks';out.mkdir(exist_ok=True);results={}
 for sid in pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').shot_id:
  r=extract(sid);results[sid]=r;(out/f'{sid}.json').write_text(json.dumps(r));print(sid,len(r['observations']),r.get('shaft_conflicts'),r.get('body_conflicts'),flush=True)
 # Sparse point coverage only; these annotations are not independent ground truth.
 refs=pd.read_csv(ROOT/'annotations/contact_ball_points.csv');rows=[]
 for ref in refs.itertuples():
  obs=results[ref.shot_id]['observations'];dt=err=float('inf')
  if obs:
   o=min(obs,key=lambda o:abs(o['playback_s']-ref.playback_s));dt=abs(o['playback_s']-ref.playback_s)
   if dt<=.006:err=float(np.hypot(o['x']-ref.x,o['y']-ref.y))
  rows.append(dict(shot_id=ref.shot_id,panel=ref.panel,time_error=dt,error_px=err,within32=err<=32))
 dest=ROOT/'experiments/audited_tracking_v2';dest.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(dest/'reference_errors.csv',index=False);(dest/'config.json').write_text(json.dumps(CFG,indent=2));print('matches',sum(r['within32'] for r in rows),'/',len(rows),flush=True)
