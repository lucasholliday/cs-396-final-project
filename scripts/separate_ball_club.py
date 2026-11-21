"""Native-frame shaft evidence and conservative ball/club conflict truncation.
Proposals only: no inferred shaft endpoint is called a measured clubhead.
"""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CFG=dict(version=2,min_length_px=80,hand_line_distance_px=85,near_hand_px=280,ball_line_distance_px=12,contact_grace_s=.025,consecutive_conflicts=2)

def segment_distance(p,a,b):
 d=b-a;u=np.clip(np.dot(p-a,d)/max(1e-8,np.dot(d,d)),0,1);return float(np.linalg.norm(p-(a+u*d)))

def shaft_candidates(gray,background,hand,detector):
 lines=detector.detect(gray)[0]
 if lines is None:return []
 result=[];diff=cv2.absdiff(gray,background)
 for line in lines.reshape(-1,4):
  a=line[:2].astype(float);b=line[2:].astype(float)
  if np.linalg.norm(a-hand)>np.linalg.norm(b-hand):a,b=b,a
  d=b-a;length=np.linalg.norm(d)
  if length<CFG['min_length_px'] or np.linalg.norm(a-hand)>CFG['near_hand_px']:continue
  u=d/length;v=hand-a;perp=abs(u[0]*v[1]-u[1]*v[0])
  if perp>CFG['hand_line_distance_px'] or np.dot(v,u)>length-50:continue
  xy=np.rint(np.linspace(a,b,60)).astype(int);xy[:,0]=np.clip(xy[:,0],0,gray.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,gray.shape[0]-1)
  moving=float(np.mean(diff[xy[:,1],xy[:,0]]>18))
  if moving<.45:continue
  result.append(dict(a=(a+[300,650]).tolist(),b=(b+[300,650]).tolist(),length_px=float(length),hand_line_distance_px=float(perp),motion_fraction=moving,score=float(moving+length/400-perp/65)))
 return sorted(result,key=lambda r:-r['score'])[:8]

def truncate_conflicts(observations,club,seed):
 flags=[]
 for o in observations:
  lines=club.get(o['frame'],[]);p=np.array([o['x'],o['y']]);overlap=any(segment_distance(p,np.array(l['a']),np.array(l['b']))<CFG['ball_line_distance_px'] for l in lines)
  elapsed=o['playback_s']-seed['playback_s'] if seed else 0
  flags.append(bool(overlap and elapsed>CFG['contact_grace_s']))
 stop=None
 for i in range(1,len(flags)):
  if flags[i] and flags[i-1] and observations[i]['playback_s']-observations[i-1]['playback_s']<.022:stop=i-1;break
 kept=observations if stop is None else observations[:stop]
 return kept,flags,stop

def process(sid):
 old=json.loads((ROOT/f'data/processed/ball_v8_tracks/{sid}.json').read_text());pose=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text());meta=json.loads((ROOT/f'outputs/native_review/{sid}.json').read_text());out=ROOT/'data/processed/ball_club_v2';out.mkdir(exist_ok=True);target=out/f'{sid}.json';sig=hashlib.sha256(json.dumps([CFG,old,pose['signature']],sort_keys=True).encode()).hexdigest()
 if target.exists():
  cached=json.loads(target.read_text())
  if cached['signature']==sig:return cached
 cap=cv2.VideoCapture(str(ROOT/f'outputs/native_review/{sid}.mp4'));frames=[]
 while True:
  ok,frame=cap.read()
  if not ok:break
  frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
 cap.release();assert len(frames)==len(meta['source_pts']);background=np.median(np.array(frames[:30]),axis=0).astype(np.uint8);det=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);lm=np.array(pose['landmarks']);clubs={};records=[]
 # Inspect every native frame near the selected ball trajectory, including pre-departure.
 if old['observations']:
  lo=old['observations'][0]['frame']-8;hi=old['observations'][-1]['frame']+4
  for i in range(max(0,lo),min(len(frames),hi+1)):
   t=meta['source_pts'][i];pi=np.clip((t-pose['clip_start_playback_s'])*pose['fps'],0,len(lm)-1);a=int(pi);b=min(len(lm)-1,a+1);hand=((1-(pi-a))*lm[a,[15,16],:2].mean(0)+(pi-a)*lm[b,[15,16],:2].mean(0))*[1080,1920]-[300,650]
   candidates=shaft_candidates(frames[i],background,hand,det);clubs[i]=candidates;records.append(dict(frame=i,playback_s=t,candidates=candidates))
 kept,flags,stop=truncate_conflicts(old['observations'],clubs,old['seed'])
 result=dict(shot_id=sid,signature=sig,configuration=CFG,observations=kept,original_observations=old['observations'],club_frames=records,conflict_flags=flags,stop_index=stop,seed=old['seed'],status='truncated_possible_club_overlap' if stop is not None else old['status'],semantics='Unreviewed shaft line evidence; ball truncated after repeated overlap beyond contact grace. May remove real ball crossings. No complete clubhead identity or metric speed.')
 target.write_text(json.dumps(result));return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args();m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
 if not args.all:m=m.groupby('session',sort=False).head(1)
 results={}
 for sid in m.shot_id:
  r=process(sid);results[sid]=r;print(sid,len(r['original_observations']),len(r['observations']),r['stop_index'],flush=True)
 from link_ball_v6 import evaluate
 evaluate(results,10)
