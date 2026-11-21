"""Assisted native-cadence clubhead optical flow; image speed, not physical mph."""
import hashlib,json,re,subprocess
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]

def patch_match(a,b,point):
 x,y=np.rint(point).astype(int);r=12;search=90
 if min(x,y)<r or x+r>=a.shape[1] or y+r>=a.shape[0]:return None
 template=a[y-r:y+r+1,x-r:x+r+1]
 if template.std()<6:return None
 left=max(r,x-search);right=min(b.shape[1]-r-1,x+search);top=max(r,y-search);bottom=min(b.shape[0]-r-1,y+search)
 scores=cv2.matchTemplate(b[top-r:bottom+r+1,left-r:right+r+1],template,cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(scores)
 if score<.55:return None
 return np.array([left+loc[0],top+loc[1]],float)

def walk(frames,indices,point):
 result={indices[0]:np.array(point,float)};p=np.array(point,np.float32).reshape(1,1,2)
 for a,b in zip(indices[:-1],indices[1:]):
  q,ok,e=cv2.calcOpticalFlowPyrLK(frames[a],frames[b],p,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
  valid=q is not None and bool(ok[0,0])
  if valid:
   back,status,_=cv2.calcOpticalFlowPyrLK(frames[b],frames[a],q,None,winSize=(31,31),maxLevel=4)
   valid=back is not None and bool(status[0,0]) and np.linalg.norm(back-p)<=2 and np.linalg.norm(q-p)<=100 and e[0,0]<=40
  if not valid:
   candidate=patch_match(frames[a],frames[b],p[0,0])
   if candidate is None:break
   returned=patch_match(frames[b],frames[a],candidate)
   if returned is None or np.linalg.norm(returned-p[0,0])>4:break
   q=candidate.astype(np.float32).reshape(1,1,2)
  x,y=q[0,0]
  if not (0<=x<540 and 0<=y<960):break
  result[b]=q[0,0].astype(float);p=q
 return result

if __name__=='__main__':
 sid='sid_1_01';review=json.loads((ROOT/f'annotations/club_keypoints_{sid}.json').read_text());pose=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text());import pandas as pd
 row=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').set_index('shot_id').loc[sid];out=ROOT/'outputs/native_club';out.mkdir(exist_ok=True);cache=out/f'{sid}_frames.npz';start=pose['clip_start_playback_s']
 if not cache.exists():
  cmd=['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/row.video_candidate),'-t','3.5','-an','-vf','scale=540:960,showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-']
  proc=subprocess.run(cmd,capture_output=True,check=True);frames=np.frombuffer(proc.stdout,np.uint8).reshape(-1,960,540);pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',proc.stderr.decode())])+start;assert len(pts)>=len(frames),(len(frames),len(pts));pts=pts[:len(frames)];assert np.all(np.diff(pts)>0);np.savez_compressed(cache,frames=frames,pts=pts)
 data=np.load(cache);frames=data['frames'];pts=data['pts'];cap=cv2.VideoCapture(str(ROOT/f'outputs/pose/{sid}_input.mp4'));inputs=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  inputs.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
 cap.release();anchors=[];review_frame_map=[]
 for frame_index,frame in enumerate(inputs):
  expected=start+frame_index/pose['fps'];candidates=np.flatnonzero(abs(pts-expected)<.055)
  match=min(candidates,key=lambda j:np.mean(abs(frames[j][::3,::3].astype(float)-frame[::3,::3])))
  review_frame_map.append(dict(frame=frame_index,native_frame=int(match),playback_s=float(pts[match])))
 for mark in review['points']:
  candidates=np.flatnonzero(abs(pts-mark['playback_s'])<.055)
  i=min(candidates,key=lambda j:np.mean(abs(frames[j][::3,::3].astype(float)-inputs[mark['frame']][::3,::3])))
  anchors.append(dict(native_frame=int(i),playback_s=float(pts[i]),head=mark['head'],review_frame=mark['frame'],time_adjustment_s=float(pts[i]-mark['playback_s'])))
 observations={};disagreements=[]
 for a,b in zip(anchors[:-1],anchors[1:]):
  lo,hi=a['native_frame'],b['native_frame']
  if hi<=lo:continue
  forward=walk(frames,list(range(lo,hi+1)),a['head']);backward=walk(frames,list(range(hi,lo-1,-1)),b['head'])
  for i in range(lo,hi+1):
   if i in forward and i in backward:
    error=float(np.linalg.norm(forward[i]-backward[i]));disagreements.append(error)
    if error>7:continue
    w=(i-lo)/(hi-lo);point=forward[i]*(1-w)+backward[i]*w;status='bidirectional_optical_flow'
   else:continue
   observations[i]=dict(frame=i,playback_s=float(pts[i]),x=float(point[0]*2),y=float(point[1]*2),status=status,agreement_px_native=error*2)
 for a in anchors:
  i=a['native_frame'];observations[i]=dict(frame=i,playback_s=a['playback_s'],x=a['head'][0]*2,y=a['head'][1]*2,status='approximate_visual_anchor',agreement_px_native=None)
 records=sorted(observations.values(),key=lambda r:r['frame'])
 for a,b in zip(records[:-1],records[1:]):
  dt=b['playback_s']-a['playback_s']
  if b['frame']==a['frame']+1 and 0<dt<.02 and a['status']==b['status']=='bidirectional_optical_flow':b['speed_px_per_playback_s']=float(np.hypot(b['x']-a['x'],b['y']-a['y'])/dt)
 result=dict(shot_id=sid,tracking_version=2,review_frame_map=review_frame_map,anchors=anchors,observations=records,total_native_frames=len(frames),tracked_frames=len(records),source=str(row.video_candidate),review_sha256=hashlib.sha256((ROOT/f'annotations/club_keypoints_{sid}.json').read_bytes()).hexdigest(),semantics='Assisted optical-flow pilot. 1080x1920 image coordinates and source playback PTS. No depth/scale/real-time calibration; do not convert to mph. Forward/backward agreement is an internal check, not independent tracking accuracy.')
 (out/f'{sid}.json').write_text(json.dumps(result));print('native frames',len(frames),'tracked',len(records),'speed samples',sum('speed_px_per_playback_s' in r for r in records),flush=True)
 tiles=[]
 for t in [0,1.4,1.5,1.65,1.8,1.9,1.96,2.05,2.4]:
  i=int(np.argmin(abs(pts-start-t)));im=cv2.cvtColor(frames[i],cv2.COLOR_GRAY2BGR);o=observations.get(i)
  if o:cv2.circle(im,(round(o['x']/2),round(o['y']/2)),12,(0,255,255),2)
  cv2.putText(im,f'{t:.2f}s '+('tracked' if o else 'abstain'),(8,30),0,.7,(0,255,255),2);tiles.append(cv2.resize(im,(270,480)))
 cv2.imwrite(str(out/f'{sid}_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in [0,3,6]]))
