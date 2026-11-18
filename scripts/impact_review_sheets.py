"""Native-cadence local impact review sheets around automatic departure proposals.
No impact labels are inferred by this script. Human/AI review must state bounds.
"""
import json,re,subprocess
from pathlib import Path
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');errors=pd.read_csv(ROOT/'reports/misclassified_shot_audit.csv').shot_id.tolist();ids=set(errors+m.groupby('session',sort=False).head(1).shot_id.tolist());out=ROOT/'outputs/impact_review';out.mkdir(exist_ok=True);records=[];inv=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path').sha256
 for r in m[m.shot_id.isin(ids)].itertuples():
  d=json.loads((ROOT/f'data/processed/ball_v6/{r.shot_id}.json').read_text());seed=d['seeds'][0];start=seed['playback_s']-.035;x=int(seed['x'])-150;y=int(seed['y'])-180
  meta=out/f'{r.shot_id}.json'
  if meta.exists():records.extend(json.loads(meta.read_text()));continue
  proc=subprocess.run(['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/r.video_candidate),'-t','0.08','-an','-vf',f'crop=300:300:{x}:{y},showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-'],capture_output=True,check=True)
  frames=np.frombuffer(proc.stdout,np.uint8).reshape(-1,300,300);pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',proc.stderr.decode())])[:len(frames)]+start;tiles=[];shotrows=[]
  for i,frame in enumerate(frames):
   im=cv2.cvtColor(cv2.resize(frame,(240,240)),cv2.COLOR_GRAY2BGR);cv2.putText(im,f'{i} | {pts[i]:.6f}s',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(70,250,70),1);tiles.append(im);cv2.imwrite(str(out/f'{r.shot_id}_{i:02}_native.png'),frame)
   shotrows.append(dict(shot_id=r.shot_id,panel=i,playback_s=float(pts[i]),crop_x=x,crop_y=y,source=r.video_candidate,source_sha256=inv[r.video_candidate],seed_proposal_s=seed['playback_s']))
  while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
  cv2.imwrite(str(out/f'{r.shot_id}_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]));meta.write_text(json.dumps(shotrows,indent=2));records.extend(shotrows);print(r.shot_id,len(frames),flush=True)
 pd.DataFrame(records).to_csv(out/'frame_index.csv',index=False)
