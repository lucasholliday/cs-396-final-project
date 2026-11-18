"""Deterministic every-third-plus-last native flight review, without outcome-based selection."""
import json,re,subprocess,hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');out=ROOT/'outputs/contact_video_review';out.mkdir(exist_ok=True)
 overrides=json.loads((ROOT/'annotations/contact_video_timing_overrides.json').read_text())
 rows=[];inv=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path')
 for r in m.itertuples():
  last=m[m.session==r.session].shot_in_session.max()
  if (r.shot_in_session-1)%3 and r.shot_in_session!=last:continue
  ball=json.loads((ROOT/f'data/processed/ball/{r.shot_id}.json').read_text())
  departure=ball['departure_candidate_s']
  if departure is None:departure=r.video_peak_candidate_s-.5
  if r.shot_id in overrides:departure=overrides[r.shot_id]['departure_s']
  start=departure-.03;offsets=[0,.04,.08,.14,.22,.34]
  meta=out/f'{r.shot_id}.json'
  if meta.exists():
   old=json.loads(meta.read_text())
   if abs(old[0]['departure_unverified_s']-departure)<1e-6:rows.extend(old);continue
   import shutil
   archive=ROOT/'outputs/contact_video_review_timing_v1';archive.mkdir(exist_ok=True)
   for item in out.glob(r.shot_id+'*'):shutil.copy2(item,archive/item.name)
  proc=subprocess.run(['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/r.video_candidate),'-t','0.4','-an','-vf','crop=780:1100:300:650,showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-'],capture_output=True)
  if proc.returncode:
   # FFmpeg 4.3 requires a leading zero for duration.
   cmd=['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/r.video_candidate),'-t','0.4','-an','-vf','crop=780:1100:300:650,showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-']
   proc=subprocess.run(cmd,capture_output=True,check=True)
  frames=np.frombuffer(proc.stdout,np.uint8).reshape(-1,1100,780)
  pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',proc.stderr.decode())])[:len(frames)]+start
  tiles=[];shotrows=[]
  for j,delta in enumerate(offsets):
   i=int(np.argmin(abs(pts-(departure+delta))))
   native=frames[i];im=cv2.cvtColor(cv2.resize(native,(390,550)),cv2.COLOR_GRAY2BGR)
   cv2.putText(im,f'{r.shot_id} panel {j} PTS {pts[i]:.5f}',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
   tiles.append(im);cv2.imwrite(str(out/f'{r.shot_id}_{j}_native.png'),native)
   shotrows.append(dict(shot_id=r.shot_id,panel=j,playback_s=float(pts[i]),crop_x=300,crop_y=650,width=780,height=1100,source=r.video_candidate,source_sha256=inv.loc[r.video_candidate,'sha256'],departure_unverified_s=departure))
  cv2.imwrite(str(out/f'{r.shot_id}_sheet.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
  meta.write_text(json.dumps(shotrows,indent=2));rows.extend(shotrows);print(r.shot_id,flush=True)
 pd.DataFrame(rows).to_csv(out/'frame_index.csv',index=False)
