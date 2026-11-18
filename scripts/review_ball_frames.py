"""Export unannotated native-resolution flight frames for independent point review.
Frame selection uses fixed offsets from the watch proxy, never model trajectories or labels.
"""
import json,re,subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/ball_reference';out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').groupby('session',sort=False).head(1)
records=[]
for row in m.itertuples():
 start=row.video_peak_candidate_s-.65
 p=subprocess.run(['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/row.video_candidate),'-t','0.4','-an','-vf','crop=440:1040:500:700,showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-'],capture_output=True,check=True)
 frames=np.frombuffer(p.stdout,np.uint8).reshape(-1,1040,440)
 pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',p.stderr.decode())])+start
 pts=pts[:len(frames)]
 assert len(pts)==len(frames)
 tiles=[]
 for j,offset in enumerate([-.47,-.42,-.37]):
  i=int(np.argmin(abs(pts-(row.video_peak_candidate_s+offset))))
  im=cv2.cvtColor(frames[i],cv2.COLOR_GRAY2BGR)
  cv2.putText(im,f'{row.shot_id} / {j} / {pts[i]:.6f}s',(6,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
  tiles.append(im)
  records.append(dict(shot_id=row.shot_id,panel=j,playback_s=pts[i],crop_x=500,crop_y=700,frame_index=i,source=row.video_candidate))
 cv2.imwrite(str(out/f'{row.shot_id}.jpg'),np.hstack(tiles))
 print(row.shot_id,flush=True)
pd.DataFrame(records).to_csv(out/'frame_index.csv',index=False)
