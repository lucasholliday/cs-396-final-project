"""Native-resolution single-shot bright-ball diagnostic, no physical-speed claims."""
import json
import re
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/ball'
out.mkdir(parents=True,exist_ok=True)
row=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').iloc[0]
start=float(row.video_peak_candidate_s)-.9
# ROI manually selected from inspected native portrait frame; coordinates in displayed video.
x,y,w,h=500,800,400,900
cache=out/'sid_1_01_native_roi.npz'
if cache.exists():
    d=np.load(cache);frames=d['frames'];pts=d['pts']
else:
    cmd=['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/row.video_candidate),'-t','0.65',
         '-an','-vf',f'crop={w}:{h}:{x}:{y},showinfo','-vsync','0','-pix_fmt','bgr24','-f','rawvideo','-']
    p=subprocess.run(cmd,capture_output=True,check=True)
    frames=np.frombuffer(p.stdout,dtype=np.uint8).reshape(-1,h,w,3)
    pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',p.stderr.decode())])[:len(frames)]+start
    if len(pts)!=len(frames): raise ValueError('Frame/timestamp count mismatch')
    np.savez_compressed(cache,frames=frames,pts=pts)
rows=[]
for i,frame in enumerate(frames):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,0,140]),np.array([179,100,255]))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    for j in range(1,n):
        xx,yy,ww,hh,area=stats[j]
        if 12<=area<=650 and 2<=ww<=40 and 2<=hh<=50 and max(ww,hh)/min(ww,hh)<3:
            rows.append(dict(frame=i,playback_s=pts[i],x=centers[j,0]+x,y=centers[j,1]+y,area=int(area),width=int(ww),height=int(hh)))
df=pd.DataFrame(rows)
df.to_csv(out/'sid_1_01_ball_candidates.csv',index=False)
# Diagnostic montage every ~0.04 playback seconds, with candidate boxes. Not an accepted trajectory.
samples=np.linspace(0,len(frames)-1,20).astype(int)
tiles=[]
for i in samples:
    im=frames[i].copy()
    for r in df[df.frame==i].itertuples():
        cv2.circle(im,(round(r.x-x),round(r.y-y)),12,(0,0,255),2)
    im=cv2.resize(im,(160,360))
    cv2.putText(im,f'{pts[i]:.3f}s',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    tiles.append(im)
cv2.imwrite(str(out/'sid_1_01_candidates.jpg'),np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,20,5)]))
result=dict(shot_id=row.shot_id,source=row.video_candidate,start_playback_s=start,frames=len(frames),
            native_roi=[x,y,w,h],candidate_count=len(df),status='unvalidated_bright_blob_candidates',
            method='HSV brightness/low-saturation components; manually specified native-resolution ROI',
            limitations='Club highlights/background/tee may be false positives; no accepted trajectory or speed')
(out/'diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
