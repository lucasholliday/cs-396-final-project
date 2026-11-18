"""Partial shaft line proposals in an inspected impact ROI, not a clubhead tracker."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/club';out.mkdir(parents=True,exist_ok=True)
d=np.load(ROOT/'outputs/ball/sid_1_01_native_roi.npz');frames=d['frames'];pts=d['pts']
parser=argparse.ArgumentParser();parser.add_argument('--motion',action='store_true');args=parser.parse_args()
suffix='_motion' if args.motion else ''
background=cv2.cvtColor(np.median(frames[:25],axis=0).astype('uint8'),cv2.COLOR_BGR2GRAY)
rows=[];tiles=[]
for i,frame in enumerate(frames):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,60,150)
    if args.motion:edges=cv2.bitwise_and(edges,((cv2.absdiff(gray,background)>25)*255).astype('uint8'))
    lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=60,minLineLength=90,maxLineGap=15)
    candidates=[]
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            dx,dy=float(x2-x1),float(y2-y1)
            angle=np.degrees(np.arctan2(dy,dx))%180
            length=np.hypot(dx,dy)
            if 45<angle<85 and (y1+y2)/2>400:
                candidates.append((length,(int(x1),int(y1),int(x2),int(y2)),angle))
    im=frame.copy()
    if candidates:
        length,line,angle=max(candidates,key=lambda a:a[0]);x1,y1,x2,y2=line
        rows.append(dict(frame=i,playback_s=pts[i],x1=x1+500,y1=y1+800,x2=x2+500,y2=y2+800,projected_angle_deg=angle,segment_length_px=length,status='unverified_partial_shaft_line'))
        cv2.line(im,(x1,y1),(x2,y2),(0,255,255),3)
    if i in [59,61,63,65,67,69,71,73]:
        im=cv2.resize(im,(160,360));cv2.putText(im,f'{pts[i]:.4f}s',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);tiles.append(im)
pd.DataFrame(rows).to_csv(out/f'sid_1_01_shaft_candidates{suffix}.csv',index=False)
cv2.imwrite(str(out/f'sid_1_01_shaft_candidates{suffix}.jpg'),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
print('Frames with shaft-line candidates:',len(rows),'of',len(frames))
