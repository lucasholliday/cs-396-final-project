"""Manually initialized single-shot background-difference ball track; exploratory."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/ball'
d=np.load(out/'sid_1_01_native_roi.npz');frames=d['frames'];pts=d['pts']
bg=cv2.cvtColor(np.median(frames[:25],axis=0).astype('uint8'),cv2.COLOR_BGR2GRAY)
# Initial visible flight location selected by inspecting native frames, not TrackMan labels.
position=np.array([706.,1495.]);velocity=np.array([100.,-13000.])
track=[];last_time=pts[64]
for i in range(64,len(frames)):
    gray=cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)
    mask=(cv2.absdiff(gray,bg)>30).astype('uint8')
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    dt=pts[i]-last_time
    predicted=position+velocity*dt
    for s,c in zip(stats[1:],centers[1:]):
        p=c+[500,800]
        if 5<=s[4]<=800 and 2<=s[2]<=45 and 2<=s[3]<=60 and max(s[2],s[3])/min(s[2],s[3])<3:
            if 665<p[0]<815 and p[1]<position[1]+5:
                distance=np.linalg.norm(p-predicted)
                if distance<max(35,abs(velocity[1])*max(dt,.004)*.75):
                    candidates.append((distance,p,s[4]))
    if not candidates: break
    distance,chosen,area=min(candidates,key=lambda c:c[0])
    if dt>0:velocity=.55*((chosen-position)/dt)+.45*velocity
    position=chosen;last_time=pts[i]
    track.append(dict(frame=i,playback_s=pts[i],x=chosen[0],y=chosen[1],area=int(area),prediction_residual_px=distance))
df=pd.DataFrame(track);df.to_csv(out/'sid_1_01_track.csv',index=False)
writer=cv2.VideoWriter(str(out/'sid_1_01_track_slow.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(400,900))
for i,frame in enumerate(frames):
    im=frame.copy();sub=df[df.frame<=i]
    points=(sub[['x','y']].to_numpy()-[500,800]).astype('int32')
    if len(points)>1:cv2.polylines(im,[points],False,(0,0,255),2)
    if len(points) and i in set(df.frame):cv2.circle(im,tuple(points[-1]),12,(0,255,255),2)
    cv2.putText(im,f'PTS {pts[i]:.4f}s | manual seed',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    writer.write(im)
writer.release()
summary=dict(shot='sid_1_01',frames_tracked=len(df),start_playback_s=float(df.playback_s.iloc[0]),
    end_playback_s=float(df.playback_s.iloc[-1]),native_pixel_displacement=float(np.linalg.norm(df[['x','y']].iloc[-1]-df[['x','y']].iloc[0])),
    status='single_shot_manually_seeded_exploratory_track_requires_localization_validation',
    limitations='No pixel-to-meter calibration, physical speed/launch angles/carry not estimated. Static-background assumption. Flight exits crop. Not validated across shots.',
    seed=dict(frame=64,display_x=706,display_y=1495))
(out/'track_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
# Final track over a frame where ball is already in flight, original portrait coordinates.
im=frames[70].copy();points=(df[['x','y']].to_numpy()-[500,800]).astype('int32')
cv2.polylines(im,[points],False,(0,0,255),2)
cv2.imwrite(str(out/'sid_1_01_track.jpg'),im)
