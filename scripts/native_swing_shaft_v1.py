"""Search every decoded frame of each swing for hand-connected shaft segments."""
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/native_swing_shaft'
SIZE=(270,480)


def run(row):
    cv2.setNumThreads(1)
    sid,source=row
    pose=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text())
    start=pose['clip_start_playback_s']
    cmd=['ffmpeg','-threads','2','-v','info','-ss',str(start),'-i',str(ROOT/source),'-t','3.5',
         '-an','-filter_threads','1','-vf','scale=270:480,showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-']
    proc=subprocess.run(cmd,capture_output=True,check=True)
    images=np.frombuffer(proc.stdout,np.uint8).reshape(-1,480,270)
    pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',proc.stderr.decode())])[:len(images)]
    assert len(pts)==len(images) and np.all(np.diff(pts)>0)
    bg=np.median(images[::24],axis=0).astype(np.uint8)
    lm=np.array(pose['landmarks'])
    hands=lm[:,[15,16],:2].mean(1)*SIZE
    ts=np.arange(len(hands))/pose['fps']
    grip=np.stack([np.interp(pts,ts,hands[:,i]) for i in range(2)],axis=1)
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE,.8)
    proposals=[]
    for i,(im,g) in enumerate(zip(images,grip)):
        lines=detector.detect(im)[0]
        if lines is None:continue
        lines=lines.reshape(-1,2,2)
        near=np.linalg.norm(lines-g,axis=2)
        reverse=near[:,0]>near[:,1]
        lines[reverse]=lines[reverse,::-1]
        a,b=lines[:,0],lines[:,1]
        length=np.linalg.norm(b-a,axis=1)
        v=(b-a)/length[:,None]
        radial=a-g
        perp=np.abs(radial[:,0]*v[:,1]-radial[:,1]*v[:,0])
        accepted=np.where((length>33)&(perp<9)&(near.min(1)<43)&((radial*v).sum(1)>-13))[0]
        options=[]
        for j in accepted:
            sample=np.rint(np.linspace(a[j],b[j],23)).astype(int)
            x,y=sample[:,0].clip(0,269),sample[:,1].clip(0,479)
            motion=np.mean(np.abs(im[y,x].astype(float)-bg[y,x])>12)
            if motion<.48:continue
            score=motion+min(length[j]/85,1)-perp[j]/14-near[j].min()/65
            options.append((score,j))
        if options:
            score,j=max(options)
            if score>.45:
                proposals.append(dict(frame=i,t=float(pts[i]),v=v[j].tolist(),a=a[j].tolist(),b=b[j].tolist()))
    kept=[]
    for i,p in enumerate(proposals):
        nearby=proposals[max(0,i-3):i]+proposals[i+1:i+4]
        if sum(abs(q['t']-p['t'])<.024 and np.dot(q['v'],p['v'])>.94 for q in nearby)>=2:
            kept.append(p)
    (OUT/f'{sid}.json').write_text(json.dumps(dict(start=start,frames=len(images),times=pts.tolist(),candidates=kept)))
    return sid,len(images),len(kept)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    manifest=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
    with ProcessPoolExecutor(4) as pool:
        for record in pool.map(run,manifest[['shot_id','video_candidate']].itertuples(index=False,name=None)):
            print(*record,flush=True)
