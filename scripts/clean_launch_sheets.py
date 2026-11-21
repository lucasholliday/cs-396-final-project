"""Label-free native-review strips. Pixel coordinates stored in panel_map.csv."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/clean_launch_audit'
OUT.mkdir(exist_ok=True)
rows = []
for sid in ['lucas_1_04','andrew_1_04','lucas_1_06','andrew_1_09','eduardo_1_11','eduardo_1_08','eduardo_1_05','eduardo_1_16','andrew_1_10']:
    shot = json.loads((ROOT/f'data/processed/ball_launch_tracks/{sid}.json').read_text())
    meta = json.loads((ROOT/f'outputs/native_review/{sid}.json').read_text())
    seed = shot['seed']
    obs = {o['frame']:o for o in shot['observations']}
    path = ROOT/f'experiments/clean_launch_audit/{sid}_image.json'
    recovered = {o['frame']:o for o in json.loads(path.read_text())} if path.exists() else {}
    ts = np.array(meta['source_pts'])
    frames = np.flatnonzero((ts>=ts[seed['frame']]-.008)&(ts<=ts[seed['frame']]+.080))
    cap = cv2.VideoCapture(str(ROOT/f'outputs/native_review/{sid}.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
    panels = []
    for i in frames:
        ok,img = cap.read()
        assert ok
        x0 = int(np.clip(seed['x']-300-260,0,img.shape[1]-520))
        y0 = int(np.clip(seed['y']-650-740,0,img.shape[0]-800))
        tile = cv2.resize(img[y0:y0+800,x0:x0+520],(208,320))
        o = obs.get(i)
        if o:
            xy = (round(.4*(o['x']-300-x0)),round(.4*(o['y']-650-y0)))
            cv2.circle(tile,xy,8,(20,200,240),1)
        if i in recovered:
            r = recovered[i]
            xy = (round(.4*(r['x']-300-x0)),round(.4*(r['y']-650-y0)))
            cv2.circle(tile,xy,6,(230,230,20),1)
        cv2.putText(tile,str(i),(4,17),0,.55,(255,255,255),1)
        rows.append(dict(shot_id=sid,frame=i,x_origin=x0+300,y_origin=y0+650,scale=.4))
        panels.append(tile)
    cap.release()
    while len(panels)%6: panels.append(np.zeros_like(panels[0]))
    cv2.imwrite(str(OUT/f'{sid}.jpg'),np.vstack([np.hstack(panels[j:j+6]) for j in range(0,len(panels),6)]))
pd.DataFrame(rows).to_csv(OUT/'panel_map.csv',index=False)
