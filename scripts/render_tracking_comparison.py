"""Reviewed reference points versus v7 linked candidates on native review frames."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=7);args=parser.parse_args()
 out=ROOT/f'outputs/ball_v{args.version}_review';out.mkdir(exist_ok=True);refs=pd.read_csv(ROOT/'annotations/contact_ball_points.csv');idx=pd.read_csv(ROOT/'outputs/contact_video_review/frame_index.csv')
 for sid,g in idx.groupby('shot_id'):
  path=ROOT/f'data/processed/ball_v{args.version}_tracks/{sid}.json'
  if not path.exists():continue
  d=json.loads(path.read_text());tiles=[]
  for r in g.itertuples():
   im=cv2.imread(str(ROOT/f'outputs/contact_video_review/{sid}_{r.panel}_native.png'))
   for ref in refs[(refs.shot_id==sid)&(refs.panel==r.panel)].itertuples():cv2.circle(im,(round(ref.x-300),round(ref.y-650)),16,(255,240,0),2)
   obs=d['observations']
   if obs:
    o=min(obs,key=lambda o:abs(o['playback_s']-r.playback_s))
    if abs(o['playback_s']-r.playback_s)<=.006:cv2.circle(im,(round(o['x']-300),round(o['y']-650)),22,(0,140,255),2)
   im=cv2.resize(im,(390,550));cv2.putText(im,f'{sid} panel {r.panel}',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,200,255),1);tiles.append(im)
  sheet=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]);cv2.imwrite(str(out/f'{sid}.jpg'),sheet)
 print('Saved candidate-vs-reference diagnostics; cyan=review, orange=linked tracker')
