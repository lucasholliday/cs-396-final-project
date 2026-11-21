"""Local image scale diagnostics from the stationary ball and body, not calibrated meters."""
import json
from pathlib import Path
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def diameter(gray,xy):
 x,y=np.rint(xy).astype(int);h,w=gray.shape
 if not(25<x<w-25 and 25<y<h-25):return np.nan
 patch=gray[y-24:y+25,x-24:x+25];ring=np.r_[patch[:6].ravel(),patch[-6:].ravel(),patch[:, :6].ravel(),patch[:,-6:].ravel()];threshold=max(120,float(np.median(ring)+35));mask=(patch>threshold).astype(np.uint8);n,_,stats,centers=cv2.connectedComponentsWithStats(mask);best=[]
 for i in range(1,n):
  _,_,ww,hh,area=stats[i];dist=np.linalg.norm(centers[i]-[24,24])
  if dist<=10 and 4<=ww<=35 and 4<=hh<=35 and 12<=area<=700 and max(ww,hh)/min(ww,hh)<2.3:best.append((dist,float(np.sqrt(4*area/np.pi))))
 return min(best)[1] if best else np.nan

if __name__=='__main__':
 rows=[]
 for path in sorted((ROOT/'data/processed/ball_launch_tracks').glob('*.json')):
  d=json.loads(path.read_text());sid=path.stem;seed=d['seed'];pose=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text());lm=np.array(pose['landmarks']);height=float(np.median((lm[:20,[27,28],1].mean(1)-lm[:20,[0,7,8],1].mean(1))*1920));c=cv2.VideoCapture(str(ROOT/f'outputs/native_review/{sid}.mp4'));samples=[];sf=seed['frame'];wanted=set(range(max(0,sf-24),max(1,sf-5),3));i=0
  while i<=max(wanted):
   ok,f=c.read()
   if not ok:break
   if i in wanted:samples.append(diameter(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),[seed['x']-300,seed['y']-650]))
   i+=1
  c.release();values=np.array(samples);values=values[np.isfinite(values)];ball=float(np.median(values)) if len(values)>=3 else np.nan
  rows.append(dict(shot_id=sid,body_height_px=height,stationary_ball_diameter_px=ball,ball_diameter_cv=float(np.std(values)/np.mean(values)) if len(values)>=3 else np.nan,ball_scale_support=len(values),seed_y=seed['y']/1920))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'data/processed/launch_scale_features.csv',index=False);print('diameter available',df.stationary_ball_diameter_px.notna().sum(),'/',len(df))
