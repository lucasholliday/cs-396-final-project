"""Export illustrative monocular estimated skeleton beside source pose video."""
import argparse,json,subprocess
from pathlib import Path
import cv2
import numpy as np
from process_pose import EDGES
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--shot',default='sid_1_01');args=parser.parse_args()
 p=json.loads((ROOT/f'data/processed/pose/{args.shot}.json').read_text())
 out=ROOT/'outputs/pose_3d';out.mkdir(exist_ok=True)
 cap=cv2.VideoCapture(str(ROOT/f'outputs/pose/{args.shot}_overlay.mp4'))
 temp=out/f'{args.shot}_temporary.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),p['fps'],(1080,960))
 angle=np.deg2rad(25);contact=[]
 for i,points in enumerate(p['estimated_world_landmarks']):
  ok,left=cap.read()
  if not ok:raise RuntimeError('Pose video shorter than landmarks')
  left=cv2.resize(left,(540,960));right=np.full((960,540,3),(32,25,16),dtype=np.uint8)
  a=np.asarray(points,float);center=(a[23]+a[24])/2;a-=center
  xy=np.column_stack([270+370*(a[:,0]*np.cos(angle)+a[:,2]*np.sin(angle)),470+370*a[:,1]])
  lm=np.asarray(p['landmarks'][i],float)
  for u,v in EDGES:
   if not np.isfinite(xy[[u,v]]).all():continue
   color=(150,220,110) if min(lm[u,3],lm[v,3])>=.6 else (90,90,90)
   cv2.line(right,tuple(xy[u].astype(int)),tuple(xy[v].astype(int)),color,4)
  cv2.putText(right,'Estimated 3D skeleton',(20,45),cv2.FONT_HERSHEY_SIMPLEX,.75,(240,240,240),2)
  for j,line in enumerate(['Monocular model depth; unvalidated','Hip-centered, arbitrary display scale','Gray: low landmark confidence','Not calibrated motion capture']):
   cv2.putText(right,line,(20,820+j*30),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1)
  frame=np.hstack([left,right]);writer.write(frame)
  if i in [30,50,70]:contact.append(cv2.resize(frame,(540,480)))
 cap.release();writer.release()
 subprocess.run(['ffmpeg','-v','error','-i',str(temp),'-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(out/f'{args.shot}_estimated_3d.mp4')],check=True)
 temp.unlink()
 cv2.imwrite(str(out/f'{args.shot}_contact.jpg'),np.hstack(contact))
 print('Exported',args.shot,'estimated 3D illustration')
