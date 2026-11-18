"""Display-only temporal completion of monocular pose. Never feeds model features.
User requested a stable, fully visible avatar even when a limb is occluded.
"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1]
BONES=[(23,24,.24),(23,11,.48),(24,12,.48),(11,13,.29),(13,15,.26),(12,14,.29),(14,16,.26),(23,25,.43),(25,27,.43),(24,26,.43),(26,28,.43),(27,29,.08),(27,31,.16),(28,30,.08),(28,32,.16)]

def complete(p):
 raw=np.asarray(p['estimated_world_landmarks'],float);lm=np.asarray(p['landmarks'],float);a=raw.copy();n=len(a);inferred=(lm[:,:,3]<.6)|~np.isfinite(a).all(axis=2)
 fallback=np.zeros((33,3));fallback[0]=[0,-.65,-.03];fallback[7]=[-.07,-.65,0];fallback[8]=[.07,-.65,0]
 for left,right,y,z,x in [(11,12,-.45,0,.2),(13,14,-.2,-.13,.25),(15,16,0,-.3,.04),(23,24,0,0,.12),(25,26,.42,-.07,.14),(27,28,.84,0,.16),(29,30,.85,.04,.16),(31,32,.85,-.15,.16)]:fallback[left]=[-x,y,z];fallback[right]=[x,y,z]
 hip=(a[:,23]+a[:,24])/2;hip[~np.isfinite(hip)]=0;a-=hip[:,None,:]
 for j in range(33):
  finite=np.isfinite(a[:,j]).all(axis=1);ids=np.flatnonzero(finite)
  if not len(ids):a[:,j]=fallback[j];continue
  filled=np.column_stack([np.interp(np.arange(n),ids,a[ids,j,k]) for k in range(3)])
  trusted=finite & (lm[:,j,3]>=.6)
  # Reject isolated large leg jumps before interpolation; no limb disappears.
  if j>=23:
   jumps=np.r_[0,np.linalg.norm(np.diff(filled,axis=0),axis=1)];trusted &= jumps<.14
  anchors=np.flatnonzero(trusted)
  if len(anchors)>=3:
   temporal=np.column_stack([np.interp(np.arange(n),anchors,filled[anchors,k]) for k in range(3)])
   weight=np.where(trusted,1.,.15 if j>=23 else .65)[:,None]
   filled=weight*filled+(1-weight)*temporal
  a[:,j]=gaussian_filter1d(filled,1.2 if j>=23 else .8,axis=0,mode='nearest')
  inferred[:,j]|=~trusted
 lengths={}
 for parent,child,default in BONES:
  dist=np.linalg.norm(a[:,child]-a[:,parent],axis=1);good=np.isfinite(dist)&(dist>default*.5)&(dist<default*1.6)
  length=float(np.median(dist[good])) if good.sum()>=5 else default;lengths[f'{parent}_{child}']=length
  for i in range(n):
   direction=a[i,child]-a[i,parent];norm=np.linalg.norm(direction)
   if norm<1e-5:direction=fallback[child]-fallback[parent];norm=np.linalg.norm(direction)
   if norm<1e-5:direction=np.array([0,1.,0]);norm=1
   a[i,child]=a[i,parent]+direction/norm*length
 # Keep hands together for the display rig's common golf grip. This is inferred.
 grip=(a[:,15]+a[:,16])/2;a[:,15]=grip+[-.018,0,0];a[:,16]=grip+[.018,0,0]
 # Two-bone arm completion keeps elbows plausible while both hands hold one grip.
 for i in range(n):
  shared=(a[i,15]+a[i,16])/2
  for _ in range(3):
   for shoulder,elbow,hand in [(11,13,15),(12,14,16)]:
    reach=lengths[f'{shoulder}_{elbow}']+lengths[f'{elbow}_{hand}'];delta=shared-a[i,shoulder];distance=np.linalg.norm(delta)
    if distance>reach*.98:shared=a[i,shoulder]+delta/distance*reach*.98
  for shoulder,elbow,hand,offset in [(11,13,15,-.012),(12,14,16,.012)]:
   target=shared+np.array([offset,0,0]);origin=a[i,shoulder];direction=target-origin;distance=max(1e-6,np.linalg.norm(direction));direction/=distance
   upper=lengths[f'{shoulder}_{elbow}'];lower=lengths[f'{elbow}_{hand}'];along=(upper**2-lower**2+distance**2)/(2*distance);height=np.sqrt(max(0,upper**2-along**2));bend=a[i,elbow]-origin;bend-=direction*np.dot(bend,direction)
   if np.linalg.norm(bend)<1e-5:bend=np.cross(direction,[0,0,1.])
   if np.linalg.norm(bend)<1e-5:bend=np.array([0.,1.,0.])
   a[i,elbow]=origin+direction*along+bend/np.linalg.norm(bend)*height;a[i,hand]=target
 return dict(frames=a.tolist(),inferred=inferred.tolist(),bone_lengths=lengths,semantics='Display-only smoothed/completed pose, with constant limb lengths and shared grip. Filled limbs and club orientation are illustrative estimates, excluded from all scientific features and predictions.')

if __name__=='__main__':
 out=ROOT/'data/processed/avatar_motion';out.mkdir(exist_ok=True)
 for path in sorted((ROOT/'data/processed/pose').glob('*.json')):
  p=json.loads(path.read_text());result=complete(p);result['pose_signature']=p['signature'];(out/path.name).write_text(json.dumps(result))
 print('Completed 76 display rigs; original poses and prediction features unchanged')
