"""Pose-assisted visible shaft evidence, independent of the previous phase animation."""
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def cross2(a,b):return a[0]*b[1]-a[1]*b[0]

def detect(gray,background,hand,cfg,ball=None,previous=None,body=None):
    edges=cv2.Canny(gray,35,110);lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=28,minLineLength=cfg['minimum_segment_px'],maxLineGap=12)
    if lines is None:return []
    diff=cv2.absdiff(gray,background);cs=[]
    static_edges=cv2.dilate(cv2.Canny(background,35,110),np.ones((3,3),np.uint8))
    for line in np.asarray(lines).reshape(-1,4):
        a=line[:2].astype(float);b=line[2:].astype(float)
        if np.linalg.norm(a-hand)>np.linalg.norm(b-hand):a,b=b,a
        direction=b-a;length=np.linalg.norm(direction)
        if length<1:continue
        direction/=length;perp=abs(cross2(direction,hand-a));near=np.linalg.norm(a-hand);far=np.linalg.norm(b-hand)
        if perp>cfg['maximum_hand_line_distance_px'] or near>cfg['maximum_near_endpoint_px'] or not 65<far<360:continue
        along=np.dot(hand-a,direction)
        if along>length-35:continue
        xy=np.linspace(a,b,40).astype(int);xy[:,0]=np.clip(xy[:,0],0,gray.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,gray.shape[0]-1)
        static_fraction=float(np.mean(static_edges[xy[:,1],xy[:,0]]>0))
        if static_fraction>.55:continue
        if body is not None and np.mean(body[xy[:,1],xy[:,0]]>0)>.4:continue
        normal=np.array([-direction[1],direction[0]])*4
        side_a=np.rint(xy+normal).astype(int);side_b=np.rint(xy-normal).astype(int)
        for side in [side_a,side_b]:
            side[:,0]=np.clip(side[:,0],0,gray.shape[1]-1);side[:,1]=np.clip(side[:,1],0,gray.shape[0]-1)
        center=gray[xy[:,1],xy[:,0]].astype(float);da=center-gray[side_a[:,1],side_a[:,0]];db=center-gray[side_b[:,1],side_b[:,0]]
        ridge=float(np.mean((da*db>0)&(np.minimum(abs(da),abs(db))>9)))
        if ridge<.3:continue
        motion=float(np.mean(diff[xy[:,1],xy[:,0]]>14));ball_dist=abs(cross2(direction,ball-hand)) if ball is not None else None
        address=ball is not None and ball_dist<20 and np.dot(ball-hand,direction)>0
        if motion<cfg['minimum_motion_fraction'] and not address:continue
        score=length/180+motion+ridge-perp/25-near/180
        if ball is not None:score+=max(-.5,1-ball_dist/45) if address else -.3
        if previous is not None:score+=.2*np.dot(direction,previous)
        cs.append(dict(segment_a=a.tolist(),segment_b=b.tolist(),direction=direction.tolist(),hand_line_distance_px=float(perp),length_px=float(length),motion_fraction=motion,ridge_fraction=ridge,score=float(score),hand=hand.tolist(),status='unreviewed_shaft_candidate'))
    return sorted(cs,key=lambda c:-c['score'])[:6]

def process(row,cfg):
    out=ROOT/'outputs/shaft_tracking';out.mkdir(exist_ok=True);cache=ROOT/'data/processed/shaft_tracking';cache.mkdir(exist_ok=True);p=json.loads((ROOT/f'data/processed/pose/{row.shot_id}.json').read_text());signature=hashlib.sha256(json.dumps(dict(pose=p['signature'],cfg=cfg),sort_keys=True).encode()).hexdigest();target=cache/f'{row.shot_id}.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['signature']==signature:return old
    cap=cv2.VideoCapture(str(ROOT/f'outputs/pose/{row.shot_id}_input.mp4'));frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(frame)
    cap.release();grays=np.array([cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]);background=np.median(grays[::3],axis=0).astype(np.uint8);a=np.array(p['landmarks'],float);world=np.array(p['estimated_world_landmarks'],float);h,w=grays.shape[1:];seedpath=ROOT/f'data/processed/ball_v7/{row.shot_id}.json';ball=None
    if seedpath.exists():
        seeds=json.loads(seedpath.read_text())['seeds'];ball=np.array([seeds[0]['x']/2,seeds[0]['y']/2]) if seeds else None
    records=[];previous=None
    for i,gray in enumerate(grays):
        body=np.zeros_like(gray)
        ids=[0,7,8,11,12,13,14,15,16,23,24,25,26]
        coords=np.rint(a[i,ids,:2]*[w,h]).astype(np.int32)
        cv2.fillConvexPoly(body,cv2.convexHull(coords),255)
        hand=np.mean(a[i,[15,16],:2],axis=0)*[w,h];cs=detect(gray,background,hand,cfg,ball if i<12 else None,previous,body) if np.isfinite(hand).all() else []
        chosen=cs[0] if cs and cs[0]['score']>=cfg['minimum_score'] else None
        if chosen:previous=np.array(chosen['direction'])
        else:previous=None
        # Fit an image projection to the model's visible torso/leg landmarks.
        # Minimum-norm inversion locates the shaft in the image plane; depth unknown.
        display_direction=None;fit_error=None
        if chosen:
            ids=[j for j in [11,12,23,24,25,26,27,28] if a[i,j,3]>=.6 and np.isfinite(world[i,j]).all()]
            if len(ids)>=5:
                q=world[i,ids];xy=a[i,ids,:2]*[w,h];design=np.c_[q,np.ones(len(q))];coeff=np.linalg.lstsq(design,xy,rcond=None)[0];fit_error=float(np.mean(np.linalg.norm(design@coeff-xy,axis=1)));delta=np.array(chosen['segment_b'])-hand;v=np.linalg.lstsq(coeff[:3].T,delta,rcond=None)[0]
                if np.linalg.norm(v)>1e-6:display_direction=(v/np.linalg.norm(v)).tolist()
        records.append(dict(frame=i,playback_s=p['clip_start_playback_s']+i/p['fps'],selected=chosen,candidates=cs,display_direction=display_direction,projection_fit_error_px=fit_error))
    result=dict(shot_id=row.shot_id,signature=signature,frames=records,semantics='Unreviewed visible-shaft line candidates. Display direction is an image-plane lift using estimated pose projection; club depth and clubhead location are not recovered.')
    target.write_text(json.dumps(result));tiles=[]
    for i in [0,15,30,45,52,59,66,80,100]:
        im=frames[i].copy();r=records[i]
        if r['selected']:
            c=r['selected'];cv2.line(im,tuple(np.array(c['segment_a']).astype(int)),tuple(np.array(c['segment_b']).astype(int)),(0,220,255),3);cv2.circle(im,tuple(np.array(c['hand']).astype(int)),8,(255,150,0),2)
        im=cv2.resize(im,(270,480));cv2.putText(im,f'{row.shot_id} f{i}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,230,255),1);tiles.append(im)
    cv2.imwrite(str(out/f'{row.shot_id}_sheet.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in [0,3,6]]));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args();m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');cfg=json.loads((ROOT/'configs/shaft_tracking.json').read_text())
    if not args.all:m=m.groupby('session',sort=False).head(1)
    for row in m.itertuples():
        d=process(row,cfg);print(row.shot_id,sum(r['selected'] is not None for r in d['frames']),len(d['frames']),flush=True)
