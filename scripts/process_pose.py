"""Cached 2D pose and optical-motion baseline, with explicitly sensor-assisted clips."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]


def process(row, cfg):
    out=ROOT/cfg.get('pose_output_dir','outputs/pose')
    cache=ROOT/cfg.get('pose_cache_dir','data/processed/pose')
    out.mkdir(parents=True,exist_ok=True)
    cache.mkdir(parents=True,exist_ok=True)
    sid=row.shot_id
    record=cache/f'{sid}.json'
    start=max(0,float(row.video_peak_candidate_s)-cfg['clip_before_candidate_s'])
    duration=cfg['clip_before_candidate_s']+cfg['clip_after_candidate_s']
    signature=hashlib.sha256(json.dumps(dict(config=cfg,video=row.video_candidate,start=start,
        size=(ROOT/row.video_candidate).stat().st_size,mtime=(ROOT/row.video_candidate).stat().st_mtime_ns)).encode()).hexdigest()
    if record.exists():
        old=json.loads(record.read_text())
        if old.get('signature')==signature:
            return old['features']
    clip=out/f'{sid}_input.mp4'
    subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',str(ROOT/row.video_candidate),'-t',str(duration),
        '-an','-vf',f"fps={cfg['pose_fps']},scale={cfg['width']}:-2",'-c:v','libx264','-crf','18','-y',str(clip)],check=True,capture_output=True)
    cap=cv2.VideoCapture(str(clip))
    fps=cfg['pose_fps']
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(str(out/f'{sid}_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/cfg['pose_model'])),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    tracks=[]; world=[]; motion=[]; headpath=[]; index=0; prev=None; contacts=[]
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
        while True:
            ok,frame=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),round(index*1000/fps))
            landmarks=np.full((33,4),np.nan)
            world_lm=np.full((33,3),np.nan)
            if result.pose_landmarks:
                landmarks=np.array([[p.x,p.y,p.z,p.visibility] for p in result.pose_landmarks[0]])
                world_lm=np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]])
                for a,b in EDGES:
                    if min(landmarks[a,3],landmarks[b,3])>=cfg['min_visibility']:
                        cv2.line(frame,tuple((landmarks[a,:2]*[width,height]).astype(int)),tuple((landmarks[b,:2]*[width,height]).astype(int)),(70,230,70),2)
                head=landmarks[[0,7,8]]
                if np.all(head[:,3]>=cfg['min_visibility']):
                    headpath.append(tuple((head[:,:2].mean(0)*[width,height]).astype(int)))
                for a,b in zip(headpath,headpath[1:]): cv2.line(frame,a,b,(255,150,0),2)
            tracks.append(landmarks.tolist()); world.append(world_lm.tolist())
            # Optical motion computed on original image, foreground half excluding neighboring bay.
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            small=cv2.resize(gray,(135,240))
            if prev is not None:
                flow=cv2.calcOpticalFlowFarneback(prev,small,None,.5,3,15,3,5,1.2,0)
                magnitude=np.linalg.norm(flow[65:225,5:115],axis=2)
                motion.append([float(np.mean(magnitude)),float(np.quantile(magnitude,.95))])
            prev=small
            cv2.putText(frame,f'{sid}  estimated 2D pose  video {start+index/fps:.2f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            writer.write(frame)
            if index in [45,60,75]: contacts.append(cv2.resize(frame,(180,320)))
            index+=1
    cap.release();writer.release()
    arr=np.array(tracks); visible=arr[:,:,3]>=cfg['min_visibility']
    features={'shot_id':sid,'pose_detection_fraction':float(np.isfinite(arr[:,0,0]).mean()),
              'pose_visible_fraction':float(visible.mean())}
    # Pixel projection features, not metric 3D measurements.
    for name,ids in [('head',[0,7,8]),('hands',[15,16]),('shoulders',[11,12]),('hips',[23,24])]:
        valid=visible[:,ids].all(axis=1)
        xy=arr[:,ids,:2].mean(axis=1)
        good=xy[valid]
        features[name+'_valid_fraction']=float(valid.mean())
        for axis in range(2):
            features[f'{name}_range_{axis}']=float(np.ptp(good[:,axis])) if len(good)>1 else None
        diffs=np.linalg.norm(np.diff(xy,axis=0),axis=1)*fps
        diffs=diffs[valid[:-1]&valid[1:]]
        features[name+'_projected_speed_p95']=float(np.quantile(diffs,.95)) if len(diffs) else None
    if motion:
        m=np.array(motion)
        features.update(flow_mean=float(m[:,0].mean()),flow_p95=float(np.quantile(m[:,1],.95)),flow_max=float(m[:,1].max()))
    if len(contacts)==3:
        cv2.imwrite(str(out/f'{sid}_contact.jpg'),np.hstack(contacts))
    payload=dict(signature=signature,clip_start_playback_s=start,fps=fps,
                 semantics=cfg['description'],landmarks=tracks,estimated_world_landmarks=world,features=features)
    # JSON missing measurements are null, never nonstandard NaN.
    record.write_text(json.dumps(payload,default=float).replace('NaN','null'))
    return features


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--representatives',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/vision.json').read_text())
    df=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
    if args.representatives: df=df.groupby('session',sort=False).head(1)
    results=[]
    for row in df.itertuples():
        results.append(process(row,cfg));print(row.shot_id,'processed',flush=True)
    filename='pose_features_representatives.csv' if args.representatives else 'pose_features.csv'
    pd.DataFrame(results).to_csv(ROOT/'data/processed'/filename,index=False)
