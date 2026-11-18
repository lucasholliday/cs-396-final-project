"""Native-timestamp ball proposals using stationary seed, departure and temporal gating."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def components(mask):
    _,_,stats,centers=cv2.connectedComponentsWithStats(mask.astype('uint8'))
    return [(c,s) for c,s in zip(centers[1:],stats[1:]) if 5<=s[4]<=1200 and 2<=s[2]<=45 and 2<=s[3]<=180 and max(s[2],s[3])/min(s[2],s[3])<12]


def follow(initial_i,initial,stat,proposals,pts,seed,crop_xy,max_lateral=130):
    position=initial.copy();velocity=np.array([0.,-8000.]);last_t=pts[initial_i];misses=0
    x,y=crop_xy
    observations=[dict(frame=initial_i,playback_s=float(last_t),x=float(position[0]+x),y=float(position[1]+y),area=int(stat[4]),residual_px=0.)]
    for i in range(initial_i+1,len(pts)):
        dt=pts[i]-last_t;predicted=position+velocity*dt
        choices=[(np.linalg.norm(c-predicted),c,s) for c,s in proposals.get(i,[])
                 if abs(c[0]-seed[0])<max_lateral and c[1]<position[1]+4 and np.linalg.norm(c-predicted)<max(150 if len(observations)<4 else 35,abs(velocity[1])*dt*.8)]
        if not choices:
            misses+=1
            if misses>=3:break
            continue
        distance,new,stat=min(choices,key=lambda a:a[0]);misses=0
        velocity=.55*(new-position)/dt+.45*velocity;position=new;last_t=pts[i]
        observations.append(dict(frame=i,playback_s=float(pts[i]),x=float(position[0]+x),y=float(position[1]+y),area=int(stat[4]),residual_px=float(distance)))
    return observations


def track(row,cfg):
    out=ROOT/'outputs'/cfg.get('output_dir','ball_batch');out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'data/processed'/cfg.get('cache_dir','ball');cache.mkdir(parents=True,exist_ok=True)
    result_path=cache/f'{row.shot_id}.json'
    signature=hashlib.sha256(json.dumps(dict(config=cfg,path=row.video_candidate,peak=row.video_peak_candidate_s,mtime=(ROOT/row.video_candidate).stat().st_mtime_ns)).encode()).hexdigest()
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old.get('signature')==signature:return old['features']
    x,y,w,h=cfg['native_crop'];start=row.video_peak_candidate_s-cfg['before_peak_s']
    cmd=['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/row.video_candidate),'-t',str(cfg['duration_s']),'-an',
         '-vf',f'crop={w}:{h}:{x}:{y},showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-']
    proc=subprocess.run(cmd,capture_output=True,check=True)
    frames=np.frombuffer(proc.stdout,dtype=np.uint8).reshape(-1,h,w)
    pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',proc.stderr.decode())])[:len(frames)]+start
    if len(pts)!=len(frames):raise ValueError('PTS mismatch')
    background=np.median(frames[20:40],axis=0).astype('uint8')
    sx0,sy0,sx1,sy1=cfg['seed_region']
    white=np.zeros_like(background);white[sy0:sy1,sx0:sx1]=(background[sy0:sy1,sx0:sx1]>145)
    seeds=[(np.linalg.norm(c-np.array(cfg['seed_expected_xy'])),c,s) for c,s in components(white) if 25<=s[4]<=700 and 5<=s[2]<=40 and 5<=s[3]<=40 and max(s[2],s[3])/min(s[2],s[3])<2]
    observations=[];seed=None;departure=None;status='no_seed'
    if seeds:
        _,seed,stat=min(seeds,key=lambda a:a[0]);xx,yy=seed.astype(int)
        half=5;patch=background[yy-half:yy+half+1,xx-half:xx+half+1].astype(float)
        change=np.mean(np.abs(frames[:,yy-half:yy+half+1,xx-half:xx+half+1].astype(float)-patch),axis=(1,2))
        runs=np.convolve((change>35).astype(int),np.ones(5,dtype=int),mode='valid')
        eligible=np.where((runs==5)&(np.arange(len(runs))>=20))[0]
        status='no_departure'
        if len(eligible):
            departure=int(eligible[0])
            proposals={i:components(cv2.absdiff(frames[i],background)>cfg['difference_threshold']) for i in range(max(20,departure-3),len(frames))}
            tracks=[]
            for i in range(max(20,departure-2),min(departure+16,len(frames))):
                seeds_i=[(abs(c[0]-seed[0]),c,s) for c,s in proposals[i] if abs(c[0]-seed[0])<45 and 20<seed[1]-c[1]<250 and s[4]>=15]
                for _,initial,stat in sorted(seeds_i,key=lambda a:a[0])[:4]:
                    candidate=follow(i,initial,stat,proposals,pts,seed,(x,y),cfg.get("max_lateral_displacement",130))
                    if cfg.get("trim_stationary_tail",False):
                        from validate_ball_proposals import moving_prefix
                        candidate=moving_prefix(candidate)
                    if len(candidate)>=cfg['minimum_track_points'] and candidate[0]['y']-candidate[-1]['y']>150 and candidate[-1]['playback_s']-candidate[0]['playback_s']>=cfg.get('minimum_span_s',.1):
                        tracks.append(candidate)
            if tracks:
                observations=max(tracks,key=lambda a:(a[-1]['playback_s']-a[0]['playback_s'],-np.median([p['residual_px'] for p in a])))
            status='candidate_track' if len(observations)>=cfg['minimum_track_points'] else 'short_or_failed_track'
    features=dict(shot_id=row.shot_id,ball_track_available=int(status=='candidate_track'),ball_track_points=len(observations))
    if status=='candidate_track':
        a=pd.DataFrame(observations);dt=a.playback_s-a.playback_s.iloc[0]
        if dt.iloc[-1]<.04 or a.y.iloc[0]-a.y.iloc[-1]<40:status='short_or_failed_track';features['ball_track_available']=0
        else:
            first=a.iloc[:min(6,len(a))];t=first.playback_s-first.playback_s.iloc[0]
            vx,vy=np.polyfit(t,first.x,1)[0],np.polyfit(t,first.y,1)[0]
            features.update(ball_projected_vx=float(vx/1080),ball_projected_vy=float(vy/1920),
                ball_span_s=float(dt.iloc[-1]),ball_area_ratio=float(a.area.iloc[-1]/a.area.iloc[0]),
                ball_displacement_y=float((a.y.iloc[0]-a.y.iloc[-1])/1920),ball_median_residual_px=float(a.residual_px.median()))
    samples=[0]+[int(o['frame']) for o in observations[::max(1,len(observations)//5)]][:5]
    while len(samples)<6:samples.append(min(len(frames)-1,(departure or 20)+len(samples)*4))
    tiles=[]
    for i in samples[:6]:
        im=cv2.cvtColor(frames[i],cv2.COLOR_GRAY2BGR)
        if seed is not None:cv2.circle(im,tuple(seed.astype(int)),14,(0,255,0),2)
        points=np.array([[o['x']-x,o['y']-y] for o in observations if o['frame']<=i],dtype='int32')
        if len(points)>1:cv2.polylines(im,[points],False,(0,0,255),2)
        if len(points):cv2.circle(im,tuple(points[-1]),12,(0,255,255),2)
        im=cv2.resize(im,(176,416));cv2.putText(im,f'{pts[i]:.3f}s',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);tiles.append(im)
    cv2.imwrite(str(out/f'{row.shot_id}.jpg'),np.hstack(tiles))
    result_path.write_text(json.dumps(dict(signature=signature,shot_id=row.shot_id,status=status,
        seed_display_xy=(seed+[x,y]).tolist() if seed is not None else None,
        departure_candidate_s=float(pts[departure]) if departure is not None else None,
        observations=observations,features=features),indent=2))
    return features


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--representatives',action='store_true');parser.add_argument('--config',default='configs/ball_tracking.json');args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());df=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
    if args.representatives:df=df.groupby('session',sort=False).head(1)
    results=[]
    for row in df.itertuples():
        r=track(row,cfg);results.append(r);print(row.shot_id,r['ball_track_available'],r['ball_track_points'],flush=True)
    name='ball_features_representatives.csv' if args.representatives else 'ball_features.csv'
    pd.DataFrame(results).to_csv(ROOT/'data/processed'/cfg.get('feature_prefix','')/name,index=False)
