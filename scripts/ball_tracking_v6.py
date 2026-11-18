"""Wide-field positive-motion ball candidates and source-PTS evaluation.

Separates candidate recall from final-track accuracy. Approximate AI references
are an exploratory development set, not a blinded external test. No predictions
from this detector enter ML until trajectory review.
"""
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

def candidates(frame, background, cfg):
    positive=frame.astype(np.int16)-background.astype(np.int16)
    if cfg.get('dual_polarity',False):
        positive_cfg={**cfg,'dual_polarity':False}
        bright=candidates(frame,background,positive_cfg)
        # Inverted intensities identify dark balls silhouetted against bright sky.
        dark=candidates(255-frame,255-background,{**positive_cfg,'brightness_minimum':0})
        return [dict(c,polarity=1) for c in bright]+[dict(c,polarity=-1) for c in dark]
    mask=((positive>cfg['positive_difference']) & (frame>=cfg['brightness_minimum'])).astype(np.uint8)
    n,lab,stats,centers=cv2.connectedComponentsWithStats(mask)
    result=[]
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        if not(cfg['minimum_component_area']<=area<=cfg['maximum_component_area'] and w<=cfg['maximum_component_width'] and h<=cfg['maximum_component_height']):continue
        # Difference-weighted centroid reduces bias from dim blur tails.
        ys,xs=np.where(lab[y:y+h,x:x+w]==k);weights=positive[y+ys,x+xs].astype(float)
        result.append(dict(x=float(x+np.average(xs,weights=weights)),y=float(y+np.average(ys,weights=weights)),area=int(area),width=int(w),height=int(h),contrast=float(weights.mean())))
    return result

def seed_departures(frames,pts,background,cfg):
    ox,oy,_,_=cfg['native_crop'];x0,y0,x1,y1=cfg['seed_bounds_native'];mask=np.zeros_like(background,np.uint8)
    mask[y0-oy:y1-oy,x0-ox:x1-ox]=(background[y0-oy:y1-oy,x0-ox:x1-ox]>145)
    n,_,stats,centers=cv2.connectedComponentsWithStats(mask);result=[]
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        if not(20<=area<=700 and 4<=w<=40 and 4<=h<=40 and max(w,h)/min(w,h)<2.5):continue
        cx,cy=centers[k];xx,yy=int(cx),int(cy);patch=frames[:,yy-4:yy+5,xx-4:xx+5].mean(axis=(1,2));base=float(np.median(patch[:cfg['background_frames']]))
        # Persistent darkening is preferable to absolute change from passing shaft.
        dark=patch<base-25;eligible=np.flatnonzero(np.convolve(dark.astype(int),np.ones(8,dtype=int),'valid')>=7)
        eligible=eligible[eligible>=cfg['background_frames']]
        if len(eligible):
            i=int(eligible[0]);persistence=float(dark[i:].mean());drop=base-float(np.median(patch[-25:]));result.append(dict(x=float(cx+ox),y=float(cy+oy),frame=i,playback_s=float(pts[i]),persistence=persistence,drop=drop,score=float(persistence*max(0,drop))))
    return sorted(result,key=lambda s:-s['score'])

def process(row,cfg):
    dest=ROOT/f"data/processed/ball_v{cfg['version']}";dest.mkdir(parents=True,exist_ok=True);path=dest/f'{row.shot_id}.json'
    sig=hashlib.sha256(json.dumps(dict(cfg=cfg,source=row.video_candidate,mtime=(ROOT/row.video_candidate).stat().st_mtime_ns,peak=row.video_peak_candidate_s),sort_keys=True).encode()).hexdigest()
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']==sig:return old
    x,y,w,h=cfg['native_crop'];start=row.video_peak_candidate_s-cfg['before_peak_s']
    proc=subprocess.run(['ffmpeg','-v','info','-ss',str(start),'-i',str(ROOT/row.video_candidate),'-t',str(cfg['duration_s']),'-an','-vf',f'crop={w}:{h}:{x}:{y},showinfo','-vsync','0','-pix_fmt','gray','-f','rawvideo','-'],capture_output=True,check=True)
    frames=np.frombuffer(proc.stdout,np.uint8).reshape(-1,h,w);pts=np.array([float(t) for t in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.\-]+)',proc.stderr.decode())])+start
    # showinfo can log a terminal frame discarded by the output duration gate.
    assert len(pts)>=len(frames), (len(frames),len(pts))
    pts=pts[:len(frames)]
    assert np.all(np.diff(pts)>0)
    background=np.median(frames[:cfg['background_frames']],axis=0).astype(np.uint8)
    proposals=[]
    for i,frame in enumerate(frames):
        cs=candidates(frame,background,cfg)
        for c in cs:c['x']+=x;c['y']+=y
        proposals.append(dict(frame=i,playback_s=float(pts[i]),candidates=cs))
    seeds=seed_departures(frames,pts,background,cfg)
    result=dict(signature=sig,shot_id=row.shot_id,source=row.video_candidate,seeds=seeds,frames=proposals,configuration=cfg)
    path.write_text(json.dumps(result));return result

def evaluate(version=6):
    refs=pd.read_csv(ROOT/'annotations/contact_ball_points.csv');rows=[]
    for sid,g in refs.groupby('shot_id'):
        p=ROOT/f'data/processed/ball_v{version}/{sid}.json'
        if not p.exists():continue
        d=json.loads(p.read_text());old=json.loads((ROOT/f'data/processed/ball/{sid}.json').read_text());times=np.array([f['playback_s'] for f in d['frames']])
        for r in g.itertuples():
            i=int(np.argmin(abs(times-r.playback_s)));f=d['frames'][i];dt=abs(times[i]-r.playback_s);cs=f['candidates']
            error=min([np.hypot(c['x']-r.x,c['y']-r.y) for c in cs],default=float('inf')) if dt<=.006 else float('inf')
            obs=old['observations'];oe=float('inf');odt=float('inf')
            if obs:
                o=min(obs,key=lambda o:abs(o['playback_s']-r.playback_s));odt=abs(o['playback_s']-r.playback_s)
                if odt<=.006:oe=float(np.hypot(o['x']-r.x,o['y']-r.y))
            rows.append(dict(shot_id=sid,panel=r.panel,playback_s=r.playback_s,v6_time_error_s=dt,v6_candidate_count=len(cs),v6_nearest_error_px=error,v6_within_16px=error<=16,v6_within_32px=error<=32,v4_track_time_error_s=odt,v4_track_error_px=oe,v4_track_within_32px=oe<=32))
    out=ROOT/f'experiments/ball_tracking_v{version}';out.mkdir(exist_ok=True);df=pd.DataFrame(rows);df.rename(columns=lambda c:c.replace('v6_',f'v{version}_')).to_csv(out/'reference_errors.csv',index=False)
    summary=dict(reference_points=len(df),shots=int(df.shot_id.nunique()),v6_candidate_recall_16px=float(df.v6_within_16px.mean()),v6_candidate_recall_32px=float(df.v6_within_32px.mean()),v4_final_track_recall_32px=float(df.v4_track_within_32px.mean()),v6_median_candidate_count=float(df.v6_candidate_count.median()),warning='Candidate recall is an upper bound, not tracker accuracy. References are approximate AI review; second reviewer pending.')
    summary={k.replace('v6_',f'v{version}_'):v for k,v in summary.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=6);parser.add_argument('--all',action='store_true');parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/f'configs/ball_tracking_v{args.version}.json').read_text());m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
    if not args.evaluate_only:
        if not args.all:
            ids=json.loads((ROOT/'annotations/contact_ball_points_review.json').read_text())['points'];m=m[m.shot_id.isin(ids)]
        for r in m.itertuples():
            d=process(r,cfg);print(r.shot_id,'frames',len(d['frames']),'seed candidates',len(d['seeds']),flush=True)
    evaluate(args.version)
