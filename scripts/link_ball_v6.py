"""Frozen greedy kinematic linking of v6 candidates; never reads flight references."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def follow(frames,start,c,seed):
    first=frames[start];position=np.array([c['x'],c['y']]);origin=np.array([seed['x'],seed['y']]);dt=max(.012,first['playback_s']-seed['playback_s']+.006)
    velocity=(position-origin)/dt;last=first['playback_s'];obs=[dict(frame=start,playback_s=last,**c,residual_px=0.)]
    for f in frames[start+1:]:
        dt=f['playback_s']-last
        if dt>.035 or f['playback_s']-seed['playback_s']>.6:break
        expected=position+velocity*dt;speed=np.linalg.norm(velocity);choices=[]
        for q in f['candidates']:
            v=np.array([q['x'],q['y']])-position;distance=np.linalg.norm(v)
            if q['y']>position[1]+3 or distance/dt<180 or distance/dt>18000:continue
            cosine=np.dot(v,velocity)/max(distance*speed,1e-6)
            residual=np.linalg.norm(np.array([q['x'],q['y']])-expected)
            if cosine<.8 or residual>max(18,speed*dt*.7):continue
            # Do not jump from a small ball onto a much larger nearby limb.
            if q['area']>max(80,obs[-1]['area']*3):continue
            choices.append((residual/max(12,speed*dt)+.08*abs(np.log(q['area']/obs[-1]['area'])),q,residual))
        if not choices:continue
        _,q,residual=min(choices,key=lambda x:x[0]);new=np.array([q['x'],q['y']]);velocity=.6*(new-position)/dt+.4*velocity;position=new;last=f['playback_s'];obs.append(dict(frame=f['frame'],playback_s=last,**q,residual_px=float(residual)))
    return obs

def link(d,link_version=7):
    settings=json.loads((ROOT/f'configs/ball_linking_v{link_version}.json').read_text()) if link_version>=8 else None
    tracks=[]
    for seed in d['seeds'][:2]:
        if seed['score']<40:continue
        for f in d['frames']:
            elapsed=f['playback_s']-seed['playback_s']
            if not(-.008<=elapsed<=.075):continue
            for c in f['candidates']:
                if link_version>=8 and not(settings['initial_elapsed_min_s']<=elapsed<=settings['initial_elapsed_max_s'] and c.get('polarity')==settings['initial_polarity'] and c['area']>=settings['initial_min_area']):continue
                dx=c['x']-seed['x'];dy=seed['y']-c['y']
                if not(15<dy<450 and abs(dx)<350 and abs(dx)<dy*3):continue
                track=follow(d['frames'],f['frame'],c,seed)
                span=track[-1]['playback_s']-track[0]['playback_s'];rise=track[0]['y']-track[-1]['y']
                if len(track)>=6 and span>=.035 and rise>=60:
                    score=len(track)+span*50-np.median([o['residual_px'] for o in track])*.15
                    tracks.append((score,track,seed))
    if not tracks:return dict(shot_id=d['shot_id'],status='abstain',observations=[],seed=None)
    score,track,seed=max(tracks,key=lambda q:q[0]);return dict(shot_id=d['shot_id'],status='unvalidated_candidate',observations=track,seed=seed,score=float(score))

def evaluate(results,version=6):
    refs=pd.read_csv(ROOT/'annotations/contact_ball_points.csv');rows=[]
    for r in refs.itertuples():
        if r.shot_id not in results:continue
        d=results[r.shot_id];obs=d['observations'];dt=error=float('inf')
        if obs:
            o=min(obs,key=lambda o:abs(o['playback_s']-r.playback_s));dt=abs(o['playback_s']-r.playback_s)
            if dt<=.006:error=float(np.hypot(o['x']-r.x,o['y']-r.y))
        rows.append(dict(shot_id=r.shot_id,panel=r.panel,time_error_s=dt,error_px=error,within_16px=error<=16,within_32px=error<=32))
    out=ROOT/f'experiments/ball_tracking_v{version}';out.mkdir(exist_ok=True);df=pd.DataFrame(rows);df.to_csv(out/'linked_reference_errors.csv',index=False)
    summary=dict(reference_points=len(df),shots=int(df.shot_id.nunique()),candidate_tracks=sum(bool(d['observations']) for d in results.values()),evaluated_windows=len(results),reference_recall_16px=float(df.within_16px.mean()),reference_recall_32px=float(df.within_32px.mean()),warning='Sparse reference recall is not full-track precision. Additional tracked points remain unreviewed.')
    (out/'linked_summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=6);parser.add_argument('--link-version',type=int);parser.add_argument('--reviewed-only',action='store_true');args=parser.parse_args();link_version=args.link_version or args.version
    reviewed=json.loads((ROOT/'annotations/contact_ball_points_review.json').read_text())['points']
    out=ROOT/f'data/processed/ball_v{link_version}_tracks';out.mkdir(exist_ok=True);results={}
    for path in sorted((ROOT/f'data/processed/ball_v{args.version}').glob('*.json')):
        if args.reviewed_only and path.stem not in reviewed:continue
        d=json.loads(path.read_text());saved=out/path.name
        old=json.loads(saved.read_text()) if saved.exists() else {}
        r=old if old.get('candidate_signature')==d['signature'] and old.get('link_version')==link_version else link(d,link_version)
        r['link_version']=link_version;r['candidate_signature']=d['signature'];(out/path.name).write_text(json.dumps(r));results[path.stem]=r;print(path.stem,len(r['observations']),flush=True)
    evaluate(results,link_version)
