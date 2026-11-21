"""Display-only shaft reconstruction on a 120 Hz timeline in range coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
SAMPLES = ROOT / 'outputs/club_motion'
N = 512
idx = np.arange(N) + .5
z = 1 - 2*idx/N
az = idx*np.pi*(3-np.sqrt(5))
sphere = np.column_stack((np.sqrt(1-z*z)*np.cos(az),np.sqrt(1-z*z)*np.sin(az),z))
neighbors = cKDTree(sphere).query(sphere,k=9)[1]
turn_cost = 9*(1-(sphere[:,None,:]*sphere[neighbors]).sum(-1))


def unit(v):
    return v / np.maximum(1e-9,np.linalg.norm(v,axis=-1,keepdims=True))


def unpack(frame):
    return [np.array(x) for x in frame]


def clearance(frame, directions):
    hand,_,center,quat,scale,head,hips = unpack(frame)
    rotation = Rotation.from_quat(quat).as_matrix()
    minimum = np.full(len(directions),np.inf)
    for origin,radii in [(center,scale*[.92,.51,.95]),(head,[.081,.105,.093]),(hips,[.15,.105,.105])]:
        start = ((hand-origin)@rotation)/radii
        ray = (directions@rotation)/radii
        t = np.clip(-(ray*start).sum(-1)/(ray*ray).sum(-1),.045,.867)
        minimum = np.minimum(minimum,np.linalg.norm(start+t[:,None]*ray,axis=-1))
    return minimum


def native_evidence(sid, record, desired):
    """Use corroborated image directions while retaining the estimated depth."""
    times = np.arange(len(desired))/record['fps']+record['start']
    cues = []
    if sid=='sid_1_01':
        native=json.loads((ROOT/'outputs/native_club/sid_1_01.json').read_text())
        marks=json.loads((DATA/'club_display/sid_1_01.json').read_text())['frames']
        valid=[i for i,m in enumerate(marks) if m]
        grip=np.array([marks[i]['grip'] for i in valid])
        mapping=native['review_frame_map']
        src=[f['playback_s'] for f in mapping]
        for o in native['observations']:
            t=o['playback_s']
            g=[np.interp(t,np.asarray(src)[valid],grip[:,k])*2 for k in range(2)]
            vector=np.array([o['x']-g[0],g[1]-o['y']])
            if np.linalg.norm(vector)>55:
                cues.append((record['start']+np.interp(t,src,np.arange(len(src)))/30,unit(vector),.85))
    else:
        path=DATA/f'ball_club_v2/{sid}.json'
        if path.exists():
            for f in json.loads(path.read_text())['club_frames']:
                i=int(np.clip(round((f['playback_s']-record['start'])*record['fps']),0,len(times)-1))
                for l in f['candidates']:
                    v=unit((np.array(l['b'])-l['a'])*[1,-1])
                    if (l['hand_line_distance_px']<28 and l['length_px']>115 and
                        l['motion_fraction']>.65 and v@unit(desired[i,:2])>.83):
                        cues.append((f['playback_s'],v,.45))
                        break
    used=0
    for j,(t,v,w) in enumerate(cues):
        # Isolated shaft segments are commonly clothing or range features.
        if not any(0<abs(t-t2)<.024 and v@v2>.93 for t2,v2,_ in cues[max(0,j-4):j+5]):
            continue
        mask=np.abs(times-t)<.009
        if not mask.any():continue
        target=desired[mask].copy()
        target[:,:2]=v*np.linalg.norm(target[:,:2],axis=1)[:,None]
        desired[mask]=unit(desired[mask]*(1-w)+target*w)
        used+=1
    return used


def solve(record,sid):
    frames=record['frames']
    desired=unit(np.array([f[1] for f in frames]))
    used=native_evidence(sid,record,desired)
    desired=unit(gaussian_filter1d(desired,1.6,axis=0,mode='nearest'))
    parents=np.zeros((len(frames),N),dtype=np.int16)
    cost=np.zeros(N)
    for i,frame in enumerate(frames):
        distance=clearance(frame,sphere)
        transition=cost[neighbors]+turn_cost
        best=transition.argmin(1)
        parents[i]=neighbors[np.arange(N),best]
        cost=transition[np.arange(N),best]+(1-sphere@desired[i])+180*np.maximum(0,1.09-distance)**2
        cost-=cost.min()
    path=np.empty(len(frames),dtype=int)
    path[-1]=np.argmin(cost)
    for i in range(len(frames)-1,0,-1):
        path[i-1]=parents[i,path[i]]
    smooth=unit(gaussian_filter1d(sphere[path],2.1,axis=0,mode='nearest'))
    return smooth,used


def grip_offsets(frames, directions):
    """Move an embedded grip outward slightly, with temporal easing."""
    offsets=np.zeros((len(frames),3))
    for i,(frame,d) in enumerate(zip(frames,directions)):
        hand,_,center,quat,scale,head,hips=unpack(frame)
        rot=Rotation.from_quat(quat).as_matrix()
        for origin,radii in [(hips,np.array([.15,.105,.105])),(center,scale*[.92,.51,.95]),(head,np.array([.081,.105,.093]))]:
            ray=(d@rot)/radii
            start=((hand+offsets[i]-origin)@rot)/radii
            along=np.clip(-start@ray/(ray@ray),.045,.867)
            closest=start+along*ray
            distance=np.linalg.norm(closest)
            if distance<1.10:
                normal=(closest/radii)@rot.T
                offsets[i]+=unit(normal)*min(.08,(1.10-distance)*distance/max(1e-8,np.linalg.norm(normal)))
    eased=gaussian_filter1d(offsets,1.2,axis=0,mode='nearest')
    return eased*np.minimum(1,.08/np.maximum(1e-9,np.linalg.norm(eased,axis=1)))[:,None]


def metrics(frames,d):
    angles=np.degrees(np.arccos(np.clip((d[1:]*d[:-1]).sum(1),-1,1)))
    gaps=np.array([clearance(f,u[None])[0] for f,u in zip(frames,d)])
    return dict(max_step_deg=angles.max(),steps_over_15=int((angles>15).sum()),
                body_intersections=int((gaps<1).sum()),samples=len(frames))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    out=DATA/'range_club_motion';out.mkdir(exist_ok=True)
    stats=[]
    for p in sorted((SAMPLES/('after' if args.check else 'before')).glob('*.json')):
        record=json.loads(p.read_text())
        raw=unit(np.array([f[1] for f in record['frames']]))
        if args.check:
            stats.append(dict(id=p.stem,stage='rendered',**metrics(record['frames'],raw)))
            continue
        smooth,used=solve(record,p.stem.removesuffix('_alternate'))
        (out/p.name).write_text(json.dumps(dict(fps=record['fps'],direction=smooth.round(7).tolist(),
                                               hand_offset=grip_offsets(record["frames"],smooth).round(7).tolist(),native_cues=used),separators=(',',':')))
        for stage,d in [('before',raw),('solved',smooth)]:
            stats.append(dict(id=p.stem,stage=stage,native_cues=used,**metrics(record['frames'],d)))
    df=pd.DataFrame(stats)
    df.to_csv(SAMPLES/('verified.csv' if args.check else 'comparison.csv'),index=False)
    print(df[~df.id.str.endswith('_alternate')].groupby('stage').agg(
        samples=('samples','sum'),max_step=('max_step_deg','max'),jumps=('steps_over_15','sum'),
        intersections=('body_intersections','sum')))
