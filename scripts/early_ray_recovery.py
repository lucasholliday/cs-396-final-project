"""Recover sparse departures by consensus against native-frame bright blob candidates."""
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from launch_geometry_features import fit


def image_candidates(root,sid,seed):
    meta = json.loads((root/f'outputs/native_review/{sid}.json').read_text())
    origin_x,origin_y,_,_ = meta['crop']
    cap = cv2.VideoCapture(str(root/f'outputs/native_review/{sid}.mp4'))
    start = max(0,seed['frame']-10)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    history,frames = [],[]
    for i in range(start,len(meta['source_pts'])):
        t = meta['source_pts'][i]
        if t>seed['playback_s']+.08: break
        ok,img = cap.read()
        assert ok
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if i<seed['frame']-2:
            history.append(gray)
            continue
        if i<=seed['frame']: continue
        if not frames:
            bg = np.median(history,axis=0).astype('uint8')
        diff = cv2.subtract(gray,bg)
        _,_,stats,centers = cv2.connectedComponentsWithStats((diff>22).astype('uint8'))
        candidates = []
        for (x,y,w,h,area),(cx,cy) in zip(stats[1:],centers[1:]):
            if 10<=area<=1800 and w<=80 and h<=90:
                candidates.append(dict(x=cx+origin_x,y=cy+origin_y,area=area,width=w,height=h,polarity=1))
        frames.append(dict(frame=i,playback_s=t,candidates=candidates))
    cap.release()
    return frames

def recover(root,sid,seed,from_images=False):
    frames = image_candidates(root,sid,seed) if from_images else json.loads((root/f'data/processed/ball_v7/{sid}.json').read_text())['frames']
    origin = np.array([seed['x'],seed['y']])
    samples = []
    for frame in frames:
        dt = frame['playback_s']-seed['playback_s']
        if not 0 < dt <= .08:
            continue
        blobs = [c for c in frame['candidates'] if c['polarity']==1 and c['area']>=10
                 and c['y']<seed['y'] and max(c['width'],c['height'])<=4*min(c['width'],c['height'])]
        if blobs:
            xy = np.array([[c['x'],c['y']] for c in blobs])
            samples.append((dt,frame,xy,cKDTree(xy)))
    times = np.array([s[0] for s in samples])
    best_score, best_track = 0,[]
    for offset in [-.004,0,.004]:
        for rate in [0,10,25,50]:
            fractions = (times+offset)/(1+rate*(times+offset))
            for anchor,(dt,_,xy,_) in enumerate(samples):
                if not .012<=dt<=.055: continue
                v = (xy-origin)/fractions[anchor]
                v = v[(v[:,1]<-400)&(np.linalg.norm(v,axis=1)<30000)]
                if not len(v): continue
                hits = [s[3].query(origin+f*v) for s,f in zip(samples,fractions)]
                dist = np.stack([h[0] for h in hits])
                score = np.maximum(0,1-(dist/18)**2).sum(axis=0)
                j = score.argmax()
                if score[j]<=best_score: continue
                track = []
                for s,h in zip(samples,hits):
                    if h[0][j]<=18:
                        x,y = s[2][h[1][j]]
                        track.append(dict(frame=s[1]['frame'],playback_s=s[1]['playback_s'],x=x,y=y))
                if len(track)>=5 and track[-1]['playback_s']-track[0]['playback_s']>=.03:
                    if from_images and (track[0]['playback_s']-seed['playback_s']>.025 or
                            np.hypot(track[-1]['x']-track[0]['x'],track[-1]['y']-track[0]['y'])<80):
                        continue
                    best_score,best_track = score[j],track
    return fit(best_track,seed),best_track
