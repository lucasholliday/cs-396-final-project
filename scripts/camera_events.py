"""Camera-only mat-ball disappearance proposals; watch is used only for audit comparison."""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--session',default='Sid 1');args=parser.parse_args()
cfg=json.loads((ROOT/'configs/audit.json').read_text());video=cfg['sessions'][args.session][0]
out=ROOT/'outputs/camera_events';out.mkdir(parents=True,exist_ok=True)
stem=Path(video).stem;cache=out/f'{stem}_presence.csv';fps=20
if cache.exists():d=pd.read_csv(cache)
else:
    proc=subprocess.run(['ffmpeg','-v','error','-i',str(ROOT/video),'-an',
        '-vf',f'fps={fps},crop=220:260:590:1440,scale=110:130','-pix_fmt','gray','-f','rawvideo','-'],capture_output=True,check=True)
    frames=np.frombuffer(proc.stdout,dtype='uint8').reshape(-1,130,110)
    records=[]
    for i,f in enumerate(frames):
        _,_,stats,centers=cv2.connectedComponentsWithStats((f>145).astype('uint8'))
        candidates=[(c,s) for c,s in zip(centers[1:],stats[1:]) if 6<=s[4]<=160 and 3<=s[2]<=17 and 3<=s[3]<=17 and max(s[2],s[3])/min(s[2],s[3])<2]
        records.append(dict(playback_s=i/fps,ball_present=int(bool(candidates)),candidates=len(candidates)))
    d=pd.DataFrame(records);d.to_csv(cache,index=False)
presence=median_filter(d.ball_present.to_numpy(),size=5)
events=[];present_run=0;last=-100
for i,v in enumerate(presence):
    if v:present_run+=1
    else:
        if present_run>=fps*.7 and i+4<len(presence) and not presence[i:i+4].any() and i/fps-last>3:
            events.append(i/fps);last=i/fps
        present_run=0
pd.DataFrame(dict(camera_disappearance_s=events)).to_csv(out/f'{stem}_events.csv',index=False)
# This comparison does not alter candidates/thresholds or establish impact accuracy.
expected=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
expected=expected[expected.session==args.session].video_peak_candidate_s.to_numpy()
matches=[]
for t in events:
    ix=int(np.argmin(abs(expected-t)));error=float(t-expected[ix])
    matches.append(dict(camera_s=t,nearest_order=ix+1,difference_from_watch_metadata_s=error,within_1_5_s=abs(error)<=1.5))
pd.DataFrame(matches).to_csv(out/f'{stem}_comparison.csv',index=False)
summary=dict(session=args.session,method='20fps compact bright components on mat; sustained presence-to-absence transition',
    camera_candidates=len(events),reference_swings=len(expected),near_reference_candidates=sum(m['within_1_5_s'] for m in matches),
    limitation='Not exact impact ground truth; manual scene ROI/threshold, possible ball handling and occlusion false events')
(out/f'{stem}_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
