"""Abstaining phase proposals from visible wrists; no unverified impact labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def propose(p,peak):
 a=np.asarray(p['landmarks'],dtype=float);t=p['clip_start_playback_s']+np.arange(len(a))/p['fps']
 valid=(a[:,[15,16],3]>=.6).all(axis=1)
 search=(t>=peak-2)&(t<=peak-.4)&valid
 ids=np.flatnonzero(search)
 result={'top_candidate_s':None,'top_status':'abstain_insufficient_visible_wrists','impact_s':None,'impact_status':'unannotated','window_start_s':float(t[0]),'window_end_s':float(t[-1])}
 if len(ids)>=6:
  y=a[:,[15,16],1].mean(axis=1);i=ids[np.argmin(y[ids])]
  # A top must have visible observations on both sides, not just an edge extremum.
  if sum(ids<i)>=2 and sum(ids>i)>=2:
   result.update(top_candidate_s=float(t[i]),top_status='unvalidated_projected_hand_height_extremum')
 return result

if __name__=='__main__':
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');rows=[]
 for r in m.itertuples():
  p=json.loads((ROOT/f'data/processed/pose/{r.shot_id}.json').read_text())
  rows.append(dict(shot_id=r.shot_id,**propose(p,r.video_peak_candidate_s)))
 d=pd.DataFrame(rows);d.to_csv(ROOT/'annotations/phase_candidates.csv',index=False)
 print(d.top_status.value_counts().to_string())
