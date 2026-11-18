"""Sparse visually reviewed flight features for a separate human-assisted feasibility cohort.
No automatic unreviewed trajectory enters this table; no metric speed interpretation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 cfg=json.loads((ROOT/'annotations/contact_ball_points_review.json').read_text());idx=pd.read_csv(ROOT/'outputs/contact_video_review/frame_index.csv').set_index(['shot_id','panel'])
 inv=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path').sha256
 features=[];points=[];quality=[]
 for sid,entries in cfg['points'].items():
  old=json.loads((ROOT/f'data/processed/ball/{sid}.json').read_text());seed=cfg['seed_overrides_native'].get(sid,old['seed_display_xy'])
  track=[]
  for panel,x,y in entries:
   r=idx.loc[(sid,panel)];assert r.source_sha256==inv[r.source]
   p=dict(shot_id=sid,panel=panel,playback_s=r.playback_s,x=300+2*x,y=650+2*y,source=r.source,source_sha256=r.source_sha256,uncertainty_px=cfg['metadata']['uncertainty_native_px'],reviewer='AI_visual_unmarked_frames_second_review_pending')
   points.append(p);track.append(p)
  quality.append(dict(shot_id=sid,reviewed_points=len(track),notes=cfg['review_notes'].get(sid,'')))
  if not track or seed is None:continue
  track=sorted(track,key=lambda p:p['playback_s']);first=track[0];departure=idx.loc[(sid,first['panel'])].departure_unverified_s
  elapsed=first['playback_s']-departure
  if elapsed<=0:raise ValueError('First flight point must follow departure candidate')
  # All dimensions are image projections. Exact elapsed since impact is uncertain.
  f=dict(shot_id=sid,reviewed_ball_vertical_rate=(seed[1]-first['y'])/(1920*elapsed),reviewed_ball_lateral_rate=(first['x']-seed[0])/(1080*elapsed),
   reviewed_ball_first_elapsed_s=elapsed)
  features.append(f)
 out=ROOT/'experiments/reviewed_ball_v1';out.mkdir(exist_ok=True)
 pd.DataFrame(features).to_csv(ROOT/'data/processed/reviewed_ball_features.csv',index=False)
 pd.DataFrame(points).to_csv(ROOT/'annotations/contact_ball_points.csv',index=False)
 pd.DataFrame(quality).to_csv(out/'review_coverage.csv',index=False)
 print('Reviewed',len(points),'points on',len(features),'of',len(quality),'selected shots')
