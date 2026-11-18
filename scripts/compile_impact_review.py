"""Bind manually reviewed frame bounds to source hashes without modifying master impacts."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 cfg=json.loads((ROOT/'annotations/impact_interval_review.json').read_text());idx=pd.read_csv(ROOT/'outputs/impact_review/frame_index.csv').set_index(['shot_id','panel']);inv=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path').sha256;m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').set_index('shot_id');rows=[]
 for sid,review in cfg['intervals'].items():
  lo=idx.loc[(sid,review['lower_panel'])];hi=idx.loc[(sid,review['upper_panel'])];assert lo.source==hi.source and lo.source_sha256==hi.source_sha256==inv[lo.source];assert lo.playback_s<hi.playback_s
  rows.append(dict(shot_id=sid,lower_s=lo.playback_s,upper_s=hi.playback_s,midpoint_s=(lo.playback_s+hi.playback_s)/2,width_ms=(hi.playback_s-lo.playback_s)*1000,source=lo.source,source_sha256=lo.source_sha256,review_status=cfg['metadata']['status'],note=review['note'],watch_peak_minus_departure_midpoint_s=m.loc[sid,'video_peak_candidate_s']-(lo.playback_s+hi.playback_s)/2,automatic_departure_s=lo.seed_proposal_s,automatic_inside_interval=bool(lo.playback_s<=lo.seed_proposal_s<=hi.playback_s)))
 d=pd.DataFrame(rows);d.to_csv(ROOT/'annotations/impact_intervals.csv',index=False);print(d[['shot_id','lower_s','upper_s','width_ms','watch_peak_minus_departure_midpoint_s','automatic_inside_interval']].to_string(index=False))
