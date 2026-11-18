"""Native-cadence H.264 ball-region clips with explicit source/output frame mapping."""
import json,subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 out=ROOT/'outputs/native_review';out.mkdir(exist_ok=True);m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');records=[]
 for r in m.itertuples():
  candidate=json.loads((ROOT/f'data/processed/ball_v7/{r.shot_id}.json').read_text());meta=out/f'{r.shot_id}.json';target=out/f'{r.shot_id}.mp4'
  if meta.exists() and target.exists():
   old=json.loads(meta.read_text())
   if old['candidate_signature']==candidate['signature'] and old.get('encoding_version')==2:records.append(dict(shot_id=r.shot_id,frames=len(old['source_pts']),status='verified_frame_count'));continue
  start=r.video_peak_candidate_s-1.25
  subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',str(ROOT/r.video_candidate),'-frames:v',str(len(candidate['frames'])),'-an','-vf','crop=780:1100:300:650','-vsync','0','-enc_time_base','-1','-video_track_timescale','60000','-c:v','libx264','-bf','0','-use_editlist','0','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(target)],capture_output=True,check=True)
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(target)]));review_pts=[float(f['best_effort_timestamp_time']) for f in info['frames']];source_pts=[f['playback_s'] for f in candidate['frames']]
  assert len(review_pts)==len(source_pts),(r.shot_id,len(review_pts),len(source_pts))
  import numpy as np
  cadence_error=float(np.max(abs(np.diff(review_pts)-np.diff(source_pts))));assert cadence_error<.0001,(r.shot_id,cadence_error)
  d=dict(encoding_version=2,shot_id=r.shot_id,source=r.video_candidate,candidate_signature=candidate['signature'],review_pts=review_pts,source_pts=source_pts,crop=[300,650,780,1100],cadence_error_max_s=cadence_error,semantics='Retained source cadence, H.264 re-encode; no inserted frames. Source/output correspondence verified by count and inter-frame PTS. Not original lossless pixels.')
  meta.write_text(json.dumps(d));records.append(dict(shot_id=r.shot_id,frames=len(review_pts),status='verified_frame_count'));print(r.shot_id,len(review_pts),flush=True)
 pd.DataFrame(records).to_csv(out/'verification.csv',index=False)
