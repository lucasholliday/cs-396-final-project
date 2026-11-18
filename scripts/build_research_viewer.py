"""Build a self-contained local review UI from persisted research outputs; no service/upload."""
import json
import subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/research_viewer';out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
labels=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').set_index('Shot ID')
pred=pd.read_csv(ROOT/'experiments/nested_carry_v1/predictions.csv')
pred=pred[(pred.cohort=='all_labeled')&(pred.scheme=='leave_golfer_out')&(pred.features=='camera_watch')].set_index('shot_id')
phase=pd.read_csv(ROOT/'annotations/phase_candidates.csv').set_index('shot_id')
inventory=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path')
rows=[]
(out/'videos').mkdir(exist_ok=True)
for r in m.itertuples():
 source=ROOT/f'outputs/pose/{r.shot_id}_overlay.mp4'
 target=out/f'videos/{r.shot_id}_overlay.mp4'
 if not target.exists() or target.stat().st_mtime_ns<source.stat().st_mtime_ns:
  subprocess.run(['ffmpeg','-v','error','-i',str(source),'-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(target)],check=True,capture_output=True)
 p=json.loads((ROOT/f'data/processed/pose/{r.shot_id}.json').read_text())
 rows.append(dict(id=r.shot_id,golfer=r.golfer,session=r.session,start=p['clip_start_playback_s'],fps=p['fps'],world=p['estimated_world_landmarks'],landmarks=p['landmarks'],
  carry=labels.loc[r.shot_id,'Carry flat (yd)'],prediction=float(pred.loc[r.shot_id,'predicted']) if r.shot_id in pred.index else None,
  top=phase.loc[r.shot_id,'top_candidate_s'],top_status=phase.loc[r.shot_id,'top_status'],source=r.video_candidate,sha256=inventory.loc[r.video_candidate,'sha256']))
payload=json.dumps(rows,allow_nan=True).replace('NaN','null').replace('</','<\\/')
template=(ROOT/'scripts/viewer_template.html').read_text()
(out/'index.html').write_text(template.replace('__DATA__',payload))
print('Built local viewer for',len(rows),'shots:',out/'index.html')
