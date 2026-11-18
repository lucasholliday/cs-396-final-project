"""Build a self-contained local review UI from persisted research outputs; no service/upload."""
import json
import subprocess
from pathlib import Path
import pandas as pd
from viewer_payload import encode_display_data
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/research_viewer';out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
labels=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').set_index('Shot ID')
pred=pd.read_csv(ROOT/'experiments/nested_carry_v1/predictions.csv')
pred=pred[(pred.cohort=='all_labeled')&(pred.scheme=='leave_golfer_out')&(pred.features=='camera_watch')].set_index('shot_id')
phase=pd.read_csv(ROOT/'annotations/phase_candidates.csv').set_index('shot_id')
inventory=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path')
gate=pd.read_csv(ROOT/'experiments/carry_thresholds_v1/predictions.csv')
gate=gate[(gate.features=='reviewed_ball_assisted') & (gate.threshold_yd==100)].set_index('shot_id')
assert len(gate)==31
intervals=pd.read_csv(ROOT/'annotations/impact_intervals.csv').set_index('shot_id')
reviewed_points=pd.read_csv(ROOT/'annotations/contact_ball_points.csv')
rows=[]
(out/'videos').mkdir(exist_ok=True)
for r in m.itertuples():
 source=ROOT/f'outputs/pose/{r.shot_id}_overlay.mp4'
 target=out/f'videos/{r.shot_id}_overlay.mp4'
 if not target.exists() or target.stat().st_mtime_ns<source.stat().st_mtime_ns:
  subprocess.run(['ffmpeg','-v','error','-i',str(source),'-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(target)],check=True,capture_output=True)
 p=json.loads((ROOT/f'data/processed/pose/{r.shot_id}.json').read_text())
 observations=json.loads((ROOT/f'data/processed/pose_observations/{r.shot_id}.json').read_text())
 native=json.loads((ROOT/f'outputs/native_review/{r.shot_id}.json').read_text())
 tracked=json.loads((ROOT/f'data/processed/ball_v8_tracks/{r.shot_id}.json').read_text())
 avatar=json.loads((ROOT/f'data/processed/avatar_motion/{r.shot_id}.json').read_text())
 club_path=ROOT/f'data/processed/club_display/{r.shot_id}.json'
 club_display=json.loads(club_path.read_text()) if club_path.exists() else None
 rows.append(dict(club_display=club_display,avatar_motion=avatar,native=native,ball_track=tracked['observations'],reviewed_points=reviewed_points[reviewed_points.shot_id==r.shot_id][['playback_s','x','y']].to_dict('records'),departure_interval=intervals.loc[r.shot_id,['lower_s','upper_s','width_ms']].to_dict() if r.shot_id in intervals.index else None,observations=observations,id=r.shot_id,golfer=r.golfer,session=r.session,start=p['clip_start_playback_s'],fps=p['fps'],world=p['estimated_world_landmarks'],landmarks=p['landmarks'],
  carry=labels.loc[r.shot_id,'Carry flat (yd)'],
  ball_speed=labels.loc[r.shot_id,'Ball speed (mph)'],launch=labels.loc[r.shot_id,'Launch angle (deg)'],
  club_available=bool(pd.notna(r.club_peak_candidate_s)),alignment_residual=r.club_watch_residual_s,
  workbook_row=int(r.workbook_row),
  gate_probability=float(gate.loc[r.shot_id,'low_probability']) if r.shot_id in gate.index else None,
  assisted_prediction=float(gate.loc[r.shot_id,'allshot_prediction']) if r.shot_id in gate.index else None,
  gate_error=bool((gate.loc[r.shot_id,'low_probability']>=.5)!=(gate.loc[r.shot_id,'actual']<100)) if r.shot_id in gate.index else False,
  prediction=float(pred.loc[r.shot_id,'predicted']) if r.shot_id in pred.index else None,
  top=phase.loc[r.shot_id,'top_candidate_s'],top_status=phase.loc[r.shot_id,'top_status'],source=r.video_candidate,sha256=inventory.loc[r.video_candidate,'sha256']))
payload=encode_display_data(rows)
template=(ROOT/'scripts/viewer_template.html').read_text().replace('__THREE__','/* '+(ROOT/'vendor/three/LICENSE').read_text()+' */\n'+(ROOT/'vendor/three/three.min.js').read_text()).replace('__SCENE__',(ROOT/'scripts/range_scene.js').read_text())
(out/'index.html').write_text(template.replace('__DATA__',payload))
print('Built local viewer for',len(rows),'shots:',out/'index.html')
