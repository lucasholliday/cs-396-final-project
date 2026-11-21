"""Build a self-contained local review UI from persisted research outputs; no service/upload."""
import json
import subprocess
from pathlib import Path
import pandas as pd
from rollout_summary import score_cohorts
from viewer_payload import encode_display_data
from ball_continuation import forecast
from ball_motion_gate import truncate_stationary_tail
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/research_viewer';out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
labels=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').set_index('Shot ID')
endpoint_predictions=pd.read_csv(ROOT/'experiments/label_stack_v1/predictions.csv').query("method == 'launch_roll'").set_index('shot_id',verify_integrity=True)
pred=pd.read_csv(ROOT/'experiments/nested_carry_v1/predictions.csv')
pred=pred[(pred.cohort=='all_labeled')&(pred.scheme=='leave_golfer_out')&(pred.features=='camera_watch')].set_index('shot_id')
vl_path=ROOT/'experiments/video_launch_v1/predictions.csv'
at_path=ROOT/'experiments/all_targets_v1/predictions.csv'

geom_path = ROOT / 'data/processed/launch_geometry_features.csv'
geom_df = pd.read_csv(geom_path).set_index('shot_id') if geom_path.exists() else pd.DataFrame()

# Audited Models for All Other TrackMan Targets
at_preds={}
if at_path.exists():
 at_df=pd.read_csv(at_path)
 def _get_at(t, f, m):
  sub=at_df[(at_df.target==t)&(at_df.features==f)&(at_df.method==m)]
  return sub.set_index('shot_id')['predicted'].to_dict()
 at_preds['dir']=_get_at('launch_dir_deg','video_all','nested')
 at_preds['curve']=_get_at('curve_ft','video_all','svr1')

if vl_path.exists():
 vl_df=pd.read_csv(vl_path)
 vl_dir=vl_df[(vl_df.target=='Launch direction (deg; R+)')&(vl_df.model=='ridge10')].set_index('shot_id')['predicted'].to_dict()
else:
 vl_dir={}

cf_path = ROOT / 'experiments/club_fusion_v1/predictions.csv'
if cf_path.exists():
 cf_df = pd.read_csv(cf_path)
 cf_curve = cf_df[(cf_df.target == 'Curve signed (ft; R+)') & (cf_df.model_name == 'Visual-Inertial Fusion')].set_index('shot_id')['predicted'].to_dict()
else:
 cf_curve = {}

phase=pd.read_csv(ROOT/'annotations/phase_candidates.csv').set_index('shot_id')
inventory=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path')
gate=pd.read_csv(ROOT/'experiments/carry_thresholds_v1/predictions.csv')
gate=gate[(gate.features=='reviewed_ball_assisted') & (gate.threshold_yd==100)].set_index('shot_id')
assert len(gate)==31
intervals=pd.read_csv(ROOT/'annotations/impact_intervals.csv').set_index('shot_id')
reviewed_points=pd.read_csv(ROOT/'annotations/contact_ball_points.csv')
wide_review_path = ROOT / 'configs/ball_extension_viewer.json'
wide_reviews = json.loads(wide_review_path.read_text()) if wide_review_path.exists() else {}
rejection_path = ROOT / 'configs/ball_track_review_rejections.json'
ball_rejections = json.loads(rejection_path.read_text()) if rejection_path.exists() else {}
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
 separated_path=ROOT/f'data/processed/ball_club_v2/{r.shot_id}.json'
 separated=json.loads(separated_path.read_text()) if separated_path.exists() else None
 head_path=ROOT/f'outputs/native_club/{r.shot_id}.json'
 head_data=json.loads(head_path.read_text()) if head_path.exists() else {}
 head_track=head_data.get('observations',[])
 clean_track_path = ROOT / f'data/processed/ball_launch_tracks/{r.shot_id}.json'
 clean_data = json.loads(clean_track_path.read_text()) if clean_track_path.exists() else None
 if clean_data:
  tracked = {**tracked, 'observations': clean_data['observations']}
  separated_status = 'physics_gated_ball_launch'
 elif separated:
  tracked = {**tracked, 'observations': separated['observations']}
  separated_status = separated['status']
 else:
  separated_status = 'not_run'
 # Enable only the explicitly reviewed pilot paths with matching native clip timing.
 wide_variant = wide_reviews.get(r.shot_id)
 if wide_variant and clean_data:
  wide_path = ROOT/f'outputs/native_review_wide/{r.shot_id}.json'
  extension_path = ROOT/f'data/processed/ball_extension_{wide_variant}/{r.shot_id}.json'
  if wide_path.is_file() and extension_path.is_file() and wide_path.with_suffix('.mp4').is_file():
   wide = json.loads(wide_path.read_text())
   ext = json.loads(extension_path.read_text())
   prefix = ext['observations'][:ext['prefix_length']]
   if wide['candidate_signature'] == ext['candidate_signature'] and prefix == clean_data['observations']:
    native = {**wide, 'media_directory': 'native_review_wide'}
    tracked = {**tracked, 'observations': ext['observations']}
    separated_status = 'sample_reviewed_ball_extension'
 ball_rejection = ball_rejections.get(r.shot_id)
 if ball_rejection:
  clean_data = None
  tracked = {**tracked, 'observations': []}
  separated_status = 'rejected_after_native_frame_review'
 used_obs = []
 if clean_data and 'observations' in clean_data and 'seed' in clean_data:
  t0 = clean_data['seed'].get('playback_s', clean_data['observations'][0]['playback_s'] if clean_data['observations'] else 0)
  used_obs = [o for o in clean_data['observations'] if 0 < o.get('playback_s', 0) - t0 <= 0.150]
 used_count = len(used_obs)
 used_range = f"{used_obs[0]['frame']}–{used_obs[-1]['frame']}" if used_obs else None
 c_curve=float(labels.loc[r.shot_id,'Curve signed (ft; R+)']) if r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id,'Curve signed (ft; R+)']) else None
 c_side=float(labels.loc[r.shot_id,'Carry side (ft; R+)']) if r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id,'Carry side (ft; R+)']) else None
 c_height=float(labels.loc[r.shot_id,'Height (ft)']) if r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id,'Height (ft)']) else None
 c_hang=float(labels.loc[r.shot_id,'Hang time (s)']) if r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id,'Hang time (s)']) else None
 c_carry=float(labels.loc[r.shot_id,'Carry flat (yd)']) if r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id,'Carry flat (yd)']) else None
 fc_flat=forecast(tracked['observations'],1100.,curve_ft=c_curve,carry_side_ft=c_side,height_ft=c_height,hang_time_s=c_hang,carry_yd=c_carry)
 forecasts={str(y):fc_flat for y in [1000,1100,1200]}

 club_path=ROOT/f'data/processed/club_display/{r.shot_id}.json'
 club_display=json.loads(club_path.read_text()) if club_path.exists() else None
 fusion_path=ROOT/f'data/processed/club_fusion/{r.shot_id}.json'
 club_fusion=json.loads(fusion_path.read_text()) if fusion_path.exists() else None
 pf_path=ROOT/f'data/processed/poseformer_v2/{r.shot_id}.json'
 poseformer_v2=json.loads(pf_path.read_text()) if pf_path.exists() else None
 has_radar = r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id, 'Carry flat (yd)'])
 has_geom = (r.shot_id in geom_df.index and 
             pd.notna(geom_df.loc[r.shot_id, 'proj_vy']) and 
             pd.notna(geom_df.loc[r.shot_id, 'proj_depth_rate']))
 used_primary = bool(has_radar and has_geom)

 if not has_radar:
  fallback_reason = "TrackMan radar did not register this strike (ground truth unavailable)"
 elif used_primary:
  fallback_reason = None
 else:
  obs_len = len(tracked['observations']) if tracked and 'observations' in tracked else 0
  if obs_len == 0:
   fallback_reason = "Tee-mat glare / unrecorded departure (camera missed ball lock)"
  elif obs_len < 5:
   fallback_reason = f"Track had only {obs_len} early frames (requires >= 5 points for perspective launch fit)"
  else:
   fallback_reason = f"Unrecovered departure trajectory ({obs_len} frames; non-linear fit in initial geometry pass)"

 if clean_data and has_radar:
  endpoint=endpoint_predictions.loc[r.shot_id]
  b_carry=float(endpoint['Carry flat (yd)'])
  b_dir=float(at_preds['dir'][r.shot_id]) if 'dir' in at_preds and r.shot_id in at_preds['dir'] else (float(vl_dir[r.shot_id]) if r.shot_id in vl_dir else 0.0)
  b_angle=float(endpoint['Launch angle (deg)'])
  b_speed=float(endpoint['Ball speed (mph)'])
  b_side=float(endpoint['Carry side (yd; R+)'])*3
  b_curve=float(cf_curve[r.shot_id]) if r.shot_id in cf_curve else (float(at_preds['curve'][r.shot_id]) if 'curve' in at_preds and r.shot_id in at_preds['curve'] else None)
  b_height=float(endpoint['Height (ft)'])
  b_land=float(endpoint['Landing angle (deg)'])
  b_hang=float(endpoint['Hang time (s)'])
  b_final_side=float(endpoint['Total side (yd; R+)'])*3
  b_total=float(endpoint['Total (yd)'])
 else:
  b_carry=b_dir=b_angle=b_speed=b_side=b_curve=b_height=b_land=b_hang=b_total=b_final_side=None
 display_track, ball_motion_stop = truncate_stationary_tail(tracked['observations'])
 motion_path = ROOT/f'data/processed/range_club_motion/{r.shot_id}.json'
 motion = json.loads(motion_path.read_text()) if motion_path.exists() else None
 alternate = motion_path.with_stem(motion_path.stem+'_alternate')
 if motion and alternate.exists():
  alternate_motion=json.loads(alternate.read_text())
  motion['alternate']=alternate_motion['direction']
  motion['alternate_hand_offset']=alternate_motion.get('hand_offset')
 rows.append(dict(club_motion=motion,clubhead_frame_map=head_data.get('review_frame_map',[]),ball_forecasts=forecasts,clubhead_track=head_track,club_native=separated['club_frames'] if separated else [],ball_separation_status=separated_status,ball_rejection=ball_rejection,club_display=club_display,club_fusion=club_fusion,poseformer_v2=poseformer_v2,avatar_motion=avatar,native=native,ball_track=display_track,ball_motion_stop=ball_motion_stop,reviewed_points=reviewed_points[reviewed_points.shot_id==r.shot_id][['playback_s','x','y']].to_dict('records'),departure_interval=intervals.loc[r.shot_id,['lower_s','upper_s','width_ms']].to_dict() if r.shot_id in intervals.index else None,observations=observations,id=r.shot_id,golfer=r.golfer,session=r.session,start=p['clip_start_playback_s'],fps=p['fps'],world=p['estimated_world_landmarks'],landmarks=p['landmarks'],
  carry=labels.loc[r.shot_id,'Carry flat (yd)'],
  total_actual=float(labels.loc[r.shot_id,'Total (yd)']) if r.shot_id in labels.index and pd.notna(labels.loc[r.shot_id,'Total (yd)']) else None,
  ball_angle_deg=clean_data['metrics']['ball_angle_deg'] if clean_data and 'metrics' in clean_data else None,
  launch_direction=labels.loc[r.shot_id,'Launch direction (deg; R+)'] if r.shot_id in labels.index else None,
  ball_speed=labels.loc[r.shot_id,'Ball speed (mph)'],launch=labels.loc[r.shot_id,'Launch angle (deg)'],
  curve_ft=c_curve,carry_side_ft=c_side,height_ft=c_height,hang_time_s=c_hang,
  best_prediction=b_carry,best_launch_direction=b_dir,best_launch_angle=b_angle,best_ball_speed=b_speed,
  total_side_ft=labels.loc[r.shot_id,'Total side (ft; R+)'],best_total_side_ft=b_final_side,
  best_carry_side_ft=b_side,best_curve_ft=b_curve,best_height_ft=b_height,best_landing_angle=b_land,best_hang_time=b_hang,best_total_yd=b_total,
  launch_frames_used=used_count,launch_frames_range=used_range,
  used_primary_model=used_primary,fallback_reason=fallback_reason,
  club_available=bool(pd.notna(r.club_peak_candidate_s)),alignment_residual=r.club_watch_residual_s,
  workbook_row=int(r.workbook_row),
  gate_probability=float(gate.loc[r.shot_id,'low_probability']) if r.shot_id in gate.index else None,
  assisted_prediction=float(gate.loc[r.shot_id,'allshot_prediction']) if r.shot_id in gate.index else None,
  gate_error=bool((gate.loc[r.shot_id,'low_probability']>=.5)!=(gate.loc[r.shot_id,'actual']<100)) if r.shot_id in gate.index else False,
  prediction=float(pred.loc[r.shot_id,'predicted']) if r.shot_id in pred.index else None,
  top=phase.loc[r.shot_id,'top_candidate_s'],top_status=phase.loc[r.shot_id,'top_status'],source=r.video_candidate,sha256=inventory.loc[r.video_candidate,'sha256']))
summary=score_cohorts(rows,endpoint_predictions,labels)
summary.to_csv(ROOT/'experiments/label_stack_v1/deployed_cohorts.csv',index=False)
summary_html=[]
for cohort,group in summary.groupby('cohort',sort=False):
 title=f"{cohort} · {group.shots.iloc[0]} shots"
 table=group[['metric','model','baseline','reduction']].copy()
 table.columns=['Mean error','Model (yd)','Training average (yd)','Error reduction (%)']
 summary_html.append('<h3>'+title+'</h3>'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}'))
payload=encode_display_data(rows)
template=(ROOT/'scripts/viewer_template.html').read_text().replace('__THREE__','/* '+(ROOT/'vendor/three/LICENSE').read_text()+' */\n'+(ROOT/'vendor/three/three.min.js').read_text()).replace('__SCENE__',(ROOT/'scripts/range_scene.js').read_text()).replace('__POSEFORMER_SCENE__',(ROOT/'scripts/poseformer_scene.js').read_text())
(out/'index.html').write_text(template.replace('__DATA__',payload).replace('__ROLLOUT_SUMMARY__',''.join(summary_html)))
print('Built local viewer for',len(rows),'shots:',out/'index.html')
