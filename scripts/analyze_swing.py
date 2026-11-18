"""Process a manually selected 3.5-second window into an isolated local demo.
No raw file changes or uploads. Research manifest and evaluation data untouched.
"""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from types import SimpleNamespace
import joblib,pandas as pd
from process_pose import process
from pose_observations import observations
from avatar_motion import complete
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def analyze(video,start,name):
 video=Path(video).expanduser().resolve()
 if not video.is_file():raise ValueError('Video file does not exist')
 if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',name):raise ValueError('Name must contain 1–64 letters, numbers, underscores or hyphens')
 if start<0:raise ValueError('Clip start must be nonnegative')
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type','-of','json',str(video)]))
 if not any(s['codec_type']=='video' for s in info['streams']):raise ValueError('No video stream')
 if float(info['format']['duration'])<start+3.5:raise ValueError('Selected window must contain 3.5 seconds; move the start earlier')
 out=ROOT/'outputs/new_swings'/name;out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'configs/vision.json').read_text());cfg.update(pose_output_dir=str(out.relative_to(ROOT)),pose_cache_dir=str((out/'cache').relative_to(ROOT)),description='User-selected whole-swing window. Video-only new-recording prototype; not independently validated.')
 provenance=out/'provenance.json';source_hash=sha256(video)
 if provenance.exists():
  old=json.loads(provenance.read_text())
  if old['source_sha256']!=source_hash or old['clip_start_s']!=start:raise ValueError('This output name already belongs to a different recording/window; choose a new name')
 row=SimpleNamespace(shot_id=name,video_candidate=str(video),video_peak_candidate_s=start+cfg['clip_before_candidate_s'])
 f=process(row,cfg);p=json.loads((out/'cache'/f'{name}.json').read_text());predictions=[]
 availability=json.loads((ROOT/'configs/new_video.json').read_text());enough_pose=f['pose_detection_fraction']>=availability['minimum_pose_detection_fraction']
 manifest=json.loads((ROOT/'models/reference_ridge/manifest.json').read_text()) if enough_pose else []
 for record in manifest:
  if record['features']!='camera':continue
  path=ROOT/record['path']
  if sha256(path)!=record['sha256']:raise ValueError('Reference model hash mismatch')
  artifact=joblib.load(path);values=pd.DataFrame([f]).reindex(columns=artifact['features'])
  if values.isna().all(axis=None):raise ValueError('No usable model input features')
  value=float(artifact['model'].predict(values)[0]);predictions.append(dict(target=artifact['target'],predicted=value,model_sha256=record['sha256'],status='full_training_data_reference_model_new_video_unvalidated'))
 target=out/f'{name}_browser_overlay.mp4'
 if not target.exists() or target.stat().st_mtime_ns<(out/f'{name}_overlay.mp4').stat().st_mtime_ns:
  subprocess.run(['ffmpeg','-v','error','-i',str(out/f'{name}_overlay.mp4'),'-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(target)],capture_output=True,check=True)
 carry=next((r['predicted'] for r in predictions if r['target']=='Carry flat (yd)'),None)
 item=dict(avatar_motion=complete(p),reference_predictions={r['target']:r['predicted'] for r in predictions},prediction_status='experimental_reference_fit' if enough_pose else 'abstained_insufficient_pose',id=name,golfer='New recording',session=name,start=p['clip_start_playback_s'],fps=p['fps'],world=p['estimated_world_landmarks'],landmarks=p['landmarks'],observations=observations(p),carry=None,prediction=carry,assisted_prediction=None,gate_probability=None,gate_error=False,ball_speed=None,launch=None,club_available=False,alignment_residual=None,workbook_row=None,top=None,top_status='unreviewed',source=str(video),sha256=source_hash,departure_interval=None,mode='new',overlay_url=target.name,input_url=f'{name}_input.mp4')
 template=(ROOT/'scripts/viewer_template.html').read_text().replace('__THREE__','/* '+(ROOT/'vendor/three/LICENSE').read_text()+' */\n'+(ROOT/'vendor/three/three.min.js').read_text()).replace('__SCENE__',(ROOT/'scripts/range_scene.js').read_text());template=template.replace('All 76 swings','New recording').replace('76 strikes · five golfers · six sessions','One new recording · video-only prototype').replace('This demo explores recorded sessions; new-recording inference is not yet integrated.','This recording is kept separate from the research evaluation.').replace('<option value="errors">Six classification errors</option>','').replace('<option value="assisted">Reviewed ball pilot</option>','')
 template=template.replace('../../reports/','../../../reports/');(out/'index.html').write_text(template.replace('__DATA__',json.dumps([item]).replace('NaN','null').replace('</','<\\/')))
 (out/'predictions.json').write_text(json.dumps(predictions,indent=2));provenance.write_text(json.dumps(dict(source=str(video),source_sha256=source_hash,clip_start_s=start,duration_s=3.5,selection='manual',configuration=cfg,availability_rule=availability,pose_detection_fraction=f['pose_detection_fraction'],model_status='reference fits; new-video accuracy unvalidated' if enough_pose else 'abstained_insufficient_pose'),indent=2))
 print(out/'index.html');return out

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--video',required=True);parser.add_argument('--start',type=float,required=True,help='Start of a 3.5-second window containing one complete swing');parser.add_argument('--name',required=True);args=parser.parse_args();analyze(args.video,args.start,args.name)
