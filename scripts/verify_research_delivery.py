"""Verify derived deliverable contracts without rehashing unchanged 12-GB originals."""
import json,re,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def check_media(path):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height:format=duration','-of','json',str(path)]))
 s=d['streams'][0]
 assert s['codec_name']=='h264' and s['width']==540 and s['height']==960,path
 assert float(d['format']['duration'])>=3.4,path
 return path.name

if __name__=='__main__':
 files=list((ROOT/'outputs/research_viewer/videos').glob('*.mp4'))
 with ThreadPoolExecutor(max_workers=4) as pool:checked=list(pool.map(check_media,files))
 assert len(checked)==76
 html=(ROOT/'outputs/research_viewer/index.html').read_text()
 data=json.loads(re.search(r'const data=(.*?);const \$',html).group(1))
 assert len(data)==76 and len(set(d['id'] for d in data))==76
 for d in data:
  assert (ROOT/'outputs/pose'/f"{d['id']}_input.mp4").exists()
  assert len(d['world'])==len(d['landmarks'])==105
  assert len(d['sha256'])==64
 p=pd.read_csv(ROOT/'experiments/early_swing_v1/predictions.csv')
 assert not p.duplicated(['scheme','features','shot_id']).any()
 assert p.groupby(['scheme','features']).size().eq(72).all()
 inventory=pd.read_csv(ROOT/'data/manifests/file_inventory.csv').set_index('path').sha256
 refs=pd.read_csv(ROOT/'annotations/ball_reference_points.csv')
 assert (refs.source.map(inventory)==refs.source_sha256).all()
 result=dict(viewer_records=76,h264_media_checked=76,pose_frames_per_clip=105,early_oof_evaluations=6,source_bound_point_review_panels=len(refs),browser_ui_check='not_completed_browser_access_timeout')
 (ROOT/'reports/viewer_verification.json').write_text(json.dumps(result,indent=2));print(result)
