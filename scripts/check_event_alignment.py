"""First/last wrist-event visual checks using unverified camera creation timestamps."""
import json
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT/'configs/audit.json').read_text())
events = pd.read_csv(ROOT/'data/manifests/event_candidates.csv')
out = ROOT/'outputs/alignment'
out.mkdir(parents=True, exist_ok=True)
records = []
for session,(video,_,_) in config['sessions'].items():
    subset = events[(events.session==session)&(events.modality=='watch')]
    frames=[]
    for kind,(_,row) in zip(['first','last'],subset.iloc[[0,-1]].iterrows()):
        for j,delta in enumerate([-1,-.5,0,.5,1]):
            t = row.video_time_from_metadata_s+delta
            path = out/f'{Path(video).stem}_{kind}_{j}.png'
            records.append(dict(session=session,event=kind,ordinal=int(row.ordinal),playback_s=t,offset_s=delta,path=str(path.relative_to(ROOT))))
            if not path.exists():
                subprocess.run(['ffmpeg','-v','error','-ss',str(max(0,t)),'-i',str(ROOT/video),
                    '-frames:v','1','-vf','scale=320:180','-y',str(path)],check=True,capture_output=True)
            frames.append(path)
    cmd=['ffmpeg','-v','error']
    for p in frames:
        cmd.extend(['-i',str(p)])
    cmd.extend(['-filter_complex','[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[a];[5:v][6:v][7:v][8:v][9:v]hstack=inputs=5[b];[a][b]vstack=inputs=2',
        '-frames:v','1','-y',str(out/f'{Path(video).stem}_first_last.png')])
    subprocess.run(cmd,check=True,capture_output=True)
pd.DataFrame(records).to_csv(out/'preview_times.csv',index=False)
print('Created six first/last-event contact sheets; timings remain provisional.')
