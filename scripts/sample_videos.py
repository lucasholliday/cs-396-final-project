"""Cached sparse visual/duplicate-frame diagnostics, not a complete swing audit."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'outputs/audit'
out.mkdir(parents=True, exist_ok=True)
media = json.loads((ROOT/'data/manifests/media_audit.json').read_text())
results = []
for entry in media:
    if not entry['path'].endswith('.MOV'):
        continue
    p = ROOT/entry['path']
    duration = float(next(s['duration'] for s in entry['metadata']['streams'] if s['codec_type']=='video'))
    frames = []
    for i, fraction in enumerate([.1,.5,.9]):
        t = duration*fraction
        frame = out/f'{p.stem}_{i}.png'
        if not frame.exists():
            subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',str(p),'-frames:v','1',
                            '-vf','scale=320:180','-y',str(frame)],check=True,capture_output=True)
        frames.append(frame)
        hashes = out/f'{p.stem}_{i}.framemd5'
        if not hashes.exists():
            subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',str(p),'-t','0.25',
                '-an','-vsync','0','-f','framemd5','-y',str(hashes)],check=True,capture_output=True)
        lines = [l for l in hashes.read_text().splitlines() if not l.startswith('#')]
        values = [l.split(',')[-1].strip() for l in lines]
        results.append(dict(path=entry['path'], playback_start_s=t, window_s=.25,
            decoded_frames=len(values), adjacent_exact_duplicates=sum(a==b for a,b in zip(values,values[1:]))))
    subprocess.run(['ffmpeg','-v','error', '-i',str(frames[0]),'-i',str(frames[1]),'-i',str(frames[2]),
        '-filter_complex','hstack=inputs=3','-frames:v','1','-y',str(out/f'{p.stem}_contact.png')],check=True,capture_output=True)
(out/'sample_decode_results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
