"""Extend review clips to up to 480 frames; landing visibility is not guaranteed.
Encoded timestamps plus requested seek time are only approximate source times.
This helper does not verify correspondence to native source frames."""
import json, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = '/Users/lucasholliday/miniconda3/bin/ffmpeg'
FFPROBE = '/Users/lucasholliday/miniconda3/bin/ffprobe'

def process_shot(r):
    out = ROOT / 'outputs/native_review'
    target = out / f'{r.shot_id}.mp4'
    meta = out / f'{r.shot_id}.json'
    
    start = float(r.video_peak_candidate_s - 1.25)
    num_frames = 480 # ~3.0 seconds at ~160 fps
    
    cmd_ffmpeg = [
        FFMPEG, '-v', 'error',
        '-ss', str(start),
        '-i', str(ROOT / r.video_candidate),
        '-frames:v', str(num_frames),
        '-an',
        '-vf', 'crop=780:1100:300:650',
        '-vsync', '0',
        '-enc_time_base', '-1',
        '-video_track_timescale', '60000',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-y', str(target)
    ]
    subprocess.run(cmd_ffmpeg, check=True)
    
    cmd_ffprobe = [
        FFPROBE, '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'json', str(target)
    ]
    info = json.loads(subprocess.check_output(cmd_ffprobe))
    review_pts = [float(f['best_effort_timestamp_time']) for f in info['frames']]
    source_pts = [float(start + r_pts) for r_pts in review_pts]
    
    candidate_sig = ''
    cand_path = ROOT / f'data/processed/ball_v7/{r.shot_id}.json'
    if cand_path.exists():
        try:
            candidate_sig = json.loads(cand_path.read_text()).get('signature', '')
        except Exception:
            pass
            
    d = dict(
        encoding_version=3,
        shot_id=r.shot_id,
        source=r.video_candidate,
        candidate_signature=candidate_sig,
        review_pts=review_pts,
        source_pts=source_pts,
        crop=[300, 650, 780, 1100],
        cadence_error_max_s=None,
        source_pts_verified=False,
        semantics='Up to 480 encoded review frames. Source timestamps are seek-based estimates; cadence and landing are unverified.'
    )
    meta.write_text(json.dumps(d))
    return r.shot_id, len(review_pts)

def main():
    m = pd.read_csv(ROOT / 'data/manifests/shot_manifest.csv')
    print(f'Extending native review clips for {len(m)} shots...')
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(process_shot, [r for r in m.itertuples()]))
    elapsed = time.time() - t0
    print(f'Finished extending {len(results)} native review clips in {elapsed:.1f}s!')

if __name__ == '__main__':
    main()
