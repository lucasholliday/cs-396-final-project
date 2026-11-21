"""Check whether a few visually located clubheads can support native-frame tracking."""
import json
import subprocess

import cv2
import numpy as np
import pandas as pd

from native_club_flow import ROOT, walk

cv2.setNumThreads(1)
shot = 'lucas_1_02'
out = ROOT / 'experiments/native_swing_shaft_v2'
trace = json.loads((ROOT / f'data/processed/native_swing_shaft_v2/{shot}.json').read_text())
src = pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').set_index('shot_id').loc[shot,'video_candidate']
cmd = ['ffmpeg','-v','error','-threads','2','-ss',str(trace['start']),'-i',str(ROOT/src),
       '-t','3.5','-an','-filter_threads','1','-vf','scale=540:960','-vsync','0',
       '-f','rawvideo','-pix_fmt','gray','-']
frames = np.frombuffer(subprocess.check_output(cmd),np.uint8).reshape(-1,960,540)
assert len(frames) == len(trace['times'])
review = pd.read_csv(out/'lucas_head_review.csv')
seeds = review[review.use=='seed']
# Overlapping passes use the closest successful seed by frame distance.
best = {}
observations = {}
for row in seeds.itertuples():
    i = row.source_frame
    for indices in (range(i, -1, -1), range(i, len(frames))):
        points = walk(frames,indices,(2*row.x_270,2*row.y_480))
        for frame, xy in points.items():
            observations.setdefault(frame, {})[i] = xy
            if frame not in best or abs(frame-i) < abs(frame-best[frame][0]):
                best[frame] = (i,xy)
tracks = pd.DataFrame([dict(source_frame=k,seed=v[0],x=v[1][0],y=v[1][1]) for k,v in sorted(best.items())])
tracks.to_csv(out/'lucas_assisted_track.csv',index=False)
checks = review[review.use=='check'].merge(tracks,on='source_frame',how='left')
checks['err_px_540'] = np.hypot(checks.x-2*checks.x_270, checks.y-2*checks.y_480)
checks.to_csv(out/'lucas_assisted_checks.csv',index=False)
print(len(tracks),len(frames),checks[['source_frame','err_px_540']].round(1).to_string(index=False))

# Reject distant one-sided estimates. Keep points close in time to a visual mark
# or independently reached from a mark on either side with spatial agreement.
ts = np.asarray(trace['times'])
keep = []
for row in tracks.itertuples():
    frame = row.source_frame
    alternatives = observations[frame]
    xy = np.array([row.x, row.y])
    nearby = abs(ts[row.seed] - ts[frame]) <= .15
    corroborated = any((seed-frame)*(row.seed-frame) < 0 and np.linalg.norm(point-xy) <= 10
                       for seed, point in alternatives.items())
    keep.append(nearby or corroborated)
kept = tracks[keep].copy()
kept.to_csv(out/'lucas_assisted_gated.csv',index=False)
print('gated',len(kept),'of',len(frames),'frames')
print(checks.loc[checks.source_frame.isin(kept.source_frame),['source_frame','err_px_540']].round(1))
deltas = np.hypot(kept.x.diff(), kept.y.diff())[kept.source_frame.diff() == 1]
print('largest adjacent gated jump',deltas.max())
near_checks = pd.read_csv(out/'lucas_near_seed_checks.csv').merge(kept, on='source_frame', how='left')
near_checks['err_px_540'] = np.hypot(near_checks.x-2*near_checks.x_270, near_checks.y-2*near_checks.y_480)
near_checks.to_csv(out/'lucas_gated_spot_errors.csv', index=False)
print(near_checks[['source_frame','err_px_540']].round(1))
