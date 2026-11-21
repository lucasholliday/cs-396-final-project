"""Pilot a background-edge check on the cached native frames of Sid 01."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'experiments/native_swing_shaft_v2'
shot = 'sid_1_01'
trace = json.loads((ROOT / f'data/processed/native_swing_shaft_v2/{shot}.json').read_text())
with np.load(ROOT / f'outputs/native_club/{shot}_frames.npz') as data:
    frames = data['frames']
    assert np.allclose(data['pts'] - trace['start'], trace['times'], atol=.001)
background = np.median(frames[::20], axis=0).astype(np.uint8)
# Leave a small tolerance for slight camera movement and the resize used in detection.
bg_edges = cv2.Canny(background, 40, 100)
distance = cv2.distanceTransform(255 - bg_edges, cv2.DIST_L2, 3)
rows = []
for candidate in trace['candidates']:
    a, b = [2 * np.array(candidate[k]) for k in ('a', 'b')]
    samples = np.rint(np.linspace(a, b, 48)[8:-4]).astype(int)
    x, y = samples.T
    shared = np.mean(distance[y, x] <= 2.5)
    rows.append(dict(source_frame=candidate['frame'], background_edge_fraction=shared,
                     accepted=shared < .75))
result = pd.DataFrame(rows)
result.to_csv(out / 'sid_background_check.csv', index=False)
reference = pd.read_csv(out / 'sid_anchor_errors.csv')
print(reference.merge(result, on='source_frame').round(2).to_string(index=False))
