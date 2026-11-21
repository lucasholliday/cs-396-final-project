"""Render selected successes, missed detections and direction errors on source frames."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
shot = 'sid_1_01'
observations = json.loads((ROOT / f'data/processed/native_swing_shaft_v2/{shot}.json').read_text())
candidates = {row['frame']: row for row in observations['candidates']}
references = json.loads((ROOT / f'annotations/club_keypoints_{shot}.json').read_text())
marks = {row['frame']: row for row in references['points']}
audit = pd.read_csv(ROOT / 'experiments/native_swing_shaft_v2/sid_anchor_errors.csv').set_index('review_frame')
with np.load(ROOT / f'outputs/native_club/{shot}_frames.npz') as data:
    frames = data['frames']
tiles = []
for review_frame in (0, 46, 52, 58, 64, 80):
    row = audit.loc[review_frame]
    idx = int(row.source_frame)
    img = cv2.cvtColor(cv2.resize(frames[idx], (270, 480)), cv2.COLOR_GRAY2BGR)
    mark = marks[review_frame]
    grip, head = [tuple(np.rint(np.array(mark[k]) / 2).astype(int)) for k in ('grip', 'head')]
    cv2.arrowedLine(img, grip, head, (255, 225, 40), 2, cv2.LINE_AA, tipLength=.08)
    msg = 'No detection'
    if idx in candidates:
        a, b = [tuple(np.rint(candidates[idx][k]).astype(int)) for k in ('a', 'b')]
        cv2.arrowedLine(img, a, b, (0, 150, 255), 2, cv2.LINE_AA, tipLength=.1)
        msg = f'{row.dir_error_deg:.1f} deg error'
    tile = cv2.copyMakeBorder(img, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(tile, f'Frame {idx}  |  {msg}', (9, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (240,)*3, 1, cv2.LINE_AA)
    tiles.append(tile)
sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
sheet = cv2.copyMakeBorder(sheet, 38, 0, 0, 0, cv2.BORDER_CONSTANT)
cv2.putText(sheet, 'Sid 01    Cyan: approximate reference    Orange: automatic shaft segment',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .59, (240,)*3, 1, cv2.LINE_AA)
out = ROOT / 'experiments/native_swing_shaft_v2/sid_visual_check.jpg'
assert cv2.imwrite(str(out), sheet)
