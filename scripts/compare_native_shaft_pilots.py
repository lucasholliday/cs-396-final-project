"""Score shaft directions on two swings, preserving misses in the denominator."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'experiments/native_swing_shaft_v2'
sid_audit = pd.read_csv(out / 'sid_anchor_errors.csv')
marks = json.loads((ROOT / 'annotations/club_keypoints_sid_1_01.json').read_text())['points']
by_frame = {m['frame']: m for m in marks}
sid_reference = []
for row in sid_audit.itertuples():
    m = by_frame[row.review_frame]
    dx, dy = np.subtract(m['head'], m['grip'])
    sid_reference.append(dict(source_frame=row.source_frame, dir_deg=np.degrees(np.arctan2(dy, dx))))
references = {'sid_1_01': pd.DataFrame(sid_reference),
              'lucas_1_02': pd.read_csv(out / 'lucas_blind_directions.csv').dropna()}
rows = []
for shot, reference in references.items():
    for variant, dirname in [('first', 'native_swing_shaft'), ('body filter', 'native_swing_shaft_v2'),
                             ('higher resolution', None)]:
        if dirname:
            trace = json.loads((ROOT / f'data/processed/{dirname}/{shot}.json').read_text())
            candidates = pd.DataFrame([dict(source_frame=p['frame'],
                                      estimate=np.degrees(np.arctan2(*p['v'][::-1]))) for p in trace['candidates']])
        else:
            candidates = pd.read_csv(ROOT / f'experiments/native_shaft_highres/{shot}.csv').rename(columns={'dir_deg':'estimate'})
        audit = reference.merge(candidates, how='left', on='source_frame')
        audit['error'] = abs((audit.estimate - audit.dir_deg + 180) % 360 - 180)
        audit['shot'] = shot
        audit['method'] = variant
        rows.append(audit)
data = pd.concat(rows, ignore_index=True)
data.to_csv(out / 'pilot_comparison.csv', index=False)
print(data.groupby(['shot','method']).error.agg(['size','count',lambda s: (s<=10).sum()]))
