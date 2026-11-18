"""Paired golfer-cluster resampling of nested carry residuals (five clusters only)."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'experiments/nested_carry_v1'
p=pd.read_csv(out/'predictions.csv')
q=p[(p.cohort=='matched_club')&(p.scheme=='leave_golfer_out')]
rng=np.random.default_rng(396);results=[]
for a,b in [('camera_watch','camera'),('all','camera_watch'),('all','camera')]:
    left=q[q.features==a].set_index('shot_id');right=q[q.features==b].set_index('shot_id').loc[left.index]
    delta=(left.residual.abs()-right.residual.abs()).to_numpy();groups=left.golfer.to_numpy();names=np.unique(groups)
    draws=[np.concatenate([delta[groups==g] for g in rng.choice(names,len(names),replace=True)]).mean() for _ in range(2000)]
    low,high=np.quantile(draws,[.025,.975])
    results.append(dict(comparison=f'{a} minus {b}',delta_mae=delta.mean(),low=low,high=high,clusters=len(names)))
pd.DataFrame(results).to_csv(out/'paired_uncertainty.csv',index=False)
print('Saved nominal cluster intervals; no multiple-comparison correction; five golfers.')
