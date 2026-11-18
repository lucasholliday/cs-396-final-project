"""Descriptive watch/carry relationships; no model selection or causal claims."""
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parents[1]
s=pd.read_csv(ROOT/'data/processed/sensor_features.csv');labels=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv')
d=s.merge(labels,left_on='shot_id',right_on='Shot ID',validate='one_to_one').dropna(subset=['Carry flat (yd)'])
rows=[];by_golfer=[]
for c in [c for c in s if c.startswith('watch_')]:
 x=d[c];y=d['Carry flat (yd)'];xd=x-d.groupby('Golfer')[c].transform('mean');yd=y-d.groupby('Golfer')['Carry flat (yd)'].transform('mean');above=y>=100
 rows.append(dict(feature=c,n=len(d),pearson=pearsonr(x,y).statistic,spearman=spearmanr(x,y).statistic,within_golfer_pearson=pearsonr(xd,yd).statistic,above100_n=int(above.sum()),above100_spearman=spearmanr(x[above],y[above]).statistic))
 for golfer,g in d.groupby('Golfer'):
  by_golfer.append(dict(golfer=golfer,feature=c,n=len(g),spearman=spearmanr(g[c],g['Carry flat (yd)']).statistic))
out=ROOT/'experiments/watch_carry_relationships_v1';out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/'correlations.csv',index=False);pd.DataFrame(by_golfer).to_csv(out/'per_golfer.csv',index=False)
d[['shot_id','Golfer','Carry flat (yd)']+[c for c in s if c.startswith('watch_')]].to_csv(out/'shot_values.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
