"""NON-DEPLOYABLE diagnostic: recorded ball speed/launch as inputs.
Quantifies information potential of accurate launch measurements. These are
outcome labels, not camera estimates, so results must never be called camera accuracy.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from audited_launch_benchmark import estimator
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 d=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').dropna(subset=['Carry flat (yd)']);v=d['Ball speed (mph)']*.44704;angle=np.radians(d['Launch angle (deg)']);d['vacuum_range_yd']=v*v*np.sin(2*angle)/9.80665/.9144;d['speed2']=v*v;cols=['Ball speed (mph)','Launch angle (deg)','vacuum_range_yd','speed2'];preds=[]
 for cutoff in [0,100]:
  q=d if cutoff==0 else d[d['Carry flat (yd)']>cutoff]
  for golfer in q.Golfer.unique():
   tr=q[q.Golfer!=golfer];te=q[q.Golfer==golfer];scores={}
   for family in ['mean','ridge10','ridge100','svr1','svr10','trees']:
    y=[];p=[]
    for a,b in GroupKFold(min(3,tr.Golfer.nunique())).split(tr,groups=tr.Golfer):
     pipe=estimator(family);pipe.fit(tr.iloc[a][cols],tr.iloc[a]['Carry flat (yd)']);y.extend(tr.iloc[b]['Carry flat (yd)']);p.extend(pipe.predict(tr.iloc[b][cols]))
    scores[family]=mean_absolute_error(y,p)
   winner=min(scores,key=scores.get);pipe=estimator(winner);pipe.fit(tr[cols],tr['Carry flat (yd)']);p=pipe.predict(te[cols])
   for sid,y,z in zip(te['Shot ID'],te['Carry flat (yd)'],p):preds.append(dict(cutoff=cutoff,shot_id=sid,golfer=golfer,actual=y,predicted=z,abs_error=abs(y-z),model=winner))
 out=ROOT/'experiments/launch_oracle_diagnostic';out.mkdir(exist_ok=True);pd.DataFrame(preds).to_csv(out/'predictions.csv',index=False);summary=pd.DataFrame(preds).groupby('cutoff').agg(n=('shot_id','size'),mae=('abs_error','mean'));summary.to_csv(out/'metrics.csv');print(summary)
