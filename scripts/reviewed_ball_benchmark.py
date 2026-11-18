"""Separate small-cohort feasibility study using approximate reviewed ball points.
This is assisted tracking, never an end-to-end automatic validation score.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,log_loss
from benchmark import load_data,model,metrics
from contact_benchmark import classifier,probabilities,class_metrics
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 bc,d=load_data();b=pd.read_csv(ROOT/'data/processed/reviewed_ball_features.csv');c=pd.read_csv(ROOT/'data/processed/contact_features.csv')
 d=d.merge(b,on='shot_id',validate='one_to_one').merge(c,on='shot_id',how='left',validate='one_to_one')
 d=d.merge(pd.read_csv(ROOT/'experiments/baseline_v1/splits.csv')[['shot_id','golfer_fold']],on='shot_id',validate='one_to_one')
 d['short_carry']=(d['Carry flat (yd)']<60).astype(int);d['low_launch']=(d['Launch angle (deg)']<10).astype(int);d['directional_miss']=(d['Launch direction (deg; R+)'].abs()>10).astype(int)
 ball=[c for c in b if c.startswith('reviewed_ball_')];contact=[c for c in c if c.startswith('contact_')]
 cw=bc['camera_features']+[c for c in d if c.startswith('watch_gyro_') or c.startswith('watch_accel_')]
 cohorts=[('reviewed_all',d,{'camera_watch':cw,'reviewed_ball':ball,'camera_watch_reviewed_ball':cw+ball},['Carry flat (yd)','Launch angle (deg)','Launch direction (deg; R+)','short_carry','low_launch','directional_miss']),
 ('reviewed_matched_club',d.dropna(subset=['contact_gyro_hp_rms_ratio']),{'reviewed_ball':ball,'reviewed_ball_club_contact':ball+contact},['Carry flat (yd)','short_carry','low_launch'])]
 out=ROOT/'experiments/reviewed_ball_v1';out.mkdir(exist_ok=True);preds=[];choices=[]
 (out/'config.json').write_text(json.dumps(dict(seed=396,features={cohort:sets for cohort,_,sets,_ in cohorts},n={cohort:len(sub) for cohort,sub,_,_ in cohorts},scheme='leave_golfer_out',inner_splits=3,selection='pooled inner MAE or log loss',limitations='Small deterministically sampled and visibility-selected cohort; AI-reviewed approximate sparse ball points and five coarse timing corrections; second reviewer outstanding; not autonomous tracker; repeated exploratory use of dataset.'),indent=2))
 for cohort,sub,sets,tasks in cohorts:
  for task in tasks:
   regression=task not in ['short_carry','low_launch','directional_miss'];families=['mean','ridge10','extra_trees','svr'] if regression else ['prior','logistic','extra_trees']
   for feature,cols in sets.items():
    for fold in sorted(sub.golfer_fold.unique()):
     tr=sub[sub.golfer_fold!=fold];te=sub[sub.golfer_fold==fold];scores={}
     for family in families:
      truth=[];pred=[]
      for ti,vi in GroupKFold(3).split(tr,groups=tr.golfer_fold):
       a=tr.iloc[ti];b=tr.iloc[vi];pipe=model(family,396) if regression else classifier(family,a[task],396)
       pipe.fit(a[cols],a[task]);truth.extend(b[task]);pred.extend(pipe.predict(b[cols]) if regression else probabilities(pipe,b[cols]))
      scores[family]=float(mean_absolute_error(truth,pred) if regression else log_loss(truth,np.clip(pred,1e-6,1-1e-6),labels=[0,1]))
     winner=min(families,key=lambda k:scores[k]);pipe=model(winner,396) if regression else classifier(winner,tr[task],396);pipe.fit(tr[cols],tr[task]);pr=pipe.predict(te[cols]) if regression else probabilities(pipe,te[cols])
     choices.append(dict(cohort=cohort,task=task,features=feature,fold=int(fold),model=winner,inner_scores=scores))
     for sid,g,y,p in zip(te.shot_id,te.golfer,te[task],pr):preds.append(dict(cohort=cohort,task=task,features=feature,fold=int(fold),shot_id=sid,golfer=g,actual=float(y),predicted=float(p),model=winner))
    print(cohort,task,feature,flush=True)
 p=pd.DataFrame(preds);p.to_csv(out/'predictions.csv',index=False);scores=[]
 for (cohort,task,features),g in p.groupby(['cohort','task','features']):
  regression=task not in ['short_carry','low_launch','directional_miss'];scores.append(dict(cohort=cohort,task=task,features=features,n=len(g),**(metrics(g.actual,g.predicted) if regression else class_metrics(g.actual,g.predicted))))
 pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False);(out/'inner_selections.json').write_text(json.dumps(choices,indent=2));print(pd.DataFrame(scores).to_string(index=False))
