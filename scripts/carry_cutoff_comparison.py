"""Matched-cohort cutoff diagnostics: rescore existing OOF models and refit restricted models.
Oracle outcome filtering is NOT an available-at-inference gate.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from benchmark import load_data,model
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'experiments/carry_cutoff_comparison_v1';out.mkdir(exist_ok=True)
base=ROOT/'experiments/baseline_v1';sets=json.loads((base/'config.json').read_text())['feature_sets'];cfg,d=load_data()
d=d.dropna(subset=sets['club']).merge(pd.read_csv(ROOT/'data/processed/contact_features.csv'),on='shot_id',validate='one_to_one').merge(pd.read_csv(base/'splits.csv')[['shot_id','golfer_fold']],on='shot_id',validate='one_to_one');assert len(d)==65
sets={k:sets[k] for k in ['watch','camera','camera_watch','all']};sets['camera_watch_club_contact']=json.loads((ROOT/'experiments/contact_quality_v1/config.json').read_text())['features']['camera_watch_club_contact']
p=pd.read_csv(ROOT/'experiments/nested_carry_v1/predictions.csv');p=p[(p.cohort=='matched_club')&(p.scheme=='leave_golfer_out')&p.features.isin(sets)].copy()
b=pd.read_csv(base/'predictions.csv');b=b[(b.cohort=='matched_club')&(b.scheme=='leave_golfer_out')&(b.target=='Carry flat (yd)')&(b.features=='camera')&(b.model=='mean')].copy();b['features']='mean'
c=pd.read_csv(ROOT/'experiments/contact_quality_v1/predictions.csv');c=c[(c.scheme=='leave_golfer_out')&(c.task=='Carry flat (yd)')&(c.features=='camera_watch_club_contact')]
p=pd.concat([p,b,c],ignore_index=True);rows=[];preds=[];choices=[]
for cutoff in [None,70,100]:
 ids=set(d.loc[d['Carry flat (yd)']>cutoff,'shot_id']) if cutoff is not None else set(d.shot_id)
 for f in ['mean',*sets]:
  q=p[(p.features==f)&p.shot_id.isin(ids)];assert len(q)==len(ids) and q.shot_id.is_unique
  rows.append(dict(protocol='existing_allshot_predictions',cutoff=cutoff,features=f,n=len(q),golfers=q.golfer.nunique(),mae=mean_absolute_error(q.actual,q.predicted)))
for cutoff in [70,100]:
 subset=d[d['Carry flat (yd)']>cutoff]
 for feature in ['mean',*sets]:
  cols=sets['camera'] if feature=='mean' else sets[feature];results=[]
  for fold in sorted(subset.golfer_fold.unique()):
   train=subset[subset.golfer_fold!=fold];test=subset[subset.golfer_fold==fold];scores={}
   families=['mean'] if feature=='mean' else ['mean','ridge10','extra_trees','svr']
   for family in families:
    truth=[];estimate=[]
    for ti,vi in GroupKFold(min(3,train.golfer_fold.nunique())).split(train,groups=train.golfer_fold):
     tr=train.iloc[ti];va=train.iloc[vi];pipe=model(family,cfg['seed']);pipe.fit(tr[cols],tr['Carry flat (yd)']);truth.extend(va['Carry flat (yd)']);estimate.extend(pipe.predict(va[cols]))
    scores[family]=float(mean_absolute_error(truth,estimate))
   winner=min(scores,key=scores.get);pipe=model(winner,cfg['seed']);pipe.fit(train[cols],train['Carry flat (yd)']);estimate=pipe.predict(test[cols]);choices.append(dict(cutoff=cutoff,features=feature,fold=int(fold),model=winner,scores=scores))
   for sid,g,y,pr in zip(test.shot_id,test.golfer,test['Carry flat (yd)'],estimate):results.append(dict(cutoff=cutoff,features=feature,shot_id=sid,golfer=g,actual=y,predicted=pr,model=winner))
  q=pd.DataFrame(results);preds.extend(results);rows.append(dict(protocol='oracle_restricted_training_and_test',cutoff=cutoff,features=feature,n=len(q),golfers=q.golfer.nunique(),mae=mean_absolute_error(q.actual,q.predicted)))
  print(cutoff,feature,len(q),round(rows[-1]['mae'],3),flush=True)
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(preds).to_csv(out/'restricted_predictions.csv',index=False);(out/'inner_selections.json').write_text(json.dumps(choices,indent=2))
(out/'config.json').write_text(json.dumps(dict(cutoffs=[70,100],comparison='strictly greater than',cohort='65 labeled club-covered shots',scheme='leave_golfer_out',features=sets,seed=cfg['seed'],limitations='Outcome-selected conditional cohorts; no automatic gate; exploratory, no independent new data.'),indent=2))
print(pd.DataFrame(rows).round(3).to_string(index=False))
