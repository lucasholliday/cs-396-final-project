"""Nested model-family selection for carry; external folds never choose their own model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from benchmark import load_data,model,metrics

ROOT=Path(__file__).resolve().parents[1]
cfg,d=load_data();out=ROOT/'experiments/nested_carry_v1';out.mkdir(parents=True,exist_ok=True)
base=ROOT/'experiments/baseline_v1'
sets=json.loads((base/'config.json').read_text())['feature_sets']
d=d.merge(pd.read_csv(base/'splits.csv')[['shot_id','block','golfer_fold']],on='shot_id',validate='one_to_one')
families=['mean','ridge10','extra_trees','svr'];scores=[];predictions=[];choices=[]
for cohort,subset in [('all_labeled',d),('matched_club',d.dropna(subset=sets['club']))]:
    for scheme in cfg['schemes']:
        outer='block' if scheme=='within_session_blocks' else 'golfer_fold'
        for feature,columns in sets.items():
            if cohort=='all_labeled' and (feature=='all' or 'club' in feature):continue
            for fold in sorted(subset[outer].unique()):
                train=subset[subset[outer]!=fold];test=subset[subset[outer]==fold]
                groups=train[outer].to_numpy()
                inner=list(GroupKFold(n_splits=3).split(train,groups=groups))
                family_scores={}
                for family in families:
                    truth=[];pred=[]
                    for ti,vi in inner:
                        pipe=model(family,cfg['seed']);tr=train.iloc[ti];va=train.iloc[vi]
                        pipe.fit(tr[columns],tr['Carry flat (yd)'])
                        truth.extend(va['Carry flat (yd)']);pred.extend(pipe.predict(va[columns]))
                    family_scores[family]=float(mean_absolute_error(truth,pred))
                winner=min(families,key=lambda f:family_scores[f])
                pipe=model(winner,cfg['seed']);pipe.fit(train[columns],train['Carry flat (yd)'])
                pred=pipe.predict(test[columns])
                choices.append(dict(cohort=cohort,scheme=scheme,features=feature,fold=int(fold),selected=winner,inner_mae=family_scores))
                for sid,golfer,y,p in zip(test.shot_id,test.golfer,test['Carry flat (yd)'],pred):
                    predictions.append(dict(cohort=cohort,scheme=scheme,features=feature,fold=int(fold),shot_id=sid,golfer=golfer,actual=y,predicted=p,residual=y-p,model=winner))
            print(cohort,scheme,feature,'done',flush=True)
p=pd.DataFrame(predictions)
for (cohort,scheme,features),g in p.groupby(['cohort','scheme','features']):
    scores.append(dict(cohort=cohort,scheme=scheme,features=features,n=len(g),**metrics(g.actual,g.predicted)))
p.to_csv(out/'predictions.csv',index=False)
pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False)
(out/'inner_selections.json').write_text(json.dumps(choices,indent=2))
(out/'config.json').write_text(json.dumps(dict(seed=cfg['seed'],families=families,inner_folds=3,selection='pooled inner MAE',outer_split_source='experiments/baseline_v1/splits.csv',target='Carry flat (yd)'),indent=2))
print('Saved',len(scores),'nested carry comparisons')
