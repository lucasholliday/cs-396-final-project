"""Nested waveform ablations on the same club-covered cohort; outcome proxies, not contact labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss,mean_absolute_error,roc_auc_score,average_precision_score,balanced_accuracy_score,precision_score,recall_score,brier_score_loss
from benchmark import load_data,model,metrics
ROOT=Path(__file__).resolve().parents[1]

def classifier(name,y,seed):
 if len(np.unique(y))<2 or name=='prior':est=DummyClassifier(strategy='prior')
 elif name=='logistic':est=LogisticRegression(C=.3,max_iter=2000,class_weight='balanced',random_state=seed)
 else:est=ExtraTreesClassifier(n_estimators=150,min_samples_leaf=4,class_weight='balanced',random_state=seed,n_jobs=1)
 return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),est)

def probabilities(pipe,x):
 classes=list(pipe.classes_);p=pipe.predict_proba(x)
 return p[:,classes.index(1)] if 1 in classes else np.zeros(len(x))

def class_metrics(y,p):
 decision=p>=.5
 return dict(auroc=roc_auc_score(y,p) if len(set(y))>1 else np.nan,average_precision=average_precision_score(y,p),
  balanced_accuracy=balanced_accuracy_score(y,decision),precision=precision_score(y,decision,zero_division=0),recall=recall_score(y,decision,zero_division=0),brier=brier_score_loss(y,p),positive_n=int(sum(y)))

if __name__=='__main__':
 cfg=json.loads((ROOT/'configs/contact_quality.json').read_text());bc,d=load_data();f=pd.read_csv(ROOT/'data/processed/contact_features.csv')
 d=d.merge(f,on='shot_id',validate='one_to_one').merge(pd.read_csv(ROOT/'experiments/baseline_v1/splits.csv')[['shot_id','block','golfer_fold']],on='shot_id',validate='one_to_one')
 d['short_carry']=(d['Carry flat (yd)']<cfg['short_carry_threshold_yd']).astype(int)
 d['low_launch']=(d['Launch angle (deg)']<cfg['low_launch_threshold_deg']).astype(int)
 base=[c for c in d if c.startswith('club_gyro_')];gyro=[c for c in f if c.startswith('contact_gyro_')];clip=[c for c in f if c.startswith('contact_clip_')];contact=[c for c in f if c.startswith('contact_')]
 cw=bc['camera_features']+[c for c in d if c.startswith('watch_gyro_') or c.startswith('watch_accel_')]
 sets={'club_baseline':base,'club_plus_transients':base+gyro,'club_plus_clipping':base+clip,'club_all_contact':base+contact,'camera_watch':cw,'camera_watch_club_contact':cw+base+contact}
 out=ROOT/'experiments/contact_quality_v1';out.mkdir(exist_ok=True);preds=[];choices=[]
 (out/'config.json').write_text(json.dumps(dict(config=cfg,features=sets,n=len(d),selection='pooled inner MAE for carry; pooled inner log loss for outcome proxies',split_source='experiments/baseline_v1/splits.csv'),indent=2))
 d[['shot_id','golfer','session','short_carry','low_launch','Carry flat (yd)','Launch angle (deg)']].to_csv(out/'outcome_proxies.csv',index=False)
 for scheme,col in [('within_session_blocks','block'),('leave_golfer_out','golfer_fold')]:
  for task in ['Carry flat (yd)','short_carry','low_launch']:
   regression=task=='Carry flat (yd)';families=cfg['regression_models'] if regression else cfg['classification_models']
   for feature,cols in sets.items():
    for fold in sorted(d[col].unique()):
     tr=d[d[col]!=fold];te=d[d[col]==fold];scores={}
     for family in families:
      truth=[];pred=[]
      for ti,vi in GroupKFold(3).split(tr,groups=tr[col]):
       a=tr.iloc[ti];b=tr.iloc[vi];pipe=model(family,cfg['seed']) if regression else classifier(family,a[task],cfg['seed'])
       pipe.fit(a[cols],a[task]);truth.extend(b[task]);pred.extend(pipe.predict(b[cols]) if regression else probabilities(pipe,b[cols]))
      scores[family]=float(mean_absolute_error(truth,pred) if regression else log_loss(truth,np.clip(pred,1e-6,1-1e-6),labels=[0,1]))
     winner=min(families,key=lambda k:scores[k]);pipe=model(winner,cfg['seed']) if regression else classifier(winner,tr[task],cfg['seed']);pipe.fit(tr[cols],tr[task])
     result=pipe.predict(te[cols]) if regression else probabilities(pipe,te[cols])
     choices.append(dict(scheme=scheme,task=task,features=feature,fold=int(fold),model=winner,inner_scores=scores))
     for sid,g,y,p in zip(te.shot_id,te.golfer,te[task],result):preds.append(dict(scheme=scheme,task=task,features=feature,fold=int(fold),shot_id=sid,golfer=g,actual=float(y),predicted=float(p),model=winner))
    print(scheme,task,feature,flush=True)
 p=pd.DataFrame(preds);p.to_csv(out/'predictions.csv',index=False)
 results=[]
 for (s,t,f),g in p.groupby(['scheme','task','features']):results.append(dict(scheme=s,task=t,features=f,n=len(g),**(metrics(g.actual,g.predicted) if t=='Carry flat (yd)' else class_metrics(g.actual,g.predicted))))
 pd.DataFrame(results).to_csv(out/'metrics.csv',index=False);(out/'inner_selections.json').write_text(json.dumps(choices,indent=2))
 print(pd.DataFrame(results).to_string(index=False))
