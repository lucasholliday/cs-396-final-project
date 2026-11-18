"""Conditional carry regression plus independently predicted low-carry gate.
Oracle qualifying subsets are explicitly distinguished from operational acceptance.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,log_loss,confusion_matrix
from benchmark import load_data,model,metrics
from contact_benchmark import classifier,probabilities,class_metrics
ROOT=Path(__file__).resolve().parents[1]

def fit_selected(train,cols,y,classify,cfg):
 families=cfg['classification_families'] if classify else cfg['regression_families'];groups=train.golfer.to_numpy();scores={}
 # Very small conditional cohorts fall back explicitly, not to test-set values.
 if len(train)<6 or len(set(groups))<2 or (classify and len(set(y))<2):
  winner='prior' if classify else 'mean';scores={'fallback_reason':'fewer than 6 rows, 2 golfers or 2 classes'}
 else:
  splits=list(GroupKFold(min(cfg['inner_folds'],len(set(groups)))).split(train,groups=groups))
  for family in families:
   truth=[];pred=[]
   for ti,vi in splits:
    pipe=classifier(family,y[ti],cfg['seed']) if classify else model(family,cfg['seed'])
    pipe.fit(train.iloc[ti][cols],y[ti]);truth.extend(y[vi]);pred.extend(probabilities(pipe,train.iloc[vi][cols]) if classify else pipe.predict(train.iloc[vi][cols]))
   scores[family]=float(log_loss(truth,np.clip(pred,1e-6,1-1e-6),labels=[0,1]) if classify else mean_absolute_error(truth,pred))
  winner=min(families,key=lambda f:scores[f])
 pipe=classifier(winner,y,cfg['seed']) if classify else model(winner,cfg['seed']);pipe.fit(train[cols],y)
 return pipe,winner,scores

if __name__=='__main__':
 cfg=json.loads((ROOT/'configs/carry_thresholds.json').read_text());bc,d=load_data();b=pd.read_csv(ROOT/'data/processed/reviewed_ball_features.csv');c=pd.read_csv(ROOT/'data/processed/contact_features.csv')
 cw=bc['camera_features']+[k for k in d if k.startswith('watch_gyro_') or k.startswith('watch_accel_')];ball=[k for k in b if k.startswith('reviewed_ball_')];contact=[k for k in c if k.startswith('contact_')]
 dc=d.merge(c,on='shot_id',validate='one_to_one');db=d.merge(b,on='shot_id',validate='one_to_one')
 cohorts=[('video',d,bc['camera_features']),('video_watch',d,cw),('video_watch_club_contact',dc,cw+[k for k in d if k.startswith('club_gyro_')]+contact),('reviewed_ball_assisted',db,ball)]
 out=ROOT/'experiments/carry_thresholds_v1';out.mkdir(exist_ok=True);records=[];choices=[];rows=[]
 (out/'config.json').write_text(json.dumps(dict(config=cfg,feature_sets={name:cols for name,_,cols in cohorts},cohort_sizes={name:len(sub) for name,sub,_ in cohorts}),indent=2))
 for name,sub,cols in cohorts:
  for threshold in cfg['thresholds_yd']:
   for golfer in sorted(sub.golfer.unique()):
    tr=sub[sub.golfer!=golfer];te=sub[sub.golfer==golfer];y=tr['Carry flat (yd)'].to_numpy();low=y<threshold
    gate,gn,gs=fit_selected(tr,cols,low.astype(int),True,cfg)
    allreg,an,ass=fit_selected(tr,cols,y,False,cfg)
    models={};selection={}
    for group,keep in [('low',low),('high',~low)]:
     if keep.sum()==0:
      models[group]=allreg;selection[group]={'family':an,'fallback':'no training rows; use all-shot model'}
     else:
      fitted,n,s=fit_selected(tr.loc[keep],cols,y[keep],False,cfg);models[group]=fitted;selection[group]={'family':n,'inner_mae':s,'training_n':int(keep.sum())}
    p=probabilities(gate,te[cols]);hi=models['high'].predict(te[cols]);lo=models['low'].predict(te[cols]);baseline=allreg.predict(te[cols]);accepted=p<cfg['gate_probability']
    choices.append(dict(features=name,threshold_yd=threshold,held_out_golfer=golfer,gate_family=gn,gate_inner_logloss=gs,all_family=an,conditional=selection))
    for sid,g,actual,prob,h,l,base,accept in zip(te.shot_id,te.golfer,te['Carry flat (yd)'],p,hi,lo,baseline,accepted):
     records.append(dict(features=name,threshold_yd=threshold,shot_id=sid,golfer=g,actual=float(actual),low_probability=float(prob),accepted=bool(accept),high_prediction=float(h),low_prediction=float(l),allshot_prediction=float(base),gated_prediction=float(h if accept else l),mixture_prediction=float((1-prob)*h+prob*l),training_high_mean=float(y[~low].mean()) if (~low).any() else float(y.mean())))
   print(name,threshold,'done',flush=True)
 p=pd.DataFrame(records);p.to_csv(out/'predictions.csv',index=False)
 for (name,threshold),g in p.groupby(['features','threshold_yd']):
  actual_low=g.actual<threshold;good=~actual_low;accepted=g.accepted
  tn,fp,fn,tp=confusion_matrix(actual_low,~accepted,labels=[False,True]).ravel()
  row=dict(features=name,threshold_yd=threshold,n=len(g),true_above_n=int(good.sum()),true_below_n=int(actual_low.sum()),
   oracle_above_mae=float(mean_absolute_error(g.loc[good,'actual'],g.loc[good,'high_prediction'])) if good.any() else None,
   oracle_above_constant_mae=float(mean_absolute_error(g.loc[good,'actual'],g.loc[good,'training_high_mean'])) if good.any() else None,
   allshot_on_above_mae=float(mean_absolute_error(g.loc[good,'actual'],g.loc[good,'allshot_prediction'])) if good.any() else None,
   allshot_mae=float(mean_absolute_error(g.actual,g.allshot_prediction)),gated_all_mae=float(mean_absolute_error(g.actual,g.gated_prediction)),mixture_all_mae=float(mean_absolute_error(g.actual,g.mixture_prediction)),
   accepted_n=int(accepted.sum()),acceptance_fraction=float(accepted.mean()),accepted_mae=float(mean_absolute_error(g.loc[accepted,'actual'],g.loc[accepted,'high_prediction'])) if accepted.any() else None,
   accepted_allshot_baseline_mae=float(mean_absolute_error(g.loc[accepted,'actual'],g.loc[accepted,'allshot_prediction'])) if accepted.any() else None,
   false_accepted_below_n=int(fn),false_rejected_above_n=int(fp),correctly_flagged_below_n=int(tp),correctly_accepted_above_n=int(tn),**class_metrics(actual_low.astype(int),g.low_probability))
  rows.append(row)
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);(out/'inner_selections.json').write_text(json.dumps(choices,indent=2))
 print(pd.DataFrame(rows).to_string(index=False))
