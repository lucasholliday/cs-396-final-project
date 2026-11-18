"""Timing sensitivity and inter-frame flight features that do not need departure time.
Retrospective development study on existing approximate reviewed points.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,log_loss
from benchmark import load_data,model,metrics
from contact_benchmark import classifier,probabilities,class_metrics
ROOT=Path(__file__).resolve().parents[1]

def slope_features(points):
    points=points.sort_values('playback_s');t=points.playback_s.to_numpy();x=points.x.to_numpy()/1080;y=points.y.to_numpy()/1920
    if len(t)<2 or t[-1]-t[0]<.02:return None
    # Pairwise median reduces dependence on a single approximate center.
    vx=[];vy=[]
    for i in range(len(t)):
        for j in range(i+1,len(t)):
            if t[j]-t[i]>=.02:vx.append((x[j]-x[i])/(t[j]-t[i]));vy.append(-(y[j]-y[i])/(t[j]-t[i]))
    return dict(flight_vx=float(np.median(vx)),flight_vy=float(np.median(vy)),flight_span_s=float(t[-1]-t[0]))

if __name__=='__main__':
    out=ROOT/'experiments/ball_timing_v1';out.mkdir(exist_ok=True)
    refs=pd.read_csv(ROOT/'annotations/contact_ball_points.csv');rows=[]
    for sid,g in refs.groupby('shot_id'):
        f=slope_features(g)
        if f:rows.append(dict(shot_id=sid,**f))
    features=pd.DataFrame(rows);features.to_csv(out/'flight_features.csv',index=False)
    cfg,d=load_data();early=pd.read_csv(ROOT/'data/processed/reviewed_ball_features.csv');d=d.merge(early,on='shot_id',validate='one_to_one').merge(features,on='shot_id',validate='one_to_one').merge(pd.read_csv(ROOT/'experiments/baseline_v1/splits.csv')[['shot_id','golfer_fold']],on='shot_id',validate='one_to_one');d['short100']=(d['Carry flat (yd)']<100).astype(int)
    early_cols=[c for c in early if c.startswith('reviewed_ball_')];slope_cols=['flight_vx','flight_vy','flight_span_s'];sets={'first_point_departure':early_cols,'interframe_slope':slope_cols,'combined':early_cols+slope_cols}
    (out/'config.json').write_text(json.dumps(dict(seed=396,n=len(d),golfers=d.groupby('golfer').size().to_dict(),features=sets,tasks=['Carry flat (yd)','Launch angle (deg)','short100'],scheme='leave_golfer_out',inner='3 group folds; select pooled MAE or log loss',limitations='AI-reviewed sparse point development cohort, at least two points; not a blind or autonomous evaluation. All three feature sets evaluated on identical shots. No metric speed interpretation.'),indent=2))
    perturb=[]
    for r in early.itertuples():
        for shift in [-.02,.02]:
            elapsed=r.reviewed_ball_first_elapsed_s-shift
            perturb.append(dict(shot_id=r.shot_id,departure_shift_s=shift,original_elapsed_s=r.reviewed_ball_first_elapsed_s,perturbed_elapsed_s=elapsed,rate_multiplier=r.reviewed_ball_first_elapsed_s/elapsed if elapsed>0 else None))
    pd.DataFrame(perturb).to_csv(out/'departure_sensitivity.csv',index=False)
    preds=[];choices=[]
    for task in ['Carry flat (yd)','Launch angle (deg)','short100']:
        regression=task!='short100';families=['mean','ridge10','extra_trees','svr'] if regression else ['prior','logistic','extra_trees']
        for feature,cols in sets.items():
            for fold in sorted(d.golfer_fold.unique()):
                tr=d[d.golfer_fold!=fold];te=d[d.golfer_fold==fold];scores={}
                for family in families:
                    yy=[];pp=[]
                    for ti,vi in GroupKFold(3).split(tr,groups=tr.golfer_fold):
                        a=tr.iloc[ti];b=tr.iloc[vi];pipe=model(family,396) if regression else classifier(family,a[task],396);pipe.fit(a[cols],a[task]);yy.extend(b[task]);pp.extend(pipe.predict(b[cols]) if regression else probabilities(pipe,b[cols]))
                    scores[family]=float(mean_absolute_error(yy,pp) if regression else log_loss(yy,np.clip(pp,1e-6,1-1e-6),labels=[0,1]))
                winner=min(families,key=lambda k:scores[k]);pipe=model(winner,396) if regression else classifier(winner,tr[task],396);pipe.fit(tr[cols],tr[task]);pr=pipe.predict(te[cols]) if regression else probabilities(pipe,te[cols]);choices.append(dict(task=task,features=feature,fold=int(fold),model=winner,inner_scores=scores))
                for sid,g,y,p in zip(te.shot_id,te.golfer,te[task],pr):preds.append(dict(task=task,features=feature,shot_id=sid,golfer=g,fold=int(fold),actual=float(y),predicted=float(p),model=winner))
            print(task,feature,'done',flush=True)
    p=pd.DataFrame(preds);p.to_csv(out/'predictions.csv',index=False);scores=[]
    for (task,feature),g in p.groupby(['task','features']):scores.append(dict(task=task,features=feature,n=len(g),**(metrics(g.actual,g.predicted) if task!='short100' else class_metrics(g.actual,g.predicted))))
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False);(out/'inner_selections.json').write_text(json.dumps(choices,indent=2));print(pd.DataFrame(scores).to_string(index=False))
