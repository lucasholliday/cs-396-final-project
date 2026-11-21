"""Nested golfer-held-out video/fusion study. No UI changes or target-derived inputs.
Fixed-grid outputs are retrospective; training-fold selection is the primary result.
"""
import json,os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from benchmark import load_data
ROOT=Path(__file__).resolve().parents[1];TARGET='Carry flat (yd)'
FAMILIES=['mean','median','ridge10','ridge100','svr1','svr10','trees','pls2']

def estimator(name):
 if name in ['mean','median']:e=DummyRegressor(strategy=name)
 elif name.startswith('ridge'):e=Ridge(alpha=float(name[5:]))
 elif name.startswith('svr'):e=TransformedTargetRegressor(regressor=SVR(C=float(name[3:]),epsilon=.1),transformer=StandardScaler())
 elif name=='trees':e=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=4,random_state=396,n_jobs=1)
 else:e=PLSRegression(n_components=2,scale=False,max_iter=500)
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),e)

def data():
 cfg,d=load_data();legacy=pd.read_csv(ROOT/'data/processed/video_launch_features.csv').drop(columns=['golfer','session']);new=pd.read_csv(ROOT/'data/processed/audited_launch_features.csv');new=new.rename(columns={c:'launch_'+c for c in new if c!='shot_id'});contact=pd.read_csv(ROOT/'data/processed/contact_features.csv')
 d=d.merge(legacy,on='shot_id',validate='one_to_one').merge(new,on='shot_id',validate='one_to_one').merge(contact,on='shot_id',how='left',validate='one_to_one')
 base=json.loads((ROOT/'experiments/baseline_v1/config.json').read_text())['feature_sets'];clean=['ball_angle_deg','ball_speed_px_s','ball_rise_px','ball_lateral_dx_px','ball_duration_s','abs_angle_deg'];early=['launch_'+c for c in ['vx_early','vy_early','ray_angle','early_fit_error','first_delay','tracked']];fixed=['launch_'+c for c in ['dx_020','rise_020','dx_040','rise_040','dx_060','rise_060','dx_080','rise_080','tracked']];body_norm=['launch_'+c for c in ['body_dx_020','body_rise_020','body_dx_040','body_rise_040','body_dx_060','body_rise_060','body_dx_080','body_rise_080','tracked']];persp=['launch_'+c for c in ['perspective_b','perspective_vx','perspective_vy','perspective_residual','tracked']]
 compact=fixed+['launch_vx_early','launch_vy_early','launch_perspective_b','launch_perspective_residual']
 sets={'legacy_unverified':clean+['mishit_push_flag','club_speed_px_s','smash_factor_proxy'],'clean_launch':clean,'early_velocity':early,'fixed_time':fixed,'body_normalized':body_norm,'perspective':persp,'video_combined':compact,'video_body':compact+base['camera'],'video_watch':compact+base['watch'],'video_club':compact+base['club'],'video_watch_club':compact+base['watch']+base['club'],'video_all_contact':compact+base['watch']+base['club']+[c for c in contact if c.startswith('contact_')]}
 return d,sets,base

def run_fold(df,sets,cohort,cutoff,golfer):
 tr=df[df.golfer!=golfer];te=df[df.golfer==golfer];splits=list(GroupKFold(min(3,tr.golfer.nunique())).split(tr,groups=tr.golfer));predictions=[];choices=[]
 for feature,cols in sets.items():
  scores={};held={}
  for family in FAMILIES:
   y=[];pr=[]
   for ti,vi in splits:
    a=tr.iloc[ti];b=tr.iloc[vi];pipe=estimator(family);pipe.fit(a[cols],a[TARGET]);y.extend(b[TARGET]);pr.extend(np.asarray(pipe.predict(b[cols])).ravel())
   scores[family]=float(mean_absolute_error(y,pr));pipe=estimator(family);pipe.fit(tr[cols],tr[TARGET]);values=np.asarray(pipe.predict(te[cols])).ravel();held[family]=values
   for sid,g,actual,p in zip(te.shot_id,te.golfer,te[TARGET],values):predictions.append(dict(cohort=cohort,cutoff=cutoff,features=feature,method=family,shot_id=sid,golfer=g,actual=float(actual),predicted=float(p)))
  winner=min(scores,key=scores.get);choices.append(dict(cohort=cohort,cutoff=cutoff,held_golfer=golfer,features=feature,selected=winner,inner_mae=scores))
  for sid,g,actual,p in zip(te.shot_id,te.golfer,te[TARGET],held[winner]):predictions.append(dict(cohort=cohort,cutoff=cutoff,features=feature,method='nested',shot_id=sid,golfer=g,actual=float(actual),predicted=float(p)))
 # Family + feature selection both happen inside this outer training fold.
 for selection,eligible in [('nested_video',['clean_launch','early_velocity','fixed_time','body_normalized','perspective','video_combined','video_body']),('nested_fusion',[k for k in sets if k!='legacy_unverified'])]:
  candidates=[c for c in choices if c['features'] in eligible];best=min(candidates,key=lambda c:min(c['inner_mae'].values()));family=best['selected'];feature=best['features']
  source=[p for p in predictions if p['features']==feature and p['method']==family];predictions.extend([{**p,'features':selection,'method':'nested'} for p in source]);choices.append(dict(cohort=cohort,cutoff=cutoff,held_golfer=golfer,features=selection,selected=family,selected_features=feature,inner_mae=best['inner_mae']))
 return predictions,choices

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--out',default='audited_launch_v1');args=parser.parse_args();d,sets,base=data();out=ROOT/'experiments'/args.out;out.mkdir(exist_ok=True)
 tasks=[]
 for cohort,df in [('all_labeled',d),('matched_club',d.dropna(subset=base['club']))]:
  for cutoff in [0,100]:
   subset=df if cutoff==0 else df[df[TARGET]>cutoff]
   for golfer in sorted(subset.golfer.unique()):tasks.append((subset,sets,cohort,cutoff,golfer))
 (out/'config.json').write_text(json.dumps(dict(features=sets,families=FAMILIES,cutoffs=[0,100],target=TARGET,seed=396,protocol='Outer leave-golfer-out; three-group inner CV; >100 is actual-outcome oracle filtering. All-shot mode includes every labeled shot and imputed missing tracks.',legacy_warning='Unverified blob speeds retained only as an audit comparator, excluded from nested feature selection.'),indent=2))
 result=Parallel(n_jobs=4,verbose=10)(delayed(run_fold)(*t) for t in tasks);preds=[p for r,_ in result for p in r];choices=[p for _,r in result for p in r];p=pd.DataFrame(preds);p['abs_error']=abs(p.actual-p.predicted);p.to_csv(out/'predictions.csv',index=False);(out/'inner_selections.json').write_text(json.dumps(choices,indent=2));rows=[]
 for key,q in p.groupby(['cohort','cutoff','features','method']):rows.append(dict(zip(['cohort','cutoff','features','method'],key),n=len(q),mae=mean_absolute_error(q.actual,q.predicted),rmse=float(np.sqrt(mean_squared_error(q.actual,q.predicted))),r2=r2_score(q.actual,q.predicted),median_ae=float(np.median(q.abs_error))))
 m=pd.DataFrame(rows);m.to_csv(out/'metrics.csv',index=False);p.groupby(['cohort','cutoff','features','method','golfer']).agg(n=('shot_id','size'),mae=('abs_error','mean'),bias=('predicted','mean'),actual_mean=('actual','mean')).reset_index().to_csv(out/'per_golfer.csv',index=False);print(m[m.method=='nested'].round(3).to_string(index=False))
