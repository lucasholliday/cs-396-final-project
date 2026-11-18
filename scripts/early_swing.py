"""Conservative early-window ablation; not validated impact timing or practice-swing inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from benchmark import load_data,model,metrics
ROOT=Path(__file__).resolve().parents[1]

def pose_features(arr,fps):
    f={}
    for name,ids in [('head',[0,7,8]),('shoulders',[11,12]),('hips',[23,24])]:
        valid=(arr[:,ids,3]>=.6).all(axis=1)
        xy=arr[:,ids,:2].mean(axis=1);good=xy[valid]
        for axis in range(2):f[f'early_{name}_range_{axis}']=float(np.ptp(good[:,axis])) if len(good)>1 else np.nan
        speed=np.linalg.norm(np.diff(xy,axis=0),axis=1)*fps
        speed=speed[valid[:-1]&valid[1:]]
        f[f'early_{name}_speed_p95']=float(np.quantile(speed,.95)) if len(speed) else np.nan
    return f

def extract():
    m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');rows=[];cache={}
    for r in m.itertuples():
        p=json.loads((ROOT/f'data/processed/pose/{r.shot_id}.json').read_text());a=np.array(p['landmarks'],dtype=float)
        relative=p['clip_start_playback_s']+np.arange(len(a))/p['fps']-r.video_peak_candidate_s
        # Fixed conservative cutoff, before typical observed departure (~peak-0.5s).
        keep=(relative>=-2.0)&(relative<=-.8)
        f=dict(shot_id=r.shot_id,**pose_features(a[keep],p['fps']))
        if r.watch_candidate not in cache:cache[r.watch_candidate]=pd.read_csv(ROOT/r.watch_candidate)
        w=cache[r.watch_candidate];w=w[(w.seconds_elapsed>=r.watch_peak_candidate_s-2)&(w.seconds_elapsed<=r.watch_peak_candidate_s-.8)]
        for name,cols in [('gyro',['rotationRateX','rotationRateY','rotationRateZ']),('accel',['accelerationX','accelerationY','accelerationZ'])]:
            v=np.linalg.norm(w[cols],axis=1)
            f[f'early_watch_{name}_p95']=np.quantile(v,.95)
            f[f'early_watch_{name}_rms']=np.sqrt(np.mean(v*v))
        rows.append(f)
    d=pd.DataFrame(rows);d.to_csv(ROOT/'data/processed/early_features.csv',index=False);return d

if __name__=='__main__':
    early=extract();cfg,d=load_data();d=d.merge(early,on='shot_id',validate='one_to_one')
    d=d.merge(pd.read_csv(ROOT/'experiments/baseline_v1/splits.csv')[['shot_id','block','golfer_fold']],on='shot_id',validate='one_to_one')
    camera=[c for c in early if c.startswith('early_') and 'watch' not in c];watch=[c for c in early if c.startswith('early_watch')]
    sets={'early_camera':camera,'early_watch':watch,'early_camera_watch':camera+watch}
    out=ROOT/'experiments/early_swing_v1';out.mkdir(parents=True,exist_ok=True)
    families=['mean','ridge10','extra_trees','svr'];predictions=[];choices=[]
    for scheme,outer in [('within_session_blocks','block'),('leave_golfer_out','golfer_fold')]:
        for feature,cols in sets.items():
            for fold in sorted(d[outer].unique()):
                train=d[d[outer]!=fold];test=d[d[outer]==fold];scores={}
                for family in families:
                    truth=[];pred=[]
                    for ti,vi in GroupKFold(3).split(train,groups=train[outer]):
                        tr=train.iloc[ti];va=train.iloc[vi];pipe=model(family,cfg['seed'])
                        pipe.fit(tr[cols],tr['Carry flat (yd)']);truth.extend(va['Carry flat (yd)']);pred.extend(pipe.predict(va[cols]))
                    scores[family]=float(mean_absolute_error(truth,pred))
                winner=min(families,key=lambda f:scores[f]);pipe=model(winner,cfg['seed']);pipe.fit(train[cols],train['Carry flat (yd)'])
                pred=pipe.predict(test[cols]);choices.append(dict(scheme=scheme,features=feature,fold=int(fold),selected=winner,inner_mae=scores))
                for sid,g,y,p in zip(test.shot_id,test.golfer,test['Carry flat (yd)'],pred):
                    predictions.append(dict(scheme=scheme,features=feature,fold=int(fold),shot_id=sid,golfer=g,actual=y,predicted=p,model=winner))
            print(scheme,feature,flush=True)
    p=pd.DataFrame(predictions);p.to_csv(out/'predictions.csv',index=False)
    rows=[dict(scheme=s,features=f,n=len(g),**metrics(g.actual,g.predicted)) for (s,f),g in p.groupby(['scheme','features'])]
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'inner_selections.json').write_text(json.dumps(choices,indent=2))
    (out/'config.json').write_text(json.dumps(dict(seed=cfg['seed'],feature_sets=sets,window_relative_to_watch_peak_s=[-2,-.8],families=families,inner_folds=3,limitations='Offline watch-assisted early window; impact exclusion not individually validated. No actual practice swings. No outcome-based window selection.'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
