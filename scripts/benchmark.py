"""Fixed initial baselines, shared persisted shot splits, no outcome-based feature selection."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def model(name,seed):
    estimator={'mean':DummyRegressor(strategy='mean'),'median':DummyRegressor(strategy='median'),
        'ridge10':Ridge(alpha=10),
        'extra_trees':ExtraTreesRegressor(n_estimators=150,min_samples_leaf=5,random_state=seed,n_jobs=1),
        'svr':TransformedTargetRegressor(regressor=SVR(C=1,epsilon=.1),transformer=StandardScaler())}[name]
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),estimator)


def metrics(y,p):
    return dict(mae=mean_absolute_error(y,p),rmse=np.sqrt(mean_squared_error(y,p)),
        r2=r2_score(y,p),median_ae=median_absolute_error(y,p))


def load_data():
    cfg=json.loads((ROOT/'configs/benchmark.json').read_text())
    m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
    labels=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').rename(columns={'Shot ID':'shot_id'})
    pose=pd.read_csv(ROOT/'data/processed/pose_features.csv')
    sensor=pd.read_csv(ROOT/'data/processed/sensor_features.csv')
    d=m[m.training_eligible].merge(labels[['shot_id']+cfg['targets']],on='shot_id',validate='one_to_one')
    d=d.merge(pose,on='shot_id',validate='one_to_one').merge(sensor,on='shot_id',validate='one_to_one')
    return cfg,d


if __name__=='__main__':
    cfg,d=load_data();seed=cfg['seed']
    out=ROOT/'experiments/baseline_v1';out.mkdir(parents=True,exist_ok=True)
    camera=cfg['camera_features'];watch=[c for c in d if c.startswith('watch_gyro_') or c.startswith('watch_accel_')]
    club=[c for c in d if c.startswith('club_gyro_')]
    sets={'camera':camera,'watch':watch,'club':club,'camera_watch':camera+watch,
          'camera_club':camera+club,'watch_club':watch+club,'all':camera+watch+club}
    assert not set(sum(sets.values(),[]))&set(cfg['targets'])
    # Contiguous within-session blocks; same golfers/sessions in training and test.
    d['block']=d.groupby('session').cumcount()
    d['block']=(d['block']/d.groupby('session').shot_id.transform('size')*5).astype(int).clip(upper=4)
    golfers=sorted(d.golfer.unique());d['golfer_fold']=d.golfer.map(dict(zip(golfers,range(len(golfers)))))
    d[['shot_id','golfer','session','block','golfer_fold']].to_csv(out/'splits.csv',index=False)
    (out/'config.json').write_text(json.dumps(dict(config=cfg,feature_sets=sets,cohorts={'all_labeled':'all 72 label rows','matched_club':'complete gyro feature availability'}),indent=2))
    predictions=[];results=[]
    for cohort,subset in [('all_labeled',d),('matched_club',d.dropna(subset=club))]:
        for scheme in cfg['schemes']:
            foldcol='block' if scheme=='within_session_blocks' else 'golfer_fold'
            for name,cols in sets.items():
                if cohort=='all_labeled' and 'club' in name or cohort=='all_labeled' and name=='all':continue
                for target in cfg['targets']:
                    for family in cfg['models']:
                        pred=np.full(len(subset),np.nan)
                        for fold in sorted(subset[foldcol].unique()):
                            test=(subset[foldcol]==fold).to_numpy();train=~test
                            if scheme=='leave_golfer_out':
                                assert not set(subset.loc[test,'golfer'])&set(subset.loc[train,'golfer'])
                            pipe=model(family,seed)
                            pipe.fit(subset.loc[train,cols],subset.loc[train,target])
                            pred[test]=pipe.predict(subset.loc[test,cols])
                        truth=subset[target].to_numpy();score=metrics(truth,pred)
                        fold_mae=[mean_absolute_error(truth[(subset[foldcol]==fold).to_numpy()],pred[(subset[foldcol]==fold).to_numpy()]) for fold in sorted(subset[foldcol].unique())]
                        results.append(dict(cohort=cohort,scheme=scheme,features=name,target=target,model=family,n=len(subset),
                            **score,fold_mae_min=min(fold_mae),fold_mae_max=max(fold_mae),fold_mae_std=float(np.std(fold_mae))))
                        for sid,golfer,fold,y,p in zip(subset.shot_id,subset.golfer,subset[foldcol],truth,pred):
                            predictions.append(dict(cohort=cohort,scheme=scheme,features=name,target=target,model=family,shot_id=sid,
                                golfer=golfer,fold=int(fold),actual=y,predicted=p,residual=y-p))
                print(cohort,scheme,name,'done',flush=True)
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    print('Saved',len(results),'evaluations')
