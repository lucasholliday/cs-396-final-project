"""Cross-fit TrackMan label predictions and evaluate them on unseen golfers."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_joint_landing import COORDS, AUX, VIDEO, GYRO
from carry_estimators import QualityRoutedRegressor as Routed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/label_stack_v1"
OUT.mkdir(exist_ok=True)
targets = COORDS + AUX
d = pd.read_csv(ROOT/"data/manifests/trackman_labels.csv").dropna(subset=targets).rename(columns={"Shot ID":"shot_id"})
for name in ["launch_geometry_features", "club_fusion_features"]:
    features = pd.read_csv(ROOT/f"data/processed/{name}.csv")
    if "has_club_imu" in features:
        features.loc[~features.has_club_imu, GYRO] = np.nan
    d = d.merge(features, on="shot_id", how="left", validate="one_to_one")
d = d.reset_index(drop=True)
y = d[targets].to_numpy(float)
y[:, [1, 3]] /= 3  # Endpoint lateral coordinates in yards.
groups = d.Golfer.to_numpy()
golfers = tuple(sorted(set(groups)))
compact = VIDEO + ["proj_angle"]
wide = compact + ["proj_shift","fit_dx_030","fit_rise_030","fit_dx_100","fit_rise_100"]
# Fixed first-stage recipes. Use each target's own sensors and features.
recipes = [(VIDEO[:2], "route"),(wide+GYRO,"svr1"),(VIDEO[:2],"svr1"),(wide+GYRO,"svr1"),
           (VIDEO[:2],"svr1"),(VIDEO[:2],"svr1"),(wide,"ridge10"),(VIDEO[:2],"svr1"),
           (VIDEO[:2],"svr1"),(compact,"svr10"),(wide+GYRO,"svr1")]

@lru_cache(None)
def base(training):
    train = np.isin(groups, training)
    predictions = []
    for j,(cols,name) in enumerate(recipes):
        x = d[cols].to_numpy(float)
        model = Routed("svr10","mean") if name=="route" else Routed.branch(name)
        model.fit(x[train], y[train,j])
        predictions.append(model.predict(x))
    return np.column_stack(predictions)

def ridge(x,y,xnew):
    return make_pipeline(StandardScaler(),Ridge(alpha=20)).fit(x,y).predict(xnew)

def methods(training, test):
    train = np.isin(groups,training)
    oof = np.full(y.shape,np.nan)
    for val in training:
        inner = tuple(g for g in training if g!=val)
        oof[groups==val] = base(inner)[groups==val]
    a,b=oof[train],base(training)[test]
    assert np.isfinite(a).all()
    results={"independent":b,"train_mean":np.tile(y[train].mean(axis=0),(len(b),1))}
    for name in ("self_calibration","all_labels"):
        pred=b.copy()
        for j in range(y.shape[1]):
            subset=[j] if name=="self_calibration" else list(range(y.shape[1]))
            pred[:,j]+=ridge(a[:,subset],y[train,j]-a[:,j],b[:,subset])
        results[name]=pred
    # Preserve airborne predictions and learn actual forward/lateral rollout from
    # estimated labels. This prevents total-distance noise feeding back into carry.
    for name in ("mean_roll","coupled_roll"):
        pred=b.copy()
        for landing,rest in [(0,2),(1,3)]:
            delta=y[train,rest]-y[train,landing]
            run=delta.mean() if name=="mean_roll" else ridge(a,delta,b)
            pred[:,rest]=b[:,landing]+(np.maximum(0,run) if rest==2 else run)
        results[name]=pred
    for name,idx in [("carry_only_roll",[0]),("launch_roll",[0,4,5,7,8,9])]:
        pred=b.copy()
        run=ridge(a[:,idx],y[train,2]-y[train,0],b[:,idx])
        pred[:,2]=b[:,0]+np.maximum(0,run)
        results[name]=pred
    clipped=b.copy()
    clipped[:,2]=np.maximum(b[:,2],b[:,0])
    results["clipped_only"]=clipped
    return test,results

selections={}
parts=[]
for held in golfers:
    training=tuple(g for g in golfers if g!=held)
    test=groups==held
    _,results=methods(training,test)
    # Choose the fraction of learned rollout for each resting coordinate on inner
    # golfers only. Each candidate's meta-training inputs are cross-fitted again.
    weights=np.array([0,.5,1])
    errors=np.zeros((3,3))
    for val in training:
        ii=groups==val
        _,valid=methods(tuple(g for g in training if g!=val),ii)
        direct,delta=valid["independent"],valid["coupled_roll"]
        for j,w in enumerate(weights):
            for k,vw in enumerate(weights):
                estimate=direct[:,2:4]+(delta[:,2:4]-direct[:,2:4])*[w,vw]
                estimate[:,0]=np.maximum(estimate[:,0],direct[:,0])
                errors[j,k]+=np.linalg.norm(estimate-y[ii,2:4],axis=1).sum()
    pair=weights[np.array(np.unravel_index(errors.argmin(),errors.shape))]
    direct=results["independent"]
    blend=direct.copy()
    blend[:,2:4]+=(results["coupled_roll"][:,2:4]-direct[:,2:4])*pair
    blend[:,2]=np.maximum(blend[:,2],blend[:,0])
    results["inner_selected_roll"]=blend
    selections[held]=pair.tolist()
    for method,p in results.items():
        table=d.loc[test,["shot_id","Golfer"]].copy()
        table["method"]=method
        for j,col in enumerate(targets):
            table[col]=p[:,j]
        parts.append(table)
pred=pd.concat(parts,ignore_index=True)
# Compare the exact viewer outputs too, including its illustrated tangent rollout.
viewer=pd.read_csv(OUT/"pre_rollout_viewer.csv").dropna(subset=["best_prediction","best_total_yd"])
v=viewer.set_index("id").reindex(d.shot_id)
carry=v.best_prediction
side=v.best_carry_side_ft/3
slope=2*side/carry-np.tan(np.deg2rad(v.best_launch_direction))
unit=1/np.sqrt(1+slope**2)
run=v.best_total_yd-carry
display=d[["shot_id","Golfer"]].copy()
display["method"]="viewer"
for col,values in zip(targets,[carry,side,v.best_total_yd,side+run*slope*unit,v.best_ball_speed,v.best_launch_angle,v.best_launch_direction,v.best_height_ft,v.best_landing_angle,v.best_hang_time,v.best_curve_ft]):
    display[col]=values.to_numpy()
pred=pd.concat([pred,display.dropna(subset=[COORDS[0]])],ignore_index=True)
# Historical per-coordinate reference uses a directly predicted total-side label.
hist=pd.read_csv(ROOT/"experiments/joint_landing_v2/predictions.csv").query("cohort=='all' and features=='video' and method=='independent_per_target'")
h=hist[["shot_id","golfer"]].rename(columns={"golfer":"Golfer"}).copy()
h["method"]="historical"
for j,col in enumerate(COORDS): h[col]=hist[f"predicted_{j}"].to_numpy()
pred=pd.concat([pred,h],ignore_index=True)
(OUT/"selected_roll_weights.json").write_text(json.dumps(selections,indent=2))
actual=pd.DataFrame(y,columns=targets,index=d.shot_id).loc[pred.shot_id].to_numpy()
err=pred[targets].to_numpy()-actual
for j,col in enumerate(targets): pred["MAE "+(col.replace("ft; R+","yd; R+") if j in (1,3) else col)]=abs(err[:,j])
pred["landing"]=np.hypot(err[:,0],err[:,1])
pred["resting"]=np.hypot(err[:,2],err[:,3])
# Exact graphical resting depth differs from reported total when rollout is diagonal.
iv=pred.method=="viewer"
pz=(carry+run*unit).dropna()
pred["rendered_resting"]=np.nan
pred.loc[iv,"rendered_resting"]=np.hypot(pz.loc[pred.loc[iv,"shot_id"]].to_numpy()-actual[iv,2],err[iv,3])
pred["negative_roll"]=pred[COORDS[2]]<pred[COORDS[0]]
pred.rename(columns={targets[j]:targets[j].replace("ft; R+","yd; R+") for j in (1,3)}).to_csv(OUT/"predictions.csv",index=False)
metrics=["landing","resting","negative_roll"]+[c for c in pred if c.startswith("MAE ")]
tables=[]
for cohort,df in [("all",pred),("viewer_shots",pred[pred.shot_id.isin(viewer.id)])]:
    for by in [["method"],["method","Golfer"]]:
        t=df.groupby(by)[metrics].mean()
        t["n"]=df.groupby(by).size()
        tables.append(t.reset_index().assign(cohort=cohort))
summary=pd.concat(tables,ignore_index=True)
summary.to_csv(OUT/"metrics.csv",index=False)
print(summary[summary.Golfer.isna()][["cohort","method","n","landing","resting","MAE "+COORDS[0],"MAE "+COORDS[2],"negative_roll"]].round(2).to_string(index=False))
