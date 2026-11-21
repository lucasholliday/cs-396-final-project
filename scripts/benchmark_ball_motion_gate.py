"""Paired track-termination and launch-time-window experiments on cached proposals."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ball_motion_gate import truncate_motion, truncate_stationary_tail
from benchmark_joint_landing import COORDS, VIDEO, estimator, miss
from carry_estimators import QualityRoutedRegressor
from launch_geometry_features import fit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/processed"
OUT = ROOT/"experiments/ball_motion_gate_v1"
RUNS = {
    "ungated_150ms": ("none", .150),
    "absolute_150ms": ("absolute", .150),
    "relative_150ms": ("relative", .150),
    "either_150ms": ("either", .150),
    "stall_150ms": ("stall", .150),
    "ungated_80ms": ("none", .080),
    "ungated_60ms": ("none", .060),
    "either_80ms": ("either", .080),
    "either_60ms": ("either", .060),
    "robust_linear_60ms": ("none", .060),
    "robust_linear_80ms": ("none", .080),
}


def fit_early_linear(track, seed, cutoff):
    """Estimate slopes from pairs of observations entirely inside the window."""
    if not seed:
        return {}
    observed = [p for p in track if 0 < p["playback_s"]-seed["playback_s"] <= cutoff]
    if len(observed) < 4:
        return {}
    time = np.array([p["playback_s"]-seed["playback_s"] for p in observed])
    if np.ptp(time) < .018:
        return {}
    xy = np.array([[p["x"], -p["y"]] for p in observed])
    i,j = np.triu_indices(len(time),1)
    use = time[j]-time[i]>=.012
    velocities = (xy[j[use]]-xy[i[use]])/(time[j[use]]-time[i[use]])[:,None]
    vx,vy = np.median(velocities,axis=0)
    v = np.array([vx,vy])
    error = np.median(np.linalg.norm(xy-time[:,None]*v-np.median(xy-time[:,None]*v,axis=0),axis=1))
    first = np.median(np.linalg.norm(velocities[(i[use]+j[use])<=len(time)-1],axis=1))
    last = np.median(np.linalg.norm(velocities[(i[use]+j[use])>=len(time)-1],axis=1))
    # These columns share a benchmark interface with projection fits; the
    # second descriptor here is an apparent log slowdown, not a depth estimate.
    return dict(proj_vy=vy/1920,proj_vx=vx/1080,
                proj_depth_rate=np.log((first+1)/(last+1)),proj_fit_rmse=error/1920)


def compare_geometry(features, labels, name):
    d = labels.merge(features[["shot_id"]+VIDEO], on="shot_id", how="left", validate="one_to_one")
    y = d[COORDS].to_numpy(float)
    y[:, [1, 3]] /= 3
    x = d[VIDEO].to_numpy(float)
    groups = d.golfer.to_numpy()
    out, selections = [], []
    for golfer in sorted(set(groups)):
        train = np.flatnonzero(groups != golfer)
        test = np.flatnonzero(groups == golfer)
        inner = [(train[a], train[b]) for a,b in GroupKFold(3).split(train,groups=groups[train])]
        columns = []
        for j, target in enumerate(COORDS):
            options = ["ridge10", "ridge60", "svr1", "svr10"]
            if j == 0:
                options.append("routed_svr10")

            def predict(name, a, b):
                if name == "routed_svr10":
                    return QualityRoutedRegressor("svr10","mean").fit(x[a,:2],y[a,j]).predict(x[b,:2])
                return estimator(name).fit(x[a],y[a,j:j+1]).predict(x[b]).ravel()

            scores = {option: float(np.abs(np.concatenate([
                predict(option,a,b)-y[b,j] for a,b in inner
            ])).mean()) for option in options}
            best = min(scores, key=scores.get)
            columns.append(predict(best,train,test))
            selections.append(dict(variant=name,golfer=golfer,target=target,selected=best,scores=scores))
        p = np.column_stack(columns)
        errors = miss(y[test],p)
        for i, idx in enumerate(test):
            out.append(dict(variant=name,shot_id=d.shot_id.iloc[idx],golfer=golfer,
                            has_launch_fit=bool(np.isfinite(x[idx,:2]).all()),
                            actual_carry_yd=y[idx,0],predicted_carry_yd=p[i,0],
                            carry_ae_yd=abs(y[idx,0]-p[i,0]),
                            landing_2d_yd=errors[i,0],resting_2d_yd=errors[i,1]))
    return out, selections


def main():
    OUT.mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT/"data/manifests/shot_manifest.csv")
    labels = pd.read_csv(ROOT/"data/manifests/trackman_labels.csv").rename(columns={"Shot ID":"shot_id"})
    labels = manifest.loc[manifest.training_eligible,["shot_id","golfer"]].merge(
        labels[["shot_id"]+COORDS],on="shot_id",validate="one_to_one").dropna(subset=COORDS)
    track_stats, refs, predictions, selections = [], [], [], []
    ref = pd.read_csv(ROOT/"annotations/contact_ball_points.csv")
    tables = {name: [] for name in RUNS}
    for sid in manifest.shot_id:
        track = json.loads((DATA/f"ball_extension_wide35/{sid}.json").read_text())
        points = track["observations"]
        seed = track["seed"]
        start = seed["playback_s"] if seed else None
        gated = {}
        for policy in ("none", "absolute", "relative", "either", "stall", "display"):
            keep, stop = truncate_stationary_tail(points) if policy=="display" else truncate_motion(points,policy,start)
            gated[policy] = keep
            track_stats.append(dict(shot_id=sid,policy=policy,before=len(points),after=len(keep),
                                    **(stop or {})))
            for point in ref[ref.shot_id.eq(sid)].itertuples():
                nearest = min(keep,key=lambda p: abs(p["playback_s"]-point.playback_s),default=None)
                error = float("inf")
                if nearest and abs(nearest["playback_s"]-point.playback_s)<=.006:
                    error = float(np.hypot(nearest["x"]-point.x,nearest["y"]-point.y))
                refs.append(dict(shot_id=sid,policy=policy,panel=point.panel,error_px=error,
                                 within16=error<=16,within32=error<=32))
        for name,(policy,cutoff) in RUNS.items():
            extract = fit_early_linear if name.startswith("robust_linear") else fit
            tables[name].append(dict(shot_id=sid,**extract(gated[policy],seed,cutoff)))
    features = {"previous_cached_geometry": pd.read_csv(DATA/"launch_geometry_features.csv")}
    features.update({k:pd.DataFrame(v) for k,v in tables.items()})
    for name,data in features.items():
        p,s = compare_geometry(data,labels,name)
        predictions.extend(p)
        selections.extend(s)
    p = pd.DataFrame(predictions)
    assert not p.duplicated(["variant","shot_id"]).any()
    assert p.groupby("variant").size().eq(72).all()
    p.to_csv(OUT/"predictions.csv",index=False)
    metrics = p.groupby("variant").agg(
        n=("shot_id","size"),launch_fits=("has_launch_fit","sum"),carry_mae=("carry_ae_yd","mean"),
        landing_2d_yd=("landing_2d_yd","mean"),resting_2d_yd=("resting_2d_yd","mean"))
    metrics.to_csv(OUT/"metrics.csv")
    pd.DataFrame(track_stats).to_csv(OUT/"track_stops.csv",index=False)
    pd.DataFrame(refs).to_csv(OUT/"reference_errors.csv",index=False)
    (OUT/"protocol.json").write_text(json.dumps(dict(input_tracks="ball_extension_wide35",
        time_units="source playback seconds relative to candidate launch seed",
        variants=RUNS,features=VIDEO,inner_selections=selections),indent=2))
    print(metrics.round(4))


if __name__ == "__main__":
    main()
