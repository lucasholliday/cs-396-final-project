"""Paired golfer-held-out evaluation of joint landing and rollout regressions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from benchmark import load_data
from carry_estimators import QualityRoutedRegressor

ROOT = Path(__file__).resolve().parents[1]
COORDS = ["Carry flat (yd)", "Carry side (ft; R+)", "Total (yd)", "Total side (ft; R+)"]
AUX = ["Ball speed (mph)", "Launch angle (deg)", "Launch direction (deg; R+)",
       "Height (ft)", "Landing angle (deg)", "Hang time (s)", "Curve signed (ft; R+)"]
VIDEO = ["proj_vy", "proj_depth_rate", "proj_vx", "proj_fit_rmse"]
GYRO = ["roll_ratio_mean", "path_tilt_ratio", "roll_closure_ratio"]
CORE_FAMILIES = {
    "independent": ["ridge10", "ridge60", "svr1", "svr10"],
    "joint": ["pls1", "pls2", "pls3", "elastic10", "elastic30"],
}


def estimator(name):
    if name.startswith("ridge"):
        regressor = Ridge(alpha=int(name[5:]))
    elif name.startswith("svr"):
        regressor = MultiOutputRegressor(SVR(C=int(name[3:]), epsilon=.1))
    elif name.startswith("pls"):
        regressor = PLSRegression(n_components=int(name[3:]), scale=False)
    else:
        regressor = MultiTaskElasticNet(alpha=int(name[7:])/100, l1_ratio=.65, max_iter=5000)
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler()),
    )


def outcome(y, method):
    target = y.copy()
    if method.startswith("structured"):
        target[:, 2:4] -= target[:, :2]
    return target


def restore(p, method):
    p = np.asarray(p)[:, :4].copy()
    if method.startswith("structured"):
        p[:, 2:4] += p[:, :2]
    return p


def miss(y, p):
    return np.linalg.norm((y[:, :4]-p).reshape(-1, 2, 2), axis=2)


def fixed_time_features(d, variants):
    for variant in variants:
        folder = "ball_launch_tracks" if variant == "baseline" else f"ball_extension_{variant}"
        rows = []
        for p in (ROOT / "data/processed" / folder).glob("*.json"):
            track = json.loads(p.read_text())
            seed = track["seed"]
            points = track["observations"]
            row = {"shot_id": p.stem}
            if seed and points:
                t = np.array([o["playback_s"]-seed["playback_s"] for o in points])
                for horizon in [.20, .35, .50, .65, .75]:
                    k = np.searchsorted(t, horizon)
                    if 0 < k < len(t) and 0 < t[k]-t[k-1] <= .025:
                        for axis in ("x","y"):
                            row[f"{variant}_{axis}_{int(horizon*1000)}"] = (
                                np.interp(horizon,t[k-1:k+1],[p[axis] for p in points[k-1:k+1]])-seed[axis])
            rows.append(row)
        table = pd.DataFrame(rows)
        assert not table.empty
        d = d.merge(table, on="shot_id", how="left", validate="one_to_one")
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--out", default="joint_landing_v1")
    args = parser.parse_args()
    config, d = load_data()
    labels = pd.read_csv(ROOT / "data/manifests/trackman_labels.csv").rename(columns={"Shot ID":"shot_id"})
    extra = [c for c in COORDS+AUX if c not in d and c in labels]
    d = d.merge(labels[["shot_id"]+extra], on="shot_id", validate="one_to_one")
    for filename in ("launch_geometry_features.csv", "club_fusion_features.csv"):
        f = pd.read_csv(ROOT / "data/processed" / filename)
        use = [c for c in VIDEO+GYRO if c in f]
        if "has_club_imu" in f:
            f.loc[~f["has_club_imu"], use] = np.nan
        d = d.merge(f[["shot_id"]+use], on="shot_id", how="left", validate="one_to_one")
    # Auxiliary outcomes must be observed in training. Keep the same cohort for all methods.
    targets = COORDS + [c for c in AUX if c in d and d[c].notna().all()]
    d = d.dropna(subset=COORDS)
    assert d[targets].notna().all().all()
    d = fixed_time_features(d, args.variants)
    rows, selections = [], []
    watch = ["watch_gyro_p95","watch_gyro_rms","watch_accel_p95"]
    video_sets = {"video": VIDEO}
    for variant in args.variants:
        video_sets[f"video_{variant}"] = VIDEO + [c for c in d if c.startswith(variant+"_")]
    for cohort, data in [("all",d),("club_matched",d.dropna(subset=GYRO))]:
        sets = {"pose_watch": config["camera_features"]+watch,
                **video_sets, "video_watch": VIDEO+watch}
        if cohort == "club_matched":
            sets.update(video_gyro=VIDEO+GYRO, video_watch_gyro=VIDEO+watch+GYRO)
        y = data[targets].to_numpy(float)
        y[:, [1,3]] /= 3
        groups = data.golfer.to_numpy()
        for cols in sets.values():
            assert not set(cols) & set(targets)
        for held in sorted(set(groups)):
            train = np.flatnonzero(groups != held)
            test = np.flatnonzero(groups == held)
            splits = list(GroupKFold(3).split(train, groups=groups[train]))
            for feature, cols in sets.items():
                x = data[cols].to_numpy(float)
                recipes = {**CORE_FAMILIES, "structured_joint":CORE_FAMILIES["joint"],
                           "structured_independent":CORE_FAMILIES["independent"]}
                predictions = {"mean": np.tile(y[train,:4].mean(axis=0),(len(test),1))}
                for method, families in recipes.items():
                    yt = outcome(y,method)
                    if "independent" in method:
                        yt = yt[:,:4]
                    scores = {}
                    for family in families:
                        err = []
                        for a,b in splits:
                            a,b=train[a],train[b]
                            p=restore(estimator(family).fit(x[a],yt[a]).predict(x[b]),method)
                            err.extend(miss(y[b],p))
                        scores[family]=float(np.mean(err))
                    best=min(scores,key=scores.get)
                    predictions[method]=restore(estimator(best).fit(x[train],yt[train]).predict(x[test]),method)
                    selections.append(dict(cohort=cohort,golfer=held,features=feature,method=method,
                                           selected=best,inner_2d_yards=scores))
                # Let each scalar choose its own model inside the same inner folds.
                # For carry, compare the existing launch-availability route as well.
                scalar = []
                scalar_selections = {}
                for j, target in enumerate(COORDS):
                    options = list(CORE_FAMILIES["independent"])
                    use_route = j == 0 and cols[:2] == VIDEO[:2]
                    if use_route:
                        options.append("routed_svr10")
                    scores = {}
                    def predict_column(name, a, b):
                        if name == "routed_svr10":
                            return QualityRoutedRegressor("svr10","mean").fit(
                                x[a,:2],y[a,j]).predict(x[b,:2])
                        return estimator(name).fit(x[a],y[a,j:j+1]).predict(x[b]).ravel()
                    for family in options:
                        errors = []
                        for a,b in splits:
                            a,b=train[a],train[b]
                            errors.extend(abs(y[b,j]-predict_column(family,a,b)))
                        scores[family] = float(np.mean(errors))
                    best = min(scores,key=scores.get)
                    scalar.append(predict_column(best,train,test))
                    scalar_selections[target] = dict(selected=best,inner_mae=scores)
                predictions["independent_per_target"] = np.column_stack(scalar)
                selections.append(dict(cohort=cohort,golfer=held,features=feature,
                                       method="independent_per_target",targets=scalar_selections))
                for method,p in predictions.items():
                    err=miss(y[test],p)
                    for i,idx in enumerate(test):
                        rows.append(dict(cohort=cohort,features=feature,method=method,
                                         shot_id=data.iloc[idx].shot_id,golfer=held,landing_2d_yd=err[i,0],
                                         resting_2d_yd=err[i,1], carry_ae_yd=abs(p[i,0]-y[idx,0]),
                                         side_ae_yd=abs(p[i,1]-y[idx,1]),
                                         **{f"{label}_{j}":float(v) for label,values in
                                            [("actual",y[idx,:4]),("predicted",p[i])] for j,v in enumerate(values)}))
    out=ROOT/"experiments"/args.out
    out.mkdir(exist_ok=True)
    p=pd.DataFrame(rows)
    keys=["cohort","features","method"]
    assert not p.duplicated(keys+["shot_id"]).any()
    m=p.groupby(keys).agg(n=("shot_id","size"),landing_2d_yd=("landing_2d_yd","mean"),
                          resting_2d_yd=("resting_2d_yd","mean"),carry_mae=("carry_ae_yd","mean"),
                          side_mae=("side_ae_yd","mean")).reset_index()
    p.to_csv(out/"predictions.csv",index=False)
    m.to_csv(out/"metrics.csv",index=False)
    (out/"protocol.json").write_text(json.dumps(dict(targets=targets,coordinates=COORDS,
        objective="Mean Euclidean error across landing and resting endpoints in inner golfer folds. Independent_per_target uses inner scalar MAE separately for each label.",
        selection=selections),indent=2))
    print(m.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
