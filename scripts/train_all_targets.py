"""Train LOGO models for ALL TrackMan targets using audited geometry features.

Trains: Carry side (ft), Curve (ft), Height (ft), Landing angle (deg),
        Hang time (s), Ball speed (mph), Launch angle (deg), Launch direction (deg),
        Carry flat (yd), Total (yd).

Uses the same rigorous nested LOGO protocol as the audited carry benchmark:
- Outer loop: Leave-One-Golfer-Out
- Inner loop: 3-fold grouped CV for model selection
- No test-set peeking
"""
import json, os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'all_targets_v1'
OUT.mkdir(exist_ok=True)

# --- Load data ---
sys_path = str(ROOT / 'scripts')
import sys; sys.path.insert(0, sys_path)
from benchmark import load_data
_, d = load_data()

# Merge geometry features
geom = pd.read_csv(ROOT / 'data/processed/launch_geometry_features.csv')
audited = pd.read_csv(ROOT / 'data/processed/audited_launch_features.csv')
audited = audited.rename(columns={c: 'launch_' + c for c in audited if c != 'shot_id'})
contact = pd.read_csv(ROOT / 'data/processed/contact_features.csv')
d = d.merge(geom, on='shot_id', how='left').merge(audited, on='shot_id', how='left').merge(contact, on='shot_id', how='left')

# Load TrackMan labels
labels = pd.read_csv(ROOT / 'data/manifests/trackman_labels.csv')
labels = labels.rename(columns={'Shot ID': 'shot_id'})
for col in ['Curve signed (ft; R+)', 'Carry side (ft; R+)', 'Height (ft)',
            'Landing angle (deg)', 'Hang time (s)', 'Total (yd)', 'Total side (ft; R+)']:
    if col in labels.columns:
        d = d.merge(labels[['shot_id', col]].dropna(subset=[col]), on='shot_id', how='left')

# --- Feature sets ---
# Core video geometry (the proven 2-feature set + extras)
core2 = ['proj_vy', 'proj_depth_rate']
core_full = ['proj_vy', 'proj_vx', 'proj_depth_rate', 'proj_shift', 'proj_fit_rmse', 'proj_angle', 'proj_asymptote_y']
fixed_time = ['launch_dx_020', 'launch_rise_020', 'launch_dx_040', 'launch_rise_040',
              'launch_dx_060', 'launch_rise_060', 'launch_dx_080', 'launch_rise_080', 'launch_tracked']
video_compact = core2 + ['proj_vx', 'proj_angle', 'proj_fit_rmse']
video_rich = core_full + ['fit_dx_030', 'fit_rise_030', 'fit_dx_050', 'fit_rise_050',
                          'fit_dx_100', 'fit_rise_100']

# Watch features (numeric only - exclude file paths)
watch_cols = [c for c in d.columns if c.startswith('watch_') and d[c].dtype in ['float64', 'int64']
              and c not in ('watch_impact_s', 'watch_peak_candidate_s')]
# Club features (numeric only)
club_cols = [c for c in d.columns if (c.startswith('club_gyro') or c.startswith('club_accel'))
             and d[c].dtype in ['float64', 'int64']]
# Body features (numeric only)
body_cols = [c for c in d.columns if (c.startswith('head_range') or c.startswith('shoulders_range')
             or c.startswith('hips_range') or c.startswith('flow_'))
             and d[c].dtype in ['float64', 'int64']]

FEATURE_SETS = {
    'core2': core2,
    'core2_angle': core2 + ['proj_vx', 'proj_angle'],
    'video_compact': video_compact,
    'video_rich': video_rich,
    'video_fixed_time': fixed_time,
    'video_all': video_rich + fixed_time,
    'video_watch': video_compact + watch_cols,
    'video_club': video_compact + club_cols,
    'video_body': video_compact + body_cols,
    'video_fusion': video_compact + watch_cols + club_cols,
    'full_fusion': video_rich + fixed_time + watch_cols + club_cols + body_cols,
}

# Filter to only columns that actually exist
for k in FEATURE_SETS:
    FEATURE_SETS[k] = [c for c in FEATURE_SETS[k] if c in d.columns]

# --- Model families ---
FAMILIES = {
    'mean': lambda n: DummyRegressor(strategy='mean'),
    'ridge10': lambda n: make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                       StandardScaler(), Ridge(alpha=10.0)),
    'ridge100': lambda n: make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                        StandardScaler(), Ridge(alpha=100.0)),
    'svr1': lambda n: make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                    StandardScaler(), TransformedTargetRegressor(
                                        regressor=SVR(C=1.0, epsilon=0.1), transformer=StandardScaler())),
    'svr10': lambda n: make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                     StandardScaler(), TransformedTargetRegressor(
                                         regressor=SVR(C=10.0, epsilon=0.1), transformer=StandardScaler())),
    'trees': lambda n: make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                     StandardScaler(), ExtraTreesRegressor(
                                         n_estimators=150, min_samples_leaf=4, random_state=396, n_jobs=1)),
    'gbr': lambda n: make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                   StandardScaler(), GradientBoostingRegressor(
                                       n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=396)),
}

# --- Targets ---
TARGETS = {
    'Carry flat (yd)': 'carry_yd',
    'Launch direction (deg; R+)': 'launch_dir_deg',
    'Ball speed (mph)': 'ball_speed_mph',
    'Launch angle (deg)': 'launch_angle_deg',
    'Curve signed (ft; R+)': 'curve_ft',
    'Carry side (ft; R+)': 'carry_side_ft',
    'Height (ft)': 'height_ft',
    'Landing angle (deg)': 'landing_angle_deg',
    'Hang time (s)': 'hang_time_s',
    'Total (yd)': 'total_yd',
}

# Ensure no target leakage
target_cols = set(TARGETS.keys())

all_predictions = []
all_metrics = []
all_selections = []

for target_name, target_short in TARGETS.items():
    if target_name not in d.columns:
        print(f'SKIP {target_name}: not in data')
        continue

    df = d[d[target_name].notna()].copy()
    n_shots = len(df)
    golfers = sorted(df.golfer.unique())
    print(f'\n=== {target_name} ({n_shots} shots, {len(golfers)} golfers) ===')

    for feat_name, feat_cols in FEATURE_SETS.items():
        # Ensure no target leakage
        safe_cols = [c for c in feat_cols if c not in target_cols]
        if not safe_cols:
            continue

        for golfer in golfers:
            tr = df[df.golfer != golfer]
            te = df[df.golfer == golfer]

            # Inner CV for model selection
            n_inner = min(3, tr.golfer.nunique())
            splits = list(GroupKFold(n_inner).split(tr, groups=tr.golfer))

            inner_scores = {}
            outer_preds = {}

            for family_name, family_fn in FAMILIES.items():
                # Inner CV score
                y_inner, p_inner = [], []
                for ti, vi in splits:
                    a, b = tr.iloc[ti], tr.iloc[vi]
                    pipe = family_fn(len(safe_cols))
                    pipe.fit(a[safe_cols], a[target_name])
                    y_inner.extend(b[target_name])
                    p_inner.extend(np.asarray(pipe.predict(b[safe_cols])).ravel())
                inner_scores[family_name] = mean_absolute_error(y_inner, p_inner)

                # Outer prediction
                pipe = family_fn(len(safe_cols))
                pipe.fit(tr[safe_cols], tr[target_name])
                values = np.asarray(pipe.predict(te[safe_cols])).ravel()
                outer_preds[family_name] = values

                for sid, g, actual, pred_val in zip(te.shot_id, te.golfer, te[target_name], values):
                    all_predictions.append(dict(
                        target=target_short, features=feat_name, method=family_name,
                        shot_id=sid, golfer=g, actual=float(actual), predicted=float(pred_val)))

            # Nested: pick best inner family
            best_family = min(inner_scores, key=inner_scores.get)
            for sid, g, actual, pred_val in zip(te.shot_id, te.golfer, te[target_name], outer_preds[best_family]):
                all_predictions.append(dict(
                    target=target_short, features=feat_name, method='nested',
                    shot_id=sid, golfer=g, actual=float(actual), predicted=float(pred_val)))

            all_selections.append(dict(
                target=target_short, features=feat_name, held_golfer=golfer,
                selected=best_family, inner_mae=inner_scores))

    # Per feature set + method metrics
    preds_df = pd.DataFrame([p for p in all_predictions if p['target'] == target_short])
    for (feat, method), grp in preds_df.groupby(['features', 'method']):
        mae = mean_absolute_error(grp.actual, grp.predicted)
        median_ae = float(np.median(np.abs(grp.actual - grp.predicted)))
        all_metrics.append(dict(target=target_short, features=feat, method=method,
                                n=len(grp), mae=mae, median_ae=median_ae))

    # Also do nested feature selection (pick best feature set inside inner folds)
    for golfer in golfers:
        sel_entries = [s for s in all_selections if s['target'] == target_short and s['held_golfer'] == golfer]
        best_entry = min(sel_entries, key=lambda s: min(s['inner_mae'].values()))
        best_feat = best_entry['features']
        best_fam = best_entry['selected']
        source = [p for p in all_predictions
                  if p['target'] == target_short and p['features'] == best_feat
                  and p['method'] == best_fam and p['golfer'] == golfer]
        for p in source:
            all_predictions.append({**p, 'features': 'nested_best', 'method': 'nested'})

    # Metric for nested_best
    nest_preds = [p for p in all_predictions if p['target'] == target_short and p['features'] == 'nested_best']
    if nest_preds:
        ndf = pd.DataFrame(nest_preds)
        mae = mean_absolute_error(ndf.actual, ndf.predicted)
        median_ae = float(np.median(np.abs(ndf.actual - ndf.predicted)))
        all_metrics.append(dict(target=target_short, features='nested_best', method='nested',
                                n=len(ndf), mae=mae, median_ae=median_ae))
        print(f'  nested_best MAE: {mae:.2f} (median: {median_ae:.2f})')

# Save results
pd.DataFrame(all_predictions).to_csv(OUT / 'predictions.csv', index=False)
pd.DataFrame(all_metrics).to_csv(OUT / 'metrics.csv', index=False)
with open(OUT / 'selections.json', 'w') as f:
    json.dump(all_selections, f, indent=2)

# Summary table
print('\n\n=== FINAL SUMMARY: Best LOGO MAE Per Target ===')
mdf = pd.DataFrame(all_metrics)
for target_short in TARGETS.values():
    sub = mdf[mdf.target == target_short]
    if sub.empty:
        continue
    best = sub.loc[sub.mae.idxmin()]
    mean_row = sub[(sub.method == 'mean')]
    baseline = mean_row.mae.min() if len(mean_row) > 0 else float('nan')
    print(f'{target_short:>20s}:  Baseline {baseline:.2f}  |  Best {best.mae:.2f} ({best.features}/{best.method})  |  Median {best.median_ae:.2f}')

print('\nSaved to', OUT)
