"""Benchmark predictive accuracy using pure physics-gated video launch and clubhead features.
Evaluated strictly under the shared persisted Leave-One-Golfer-Out splits.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    roc_auc_score, average_precision_score, precision_score, recall_score
)

ROOT = Path(__file__).resolve().parents[1]


def build_regressor(name, seed=396):
    if name == 'ridge10':
        est = Ridge(alpha=10)
    elif name == 'extra_trees':
        est = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=4, random_state=seed, n_jobs=1)
    elif name == 'svr':
        est = TransformedTargetRegressor(regressor=SVR(C=1.0, epsilon=0.1), transformer=StandardScaler())
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)


def main():
    cfg = json.loads((ROOT / 'configs/benchmark.json').read_text())
    m = pd.read_csv(ROOT / 'data/manifests/shot_manifest.csv')
    labels = pd.read_csv(ROOT / 'data/manifests/trackman_labels.csv').rename(columns={'Shot ID': 'shot_id'})
    splits = pd.read_csv(ROOT / 'experiments/baseline_v1/splits.csv')
    v_features = pd.read_csv(ROOT / 'data/processed/video_launch_features.csv')
    
    # Merge dataset
    v_clean = v_features.drop(columns=['golfer', 'session'], errors='ignore')
    df = m[m.training_eligible].merge(labels[['shot_id'] + cfg['targets']], on='shot_id')
    df = df.merge(splits[['shot_id', 'golfer_fold', 'block']], on='shot_id')
    df = df.merge(v_clean, on='shot_id')
    
    out_dir = ROOT / 'experiments/video_launch_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [
        'ball_angle_deg', 'ball_speed_px_s', 'ball_rise_px', 
        'ball_lateral_dx_px', 'ball_duration_s', 'abs_angle_deg', 
        'mishit_push_flag', 'club_speed_px_s', 'smash_factor_proxy'
    ]
    
    # Target evaluation
    metrics_records = []
    predictions_records = []
    
    print(f'Running Leave-One-Golfer-Out evaluation on {len(df)} shots across {df.golfer_fold.nunique()} golfers...')
    
    for target in cfg['targets']:
        for model_name in ['ridge10', 'svr', 'extra_trees']:
            preds, truths, shot_ids, golfers = [], [], [], []
            for fold in sorted(df.golfer_fold.unique()):
                train = df[df.golfer_fold != fold]
                test = df[df.golfer_fold == fold]
                
                pipe = build_regressor(model_name, cfg['seed'])
                pipe.fit(train[feature_cols], train[target])
                p = pipe.predict(test[feature_cols])
                
                preds.extend(p)
                truths.extend(test[target])
                shot_ids.extend(test.shot_id)
                golfers.extend(test.golfer)
                
            y_true = np.array(truths)
            y_pred = np.array(preds)
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))
            med_ae = float(median_absolute_error(y_true, y_pred))
            
            metrics_records.append({
                'target': target,
                'model': model_name,
                'n': len(y_true),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'median_ae': med_ae
            })
            
            for sid, g, act, pred in zip(shot_ids, golfers, y_true, y_pred):
                predictions_records.append({
                    'shot_id': sid,
                    'golfer': g,
                    'target': target,
                    'model': model_name,
                    'actual': act,
                    'predicted': pred,
                    'residual': act - pred,
                    'abs_error': abs(act - pred)
                })
                
    # Also evaluate Mishit Detection (Carry < 60 yd)
    df['short_carry'] = (df['Carry flat (yd)'] < 60).astype(int)
    clf_pipe = make_pipeline(
        SimpleImputer(strategy='median'), 
        StandardScaler(), 
        LogisticRegression(C=0.5, class_weight='balanced', random_state=cfg['seed'])
    )
    clf_preds, clf_probs, clf_truths = [], [], []
    for fold in sorted(df.golfer_fold.unique()):
        train = df[df.golfer_fold != fold]
        test = df[df.golfer_fold == fold]
        clf_pipe.fit(train[feature_cols], train['short_carry'])
        probs = clf_pipe.predict_proba(test[feature_cols])[:, 1]
        p = (probs >= 0.5).astype(int)
        clf_preds.extend(p)
        clf_probs.extend(probs)
        clf_truths.extend(test['short_carry'])
        
    y_true_cls = np.array(clf_truths)
    y_prob_cls = np.array(clf_probs)
    y_pred_cls = np.array(clf_preds)
    
    auroc = float(roc_auc_score(y_true_cls, y_prob_cls))
    pr_auc = float(average_precision_score(y_true_cls, y_prob_cls))
    prec = float(precision_score(y_true_cls, y_pred_cls, zero_division=0))
    rec = float(recall_score(y_true_cls, y_pred_cls, zero_division=0))
    
    cls_summary = {
        'task': 'short_carry_under_60yd',
        'n': len(y_true_cls),
        'positive_count': int(sum(y_true_cls)),
        'auroc': auroc,
        'average_precision': pr_auc,
        'precision': prec,
        'recall': rec
    }
    
    df_metrics = pd.DataFrame(metrics_records)
    df_preds = pd.DataFrame(predictions_records)
    
    df_metrics.to_csv(out_dir / 'metrics.csv', index=False)
    df_preds.to_csv(out_dir / 'predictions.csv', index=False)
    (out_dir / 'mishit_classification.json').write_text(json.dumps(cls_summary, indent=2))
    
    print('\n=== HELD-OUT GOLFER PREDICTION RESULTS (VIDEO LAUNCH ONLY) ===')
    for target, g in df_metrics.groupby('target'):
        best = g.sort_values('mae').iloc[0]
        print(f'{target:28s} Best ({best.model:11s}): MAE={best.mae:5.2f}, RMSE={best.rmse:5.2f}, R2={best.r2:+.3f}, MedAE={best.median_ae:5.2f}')
        
    print('\n=== MISHIT CLASSIFICATION (<60 yd) ===')
    print(f'AUROC: {auroc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f} ({int(sum(y_true_cls))} short shots)')


if __name__ == '__main__':
    main()
