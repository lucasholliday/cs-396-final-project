"""Strict clean-cohort nested golfer holdout for post-launch endpoint estimation."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_joint_landing import AUX, COORDS, GYRO, VIDEO
from carry_estimators import QualityRoutedRegressor as Routed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/clean_coupling_v1'
OUT.mkdir(exist_ok=True)
mask = pd.read_csv(ROOT/'experiments/label_stack_v1/pre_rollout_viewer.csv')
ids = mask.loc[mask.used_primary_model & mask.best_prediction.notna(), 'id']
data = pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').rename(columns={'Shot ID':'shot_id'})
data = data.set_index('shot_id').loc[ids].reset_index(names='shot_id')
for name in ['launch_geometry_features','club_fusion_features','sensor_features','audited_launch_features','launch_scale_features','club_delivery_features']:
    frame = pd.read_csv(ROOT/f'data/processed/{name}.csv')
    data = data.merge(frame, on='shot_id', validate='one_to_one')
data.loc[~data.has_club_imu, GYRO] = np.nan
targets = data[COORDS+AUX].to_numpy(float)
targets[:,[1,3]] /= 3
group = data.Golfer.to_numpy()
golfers = tuple(sorted(set(group)))
assert len(data) == len(set(ids)) == 57 and len(golfers) == 5 and np.isfinite(targets).all()

core = VIDEO[:2]
wide = VIDEO+['proj_angle','proj_shift','fit_dx_030','fit_rise_030','fit_dx_100','fit_rise_100']
recipes = [(core,'svr10'),(wide+GYRO,'svr1'),(core,'svr1'),(wide+GYRO,'svr1'),
           (core,'svr1'),(core,'svr1'),(wide,'ridge10'),(core,'svr1'),
           (core,'svr1'),(VIDEO+['proj_angle'],'svr10'),(wide+GYRO,'svr1')]
# The projection model can trade fitted initial speed against depth rate.
# Apparent rise after a fixed time is less sensitive to that tradeoff.
data['rise60'] = .06*data.proj_vy/(1+.06*data.proj_depth_rate)
for key in core+['rise60']:
    data['log_'+key] = np.log1p(data[key].clip(lower=0))
alternatives = {
    'smooth_launch': (core,'svr1'),
    'log_launch': (['log_'+k for k in core+['rise60']],'ridge1'),
    'early_rise': (['rise60','fit_rise_030'],'svr1'),
    'launch_body_scale': (core+['body_height_px'],'svr1'),
    'launch_horizontal': (core+['proj_vx'],'svr1'),
    'launch_quality': (core+['proj_fit_rmse'],'svr1'),
    'two_trackers': (core+['vy_early','perspective_b'],'svr1'),
    'club_delivery': (core+['delivery_image_speed'],'svr1'),
    'launch_watch': (core+['watch_gyro_p95'],'svr1'),
    'launch_club': (core+['club_gyro_p95'],'svr1'),
}

@lru_cache(None)
def first(train):
    rows = np.isin(group,train)
    def estimate(cols, recipe, j):
        x = data[cols].to_numpy(float)
        return Routed.branch(recipe).fit(x[rows],targets[rows,j]).predict(x)
    pred = np.column_stack([estimate(cols,recipe,j) for j,(cols,recipe) in enumerate(recipes)])
    extra = {k:estimate(cols,recipe,0) for k,(cols,recipe) in alternatives.items()}
    return pred,extra

def without(train, golfer):
    return tuple(g for g in train if g != golfer)

def slope(p):
    # End tangent of a horizontal quadratic starting at the predicted launch azimuth.
    return 2*p[:,1]/np.maximum(1,p[:,0])-np.tan(np.radians(p[:,6]))

def ridge(x,y,xnew):
    return make_pipeline(StandardScaler(),Ridge(alpha=20)).fit(x,y).predict(xnew)

@lru_cache(None)
def variants(train):
    rows = np.isin(group,train)
    a = np.zeros_like(targets)
    for val in train:
        a[group==val] = first(without(train,val))[0][group==val]
    a,y = a[rows],targets[rows]
    b,extra = first(train)
    idx = [0,4,5,7,8,9]
    run = np.maximum(0,ridge(a[:,idx],y[:,2]-y[:,0],b[:,idx]))
    base = b.copy()
    base[:,2] = base[:,0]+run
    carries = {'current':base[:,0], **extra}
    carries['club_blend'] = .5*(base[:,0]+extra['launch_club'])
    carries['ensemble'] = np.mean([base[:,0],extra['smooth_launch'],extra['log_launch']],axis=0)
    # Fit a nonnegative penalty using training golfers' out-of-fold residuals.
    # Zero is a candidate when the existing carry model already absorbs curvature.
    for name, transform in [('slice_drag',lambda x: np.maximum(0,x)),('bend_drag',np.abs)]:
        curvature = transform(a[:,10])/3
        coeff = np.linspace(0,1,41)
        loss = abs(a[:,0,None]-curvature[:,None]*coeff-y[:,0,None]).mean(axis=0)
        carries[name] = np.maximum(0,base[:,0]-coeff[loss.argmin()]*transform(b[:,10])/3)
    return base,run,carries

@lru_cache(None)
def turf_factor(train):
    # Estimate turf attenuation from training endpoint labels only.
    truth = targets[np.isin(group,train)]
    ideal_shift = (truth[:,2]-truth[:,0])*slope(truth)
    grid = np.linspace(0,1,101)
    residual = truth[:,3,None]-truth[:,1,None]-ideal_shift[:,None]*grid
    return grid[np.abs(residual).mean(axis=0).argmin()]

def place(base,run,carry,roll,k=.22):
    p = base.copy()
    p[:,0],p[:,2] = carry,carry+run
    tangent = run*slope(p)
    if roll in ('anchor_rest','joint_anchor'):
        # Invert the damped tangent relation at the directly predicted final side.
        # Blending the resulting landing estimates preserves the same relation.
        inferred = (base[:,3]+k*run*np.tan(np.radians(p[:,6])))/(1+2*k*run/np.maximum(1,carry))
        w = 1 if roll=='anchor_rest' else .5
        p[:,1] = (1-w)*p[:,1]+w*inferred
        p[:,3] = p[:,1]+k*run*slope(p)
    elif roll=='tangent':
        p[:,3] = p[:,1]+tangent
    elif roll=='damped':
        p[:,3] = p[:,1]+.22*tangent
    elif roll=='bounded':
        # Nearest independent endpoint between no lateral roll and full tangent.
        p[:,3] = p[:,1]+np.clip(base[:,3]-p[:,1],np.minimum(0,tangent),np.maximum(0,tangent))
    return p

rows,choices = [],[]
for held in golfers:
    train = without(golfers,held)
    test = group==held
    base,run,carries = variants(train)
    carry_losses = {key:[] for key in carries}
    roll_losses = {key:[] for key in ['tangent','damped','bounded','anchor_rest','joint_anchor']}
    for val in train:
        inner = without(train,val)
        vi = group==val
        b,r,c = variants(inner)
        for key,value in c.items():
            carry_losses[key].extend(abs(value[vi]-targets[vi,0]))
        for key in roll_losses:
            p = place(b,r,c['current'],key)
            errors = (p-targets)[vi]
            roll_losses[key].extend(np.hypot(*errors[:,:2].T)+np.hypot(*errors[:,2:4].T))
    best = min(carries,key=lambda k:np.mean(carry_losses[k]))
    roll = min(roll_losses,key=lambda k:np.mean(roll_losses[k]))
    choices.append(dict(golfer=held,carry=best,roll=roll))
    models = {k:place(base,run,v,'direct') for k,v in carries.items()}
    for key in roll_losses:
        models[key] = place(base,run,carries['current'],key)
    models['club_blend_anchor'] = place(base,run,carries['club_blend'],'anchor_rest')
    models['learned_turf'] = place(base,run,carries['club_blend'],'anchor_rest',turf_factor(train))
    choices[-1]['train_turf_factor'] = turf_factor(train)
    models['physics_pair'] = place(base,run,carries['slice_drag'],'tangent')
    models['nested'] = place(base,run,carries[best],roll)
    models['train_mean'] = np.tile(targets[~test].mean(axis=0),(len(group),1))
    for method,p in models.items():
        block = data.loc[test,['shot_id','Golfer']].copy()
        for j,key in enumerate(['carry','side','total','final_side']):
            block[key] = p[test,j]
            block['actual_'+key] = targets[test,j]
        err = (p-targets)[test]
        block['carry_mae'],block['total_mae'] = abs(err[:,0]),abs(err[:,2])
        block['landing'],block['resting'] = np.hypot(*err[:,:2].T),np.hypot(*err[:,2:4].T)
        shift = (p[:,3]-p[:,1])[test]
        block['side_shift_mae'] = abs(shift-(targets[test,3]-targets[test,1]))
        block['arrival_slope'] = slope(p)[test]
        block['opposite'] = shift*block.arrival_slope < -1e-8
        block['opposite_over_1yd'] = block.opposite & (abs(shift)>1)
        rows.append(block.assign(method=method))
out = pd.concat(rows,ignore_index=True)
metrics = ['carry_mae','total_mae','landing','resting','side_shift_mae','opposite','opposite_over_1yd']
out.to_csv(OUT/'predictions.csv',index=False)
out.groupby(['method','Golfer'])[metrics].mean().to_csv(OUT/'per_golfer.csv')
scores = out.groupby('method')[metrics].mean()
scores.to_csv(OUT/'metrics.csv')
pd.DataFrame(choices).to_csv(OUT/'selections.csv',index=False)
print(scores.round(3))
print(choices)
