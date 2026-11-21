"""Held-golfer automatic quality routing, scale proxies and a separate point-review intervention."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from carry_estimators import QualityRoutedRegressor as Routed
from launch_geometry_features import fit
from early_ray_recovery import recover

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/clean_launch_audit'
OUT.mkdir(exist_ok=True)
old = pd.read_csv(ROOT/'experiments/clean_coupling_v1/predictions.csv').query("method=='club_blend'")
d = old[['shot_id','Golfer','actual_carry','carry']].copy().reset_index(drop=True)
order = pd.read_csv(ROOT/'experiments/label_stack_v1/pre_rollout_viewer.csv').id
d = d.set_index('shot_id').loc[order[order.isin(d.shot_id)]].reset_index(names='shot_id')
for name in ['launch_geometry_features','sensor_features','launch_scale_features']:
    d = d.merge(pd.read_csv(ROOT/f'data/processed/{name}.csv'),validate='one_to_one')
label = pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').set_index('Shot ID')
d['speed'] = label.loc[d.shot_id,'Ball speed (mph)'].to_numpy()
manual = pd.read_csv(ROOT/'annotations/clean_launch_audit_points.csv')
core = ['proj_vy','proj_depth_rate']
review = []
for i,sid in enumerate(d.shot_id):
    meta = json.loads((ROOT/f'outputs/native_review/{sid}.json').read_text())
    auto,alt = [json.loads((ROOT/f'data/processed/{folder}/{sid}.json').read_text())
                for folder in ['ball_launch_tracks','audited_ball_tracks']]
    seed = auto['seed']
    early = [o for o in auto['observations'] if 0<o['playback_s']-seed['playback_s']<=.08]
    other = {o['frame']:o for o in alt['observations']}
    dist = [np.hypot(o['x']-other[o['frame']]['x'],o['y']-other[o['frame']]['y']) for o in early if o['frame'] in other]
    d.loc[i,'support'] = len(early)
    d.loc[i,'agreement_px'] = np.median(dist) if len(dist)>=3 else np.nan
    d.loc[i,'poor'] = (len(early)<6 or len(dist)<3 or np.median(dist)>15 or (alt.get('shaft_conflicts') or 0)>0)
    ray,track = recover(ROOT,sid,seed) if len(early)<6 else ({},[])
    image_ray,image_track = recover(ROOT,sid,seed,from_images=True) if len(early)<6 else ({},[])
    if image_track:
        (OUT/f'{sid}_image.json').write_text(json.dumps(image_track))
    if track:
        (OUT/f'{sid}_ray.json').write_text(json.dumps(track))
    alt_features = fit(alt['observations'],alt['seed'])
    for k in core:
        d.loc[i,'image_'+k] = image_ray.get(k,np.nan)
        d.loc[i,'ray_'+k] = ray.get(k,np.nan)
        d.loc[i,'alt_'+k] = alt_features.get(k,np.nan)
    pts = manual[manual.shot_id==sid].copy()
    if pts.empty: continue
    pts['playback_s'] = np.array(meta['source_pts'])[pts.frame]
    measured = fit(pts.to_dict('records'),seed)
    cut = pts.playback_s.max()-seed['playback_s']
    comparable = fit(early,seed,max_elapsed_s=cut+1e-8)
    for k in core:
        d.loc[i,'review_'+k] = measured[k]
        d.loc[i,'window_'+k] = comparable.get(k,np.nan)
    matched = pts.merge(pd.DataFrame(early),on='frame',suffixes=('_review','_auto'))
    offsets = np.hypot(matched.x_review-matched.x_auto,matched.y_review-matched.y_auto)
    review.append(dict(shot_id=sid,last_reviewed_frame=max(pts.frame),manual_points=len(pts),
                       matched=len(matched),median_offset_px=np.median(offsets) if len(offsets) else np.nan,
                       first_offset_over_20_px=matched.loc[offsets>20,'frame'].min()))
d['tee_speed_proxy'] = np.hypot(d.proj_vx*1080,d.proj_vy*1920)*.04267/d.stationary_ball_diameter_px
d['alternate_ok'] = d[['alt_'+k for k in core]].notna().all(axis=1)
for k in core:
    d['recovered_'+k] = d['image_'+k].fillna(d[k])
assert len(d)==57 and d.shot_id.is_unique
groups = tuple(sorted(d.Golfer.unique()))
def other_groups(gs,g):
    return tuple(a for a in gs if a!=g)

@lru_cache(None)
def predict(gs,target,features,name):
    train = d.Golfer.isin(gs)
    cols = list(features)
    return Routed.branch(name).fit(d.loc[train,cols].to_numpy(),d.loc[train,target]).predict(d[cols].to_numpy())

records = []
for held in groups:
    gs = other_groups(groups,held)
    tr = d.Golfer.isin(gs)
    te = ~tr
    def branches(gs):
        return np.vstack([predict(gs,'actual_carry',tuple(core),'svr10'),
                          predict(gs,'actual_carry',tuple(core+['club_gyro_p95']),'svr1')])
    vid,gyro = branches(gs)
    blend = (vid+gyro)/2
    assert np.allclose(blend[te],d.carry[te])
    low = d.poor.astype(bool).to_numpy()
    weight = np.full(len(d),.5)
    # Fit two shrinkage-regularized weights on inner golfer-held-out predictions.
    oof = np.zeros((2,len(d)))
    for g in gs:
        select = d.Golfer==g
        oof[:,select] = branches(other_groups(gs,g))[:,select]
    for flag in [False,True]:
        mask = tr & (low==flag) & d.club_gyro_p95.notna()
        grid = np.array([0,.25,.5,.75,1])
        losses = abs(oof[0,mask,None]+(oof[1,mask]-oof[0,mask])[:,None]*grid-d.actual_carry.to_numpy()[mask,None]).sum(axis=0)
        weight[low==flag] = grid[(losses+8*abs(grid-.5)).argmin()]
    weight[d.club_gyro_p95.isna()] = 0
    def substituted(prefix):
        preds = []
        for name,cols in [('svr10',core),('svr1',core+['club_gyro_p95'])]:
            x = d[cols].copy()
            mask = d[[prefix+k for k in core]].notna().all(axis=1)
            x.loc[mask,core] = d.loc[mask,[prefix+k for k in core]].to_numpy()
            model = Routed.branch(name).fit(d.loc[tr,cols].to_numpy(),d.loc[tr,'actual_carry'])
            preds.append(model.predict(x.to_numpy()))
        return np.mean(preds,axis=0)
    scaled = predict(gs,'actual_carry',tuple(core+['tee_speed_proxy']),'svr1')
    recovered_core = ['recovered_'+k for k in core]
    refit = (predict(gs,'actual_carry',tuple(recovered_core),'svr10')+
             predict(gs,'actual_carry',tuple(recovered_core+['club_gyro_p95']),'svr1'))/2
    carry = {'image_recovery_refit':refit,'fixed_blend':blend,'quality_blend':vid+weight*(gyro-vid),
             'image_recovery':substituted('image_'),
             'early_ray_recovery':substituted('ray_'),
             'reroute_sparse':np.where(low & d.alternate_ok,substituted('alt_'),blend),
             'scaled_speed_blend':(vid+scaled)/2,
             'review_intervention':substituted('review_'),
             'automatic_window_control':substituted('window_')}
    speed = {'train_mean':np.full(len(d),d.loc[tr,'speed'].mean()),
             'recovered_geometry':predict(gs,'speed',tuple(recovered_core),'svr1'),
             'launch_geometry':predict(gs,'speed',tuple(core),'svr1'),
             'scale_augmented':predict(gs,'speed',tuple(core+['tee_speed_proxy']),'svr1')}
    # Calibrate typical error width for a new golfer using only training fold residuals.
    inner_errors = []
    for val in gs:
        vi = d.Golfer==val
        inner_pred = predict(other_groups(gs,val),'speed',tuple(recovered_core),'svr1')
        inner_errors.extend(abs(inner_pred[vi]-d.speed[vi]))
    width = np.quantile(inner_errors,.8)
    for target,methods,y in [('carry',carry,d.actual_carry),('speed',speed,d.speed)]:
        for method,p in methods.items():
            block = d.loc[te,['shot_id','Golfer','support','agreement_px','poor','tee_speed_proxy']].copy()
            block['weight'] = weight[te] if method=='quality_blend' else np.nan
            block['predicted'],block['actual'] = p[te],y[te]
            block['abs_error'] = abs(p[te]-y[te])
            block['speed_error_band'] = width if method=='recovered_geometry' else np.nan
            records.append(block.assign(method=method,target=target))
p = pd.concat(records,ignore_index=True)
p.to_csv(OUT/'predictions.csv',index=False)
p.groupby(['target','method','Golfer']).abs_error.mean().to_csv(OUT/'per_golfer.csv')
print(p.groupby(['target','method']).abs_error.mean())
r = pd.DataFrame(review).merge(p[p.target=='carry'].pivot(index='shot_id',columns='method',values='predicted'),on='shot_id')
r=r.merge(d[['shot_id','actual_carry','window_proj_vy']],on='shot_id')
r['automatic_window_available']=r.pop('window_proj_vy').notna()
r.to_csv(OUT/'review_intervention.csv',index=False)
print(r.round(1).to_string(index=False))
