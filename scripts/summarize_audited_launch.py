"""Recompute paired carry comparisons and save an auditable model-study summary."""
import hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/audited_launch_summary'
OUT.mkdir(exist_ok=True)
p=pd.read_csv(ROOT/'experiments/audited_launch_v7/predictions.csv')
old=pd.read_csv(ROOT/'experiments/video_launch_v1/predictions.csv')
old=old[old.target.eq('Carry flat (yd)')&old.model.eq('ridge10')]
nested=pd.read_csv(ROOT/'experiments/nested_carry_v1/predictions.csv')
nested=nested[nested.scheme.eq('leave_golfer_out')]
rows=[];paired=[]
def add(name,q):
    q=q[['shot_id','golfer','actual','predicted']].copy();q['model']=name
    q['abs_error']=abs(q.actual-q.predicted)
    paired.append(q)
    rows.append(dict(model=name,n=len(q),mae=q.abs_error.mean(),median_ae=q.abs_error.median(),within10=float((q.abs_error<=10).mean()),within20=float((q.abs_error<=20).mean())))
base=p[p.cohort.eq('all_labeled')&p.cutoff.eq(0)&p.features.eq('vertical_curve')]
add('Training mean',base[base.method.eq('mean')]);add('Gemini fixed ridge (retrospective)',old)
add('Earlier body video',nested[nested.cohort.eq('all_labeled')&nested.features.eq('camera')])
add('Earlier body video + watch',nested[nested.cohort.eq('all_labeled')&nested.features.eq('camera_watch')])
add('Audited video recipe',base[base.method.eq('nested')])
add('Audited joint selector',p[p.cohort.eq('all_labeled')&p.cutoff.eq(0)&p.features.eq('nested_fusion')&p.method.eq('nested')])
summary=pd.DataFrame(rows);summary.to_csv(OUT/'comparison.csv',index=False)
q=pd.concat(paired);q.to_csv(OUT/'paired_predictions.csv',index=False)
q.groupby(['model','golfer']).agg(n=('shot_id','size'),mae=('abs_error','mean')).to_csv(OUT/'per_golfer.csv')
new=q[q.model.eq('Audited video recipe')].copy()
new['available']=new.shot_id.isin(pd.read_csv(ROOT/'data/processed/launch_geometry_features.csv').dropna(subset=['proj_vy','proj_depth_rate']).shot_id)
new.sort_values('abs_error',ascending=False).to_csv(OUT/'error_audit.csv',index=False)
# Cluster resampling acknowledges shared golfer errors, but cannot remove adaptive search bias.
rng=np.random.default_rng(396);boot=[]
for name in ['Training mean','Gemini fixed ridge (retrospective)','Earlier body video + watch']:
    a=q[q.model.eq(name)].merge(new,on=['shot_id','golfer','actual'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(a)==72
    groups={g:v for g,v in a.groupby('golfer')};names=list(groups);dif=[]
    for i in range(10000):
        chosen=rng.choice(names,len(names),replace=True)
        d=np.concatenate([(groups[g].abs_error_old-groups[g].abs_error_new).to_numpy() for g in chosen]);dif.append(d.mean())
    boot.append(dict(comparator=name,mae_reduction=float((a.abs_error_old-a.abs_error_new).mean()),cluster_bootstrap_low=float(np.quantile(dif,.025)),cluster_bootstrap_high=float(np.quantile(dif,.975)),caveat='Only five golfers; descriptive resampling does not account for adaptive feature/tracker development.'))
pd.DataFrame(boot).to_csv(OUT/'paired_cluster_intervals.csv',index=False)
# Conditional training versus conditional scoring of the all-shot model.
threshold=[]
for cohort in ['all_labeled','matched_club']:
    full=p[p.cohort.eq(cohort)&p.cutoff.eq(0)&p.features.eq('vertical_curve')]
    for cutoff in [70,100]:
        restricted=pd.read_csv(ROOT/'experiments/audited_launch_above70/predictions.csv') if cutoff==70 else p
        restricted=restricted[restricted.cohort.eq(cohort)&restricted.cutoff.eq(cutoff)&restricted.features.eq('vertical_curve')]
        for protocol,data in [('all-shot model, score actual above cutoff',full[full.actual>cutoff]),('train/test actual above cutoff (oracle)',restricted)]:
            for method in ['mean','nested']:
                a=data[data.method.eq(method)];threshold.append(dict(cohort=cohort,cutoff=cutoff,protocol=protocol,method=method,n=len(a),mae=a.abs_error.mean()))
pd.DataFrame(threshold).to_csv(OUT/'thresholds.csv',index=False)
# Preserve local derived inputs and hashes, not raw recordings.
inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
for name in ['audited_launch_features.csv','launch_geometry_features.csv','launch_scale_features.csv','audited_ball_geometry_features.csv','video_launch_features.csv']:
    shutil.copy2(ROOT/'data/processed'/name,inputs/name)
files=list((ROOT/'scripts').glob('*launch*.py'))+[ROOT/'scripts/carry_estimators.py',ROOT/'scripts/club_delivery_candidates.py',ROOT/'scripts/launch_scale_features.py']
files+=list(inputs.glob('*.csv'))+list((ROOT/'data/processed/ball_launch_tracks').glob('*.json'))
files+=list((ROOT/'data/manifests').glob('*.csv'))+list((ROOT/'annotations').glob('*.csv'))
for exp in ['audited_launch_v'+str(i) for i in range(1,8)]+['audited_launch_above70','audited_ball_speed','audited_direction']:
    folder=ROOT/'experiments'/exp
    if not (folder/'source_script.py').exists():shutil.copy2(ROOT/'scripts/audited_launch_benchmark.py',folder/'source_script.py')
    files+=list(folder.glob('*.json'))+list(folder.glob('*.csv'))+list(folder.glob('*.py'))
hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))}
(OUT/'provenance.json').write_text(json.dumps(dict(scope='Derived inputs, source and results. Original source hashes remain in the existing audit manifest; raw media not rehashed in this model-only milestone.',sha256=hashes),indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for name,color in [('Gemini fixed ridge (retrospective)','#b56742'),('Audited video recipe','#227ca0')]:
    a=q[q.model.eq(name)];axes[0].scatter(a.actual,a.predicted,s=25,alpha=.65,label=name,color=color)
axes[0].plot([0,180],[0,180],'k--',lw=1);axes[0].set(xlabel='Recorded carry (yards)',ylabel='Held-out prediction (yards)',title='72 shots: leave golfer out');axes[0].legend(fontsize=8)
a=q[q.model.isin(['Gemini fixed ridge (retrospective)','Audited video recipe'])].groupby(['golfer','model']).abs_error.mean().unstack();a.plot.bar(ax=axes[1],color=['#227ca0','#b56742']);axes[1].set(ylabel='Mean absolute error (yards)',xlabel='',title='Each golfer held out');axes[1].tick_params(axis='x',rotation=0);axes[1].legend(fontsize=7)
fig.suptitle('Exploratory post-impact estimation; repeated dataset development');fig.tight_layout();fig.savefig(OUT/'carry_comparison.png',dpi=160);plt.close(fig)
print(summary.to_string(index=False));print(pd.DataFrame(threshold).to_string(index=False));print(pd.DataFrame(boot).to_string(index=False))
