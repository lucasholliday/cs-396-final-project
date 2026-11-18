"""Uncertainty, fixed-model learning curves, error diagnostics, and research plots."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from benchmark import load_data,model

cfg,d=load_data();base=ROOT/'experiments/baseline_v1'
out=ROOT/'outputs/plots';out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(base/'metrics.csv');p=pd.read_csv(base/'predictions.csv');splits=pd.read_csv(base/'splits.csv')
d=d.merge(splits[['shot_id','block','golfer_fold']],on='shot_id',validate='one_to_one')
config=json.loads((base/'config.json').read_text());sets=config['feature_sets']
rng=np.random.default_rng(cfg['seed']);uncertainty=[]
for scheme in cfg['schemes']:
    for family in ['ridge10','extra_trees','svr']:
        subset=p[(p.cohort=='matched_club')&(p.scheme==scheme)&(p.model==family)&(p.target=='Carry flat (yd)')]
        for a,b in [('camera_watch','camera'),('all','camera_watch'),('all','camera')]:
            left=subset[subset.features==a].set_index('shot_id')
            right=subset[subset.features==b].set_index('shot_id').loc[left.index]
            delta=np.abs(left.residual.to_numpy())-np.abs(right.residual.to_numpy())
            groups=left.golfer.to_numpy();names=np.unique(groups)
            draws=[]
            for _ in range(2000):
                sampled=rng.choice(names,len(names),replace=True)
                draws.append(np.concatenate([delta[groups==g] for g in sampled]).mean())
            lo,hi=np.quantile(draws,[.025,.975])
            uncertainty.append(dict(scheme=scheme,model=family,comparison=f'{a} minus {b}',n=len(delta),
                delta_mae=float(delta.mean()),cluster_bootstrap_low=float(lo),cluster_bootstrap_high=float(hi),
                clusters=len(names),bootstrap_replicates=2000))
pd.DataFrame(uncertainty).to_csv(base/'paired_uncertainty.csv',index=False)

curves=[]
for scheme in cfg['schemes']:
    foldcol='block' if scheme=='within_session_blocks' else 'golfer_fold'
    for feature in ['camera','watch','camera_watch']:
        for fold in sorted(d[foldcol].unique()):
            train=d[d[foldcol]!=fold];test=d[d[foldcol]==fold]
            for repeat in range(3):
                # Shuffle within golfers then round-robin to include every available training golfer.
                gen=np.random.default_rng(cfg['seed']+repeat+100*int(fold))
                buckets=[list(gen.permutation(g.index)) for _,g in train.groupby('golfer')]
                order=[]
                while any(buckets):
                    for bucket in buckets:
                        if bucket:order.append(bucket.pop())
                for n in [15,25,40,50]:
                    if n>len(order):continue
                    tr=d.loc[order[:n]];estimator=model('ridge10',cfg['seed'])
                    estimator.fit(tr[sets[feature]],tr['Carry flat (yd)'])
                    pred=estimator.predict(test[sets[feature]])
                    curves.append(dict(scheme=scheme,features=feature,fold=fold,repeat=repeat,n_train=n,
                                       mae=mean_absolute_error(test['Carry flat (yd)'],pred)))
c=pd.DataFrame(curves);c.to_csv(base/'learning_curves.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,scheme in zip(axes,cfg['schemes']):
    for f in ['camera','watch','camera_watch']:
        g=c[(c.scheme==scheme)&(c.features==f)].groupby('n_train').mae.agg(['mean','std'])
        ax.errorbar(g.index,g['mean'],yerr=g['std'],marker='o',capsize=3,label=f)
    ax.set(title=scheme.replace('_',' '),xlabel='Training shots',ylabel='Carry MAE (yd)');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Fixed ridge10 learning curves — error bars: fold/repeat SD')
fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,scheme in zip(axes,cfg['schemes']):
    sub=m[(m.cohort=='matched_club')&(m.scheme==scheme)&(m.target=='Carry flat (yd)')]
    pivot=sub.pivot(index='features',columns='model',values='mae').reindex(['camera','watch','club','camera_watch','camera_club','watch_club','all'])
    pivot[['ridge10','extra_trees','svr']].plot.bar(ax=ax,rot=40)
    ax.axhline(sub[sub.model=='median'].mae.iloc[0],color='black',ls='--',label='Median baseline')
    ax.set(title=scheme.replace('_',' '),ylabel='Carry MAE (yd)',xlabel='Feature set');ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=7)
fig.suptitle('Matched 65-shot cohort; whole-swing features, sensor-assisted video windows')
fig.tight_layout();fig.savefig(out/'carry_model_comparison.png',dpi=160);plt.close(fig)

chosen=p[(p.cohort=='all_labeled')&(p.scheme=='within_session_blocks')&(p.features=='camera')&(p.target=='Carry flat (yd)')&(p.model=='svr')]
chosen=chosen.copy();chosen['absolute_error']=chosen.residual.abs()
chosen.sort_values('absolute_error',ascending=False).to_csv(base/'carry_error_analysis.csv',index=False)
chosen.groupby('golfer').apply(lambda g:pd.Series({'n':len(g),'mae':g.residual.abs().mean(),'bias_actual_minus_predicted':g.residual.mean()}),include_groups=False).to_csv(base/'carry_per_golfer.csv')
fig,ax=plt.subplots(figsize=(6,5))
for golfer,g in chosen.groupby('golfer'):ax.scatter(g.actual,g.predicted,label=golfer,alpha=.8)
lo=min(chosen.actual.min(),chosen.predicted.min());hi=max(chosen.actual.max(),chosen.predicted.max())
ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='TrackMan carry (yd)',ylabel='Out-of-fold predicted carry (yd)',title='Video features + SVR: retrospective best within-session result')
ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'carry_predictions.png',dpi=160);plt.close(fig)
print('Saved paired uncertainty, 360 learning-curve fits, per-shot/golfer errors and three plots.')
