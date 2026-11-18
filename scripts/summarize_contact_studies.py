"""Persist reports and uncertainty from executed contact/threshold experiments."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'));os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def markdown(df):
 cols=df.columns.tolist();lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(round(x,2)) if isinstance(x,float) else str(x) for x in row)+' |')
 return '\n'.join(lines)

if __name__=='__main__':
 out=ROOT/'outputs/contact_quality';out.mkdir(exist_ok=True)
 t=pd.read_csv(ROOT/'experiments/carry_thresholds_v1/metrics.csv');p=pd.read_csv(ROOT/'experiments/carry_thresholds_v1/predictions.csv')
 assert not p.duplicated(['features','threshold_yd','shot_id']).any()
 assert (p.accepted==(p.low_probability<.5)).all()
 assert np.allclose(p.gated_prediction,np.where(p.accepted,p.high_prediction,p.low_prediction))
 assert (t.correctly_flagged_below_n+t.correctly_accepted_above_n+t.false_accepted_below_n+t.false_rejected_above_n==t.n).all()
 # Paired golfer-cluster bootstrap of the all-shot gating delta, descriptive only.
 rng=np.random.default_rng(396);uncert=[]
 for (name,threshold),g in p.groupby(['features','threshold_yd']):
  groups=sorted(g.golfer.unique());delta=abs(g.actual-g.gated_prediction)-abs(g.actual-g.allshot_prediction);samples=[]
  for _ in range(2000):
   chosen=rng.choice(groups,len(groups),replace=True);samples.append(np.concatenate([delta[g.golfer==x].to_numpy() for x in chosen]).mean())
  low,high=np.quantile(samples,[.025,.975]);uncert.append(dict(features=name,threshold_yd=threshold,gate_minus_baseline_mae=float(delta.mean()),ci_low=low,ci_high=high,golfer_clusters=len(groups)))
 pd.DataFrame(uncert).to_csv(ROOT/'experiments/carry_thresholds_v1/paired_uncertainty.csv',index=False)
 table=t[['features','threshold_yd','n','true_above_n','oracle_above_mae','oracle_above_constant_mae','accepted_n','accepted_mae','false_accepted_below_n','false_rejected_above_n','allshot_mae','gated_all_mae']]
 report='''# Carry thresholds and separate short-shot flags — 2026-11-18

Tested 80, 100 and 120 yards with leave-golfer-out evaluation. Four feature/cohort variants were evaluated at each cutoff. Model families are chosen using only the outer training golfers via inner grouped folds. The gate is also trained only on training golfers; its below-threshold probability is compared with a fixed 0.5 cutoff. All out-of-fold predictions and decisions are saved. These are exploratory tests on a repeatedly studied dataset, not a new untouched test set.

**A carry cutoff is not a contact-quality label.** A solid slower seven-iron can carry under 100 yards, and a badly directed or thin shot can exceed it. This study calls the outcomes above/below threshold; it does not infer confirmed chunks, thin strikes or hosel contact.

## What the columns mean
- `oracle_above_mae`: model trained on above-threshold training shots, scored only on test shots whose actual carry exceeds the cutoff. This assumes the qualifying category is already known.
- `oracle_above_constant_mae`: training-above-threshold mean on exactly the same qualifying test shots. Necessary because narrowing the range alone lowers MAE.
- `accepted_mae`: high-distance model error on the shots the classifier actually accepts, including mistakes. The system may flag rejected shots rather than output a distance.
- `gated_all_mae`: all-shot error if below/above-specific regressors are routed by the predicted gate. No test shot is removed. `allshot_mae` is the corresponding unfiltered regression comparator trained/evaluated on the same cohort.
- Reviewed-ball features use approximate AI visual point annotations and five coarse timing corrections; they estimate assisted-tracking potential, not current automatic-tracker accuracy. No unreviewed detector trajectories enter these models.

## Results
'''+markdown(table)+'''

## Interpretation
At 100 yards, 44 of the 72 labeled shots qualify. Video gives 11.98 yd conditional MAE; video + watch 12.86. The conditional training-mean baseline is 11.98 yd: much of the apparently impressive accuracy comes from restricting the outcome range. The data do not show accurate within-category distance discrimination beyond that simple baseline.

Operational classification is substantially harder. With video + watch at 100 yards, only 24/72 shots are accepted; seven of those are actually below 100 and 27 qualifying shots are rejected. Accepted MAE is 31.74 yd. Hard routing across all shots worsens MAE from 29.87 to 48.61 yd.

On the separate 31-shot reviewed-ball cohort, 17 actually exceed 100 yards. The gate accepts 17: 14 correct accepts, three false accepts; it rejects three qualifying shots and correctly flags 11 shorter shots. Accepted MAE is 20.00 yd, compared with 13.07 yd if qualifying membership is known perfectly. All-shot hard routing is 25.65 yd versus 24.78 for direct reviewed-ball regression. Thus an automatic distance upgrade has not been demonstrated.

At 80 yards, assisted-ball hard routing is 23.61 yd versus 24.78 direct; video routing is 29.65 versus 31.93. These small exploratory differences should not determine the cutoff. At 120 yards, the qualifying sample shrinks, the constant baseline becomes even stronger, and several gates reject many legitimate qualifying shots. Group-cluster uncertainty is persisted; five golfers and multiple comparisons limit confidence.

A reasonable product experiment is to provide a short-shot flag and report distance only when the model has enough evidence, showing acceptance rate and false acceptance explicitly. Choose a cutoff based on the intended seven-iron use case and golfer population, then freeze it before new-day/new-golfer testing. Do not call every sub-100-yard shot a mishit or present 12-yard conditional error as end-to-end accuracy.
'''
 (ROOT/'reports/carry_threshold_results.md').write_text(report)
 fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for ax,name in zip(axs,['video_watch','reviewed_ball_assisted']):
  z=t[t.features==name].sort_values('threshold_yd')
  for col,label in [('oracle_above_mae','Known qualifying category'),('oracle_above_constant_mae','Conditional training mean'),('accepted_mae','Predicted qualifying category')]:ax.plot(z.threshold_yd,z[col],marker='o',label=label)
  ax.set_title(name.replace('_',' '));ax.set_xlabel('Carry cutoff (yd)');ax.set_ylabel('MAE (yd)');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Conditional accuracy does not include classification mistakes');fig.tight_layout();fig.savefig(out/'carry_thresholds.png',dpi=160);plt.close(fig)
 # Visual comparison uses reviewed locations; no positions are invented after loss.
 review=json.loads((ROOT/'annotations/contact_ball_points_review.json').read_text());tiles=[]
 for sid,label in [('lucas_1_07','146 yd carry / 16.5 deg launch'),('lucas_1_10','16 yd carry / 2.4 deg launch')]:
  points={int(a[0]):a[1:] for a in review['points'][sid]};row=[]
  for panel in [1,2,3]:
   im=cv2.imread(str(ROOT/f'outputs/contact_video_review/{sid}_{panel}_native.png'));im=cv2.resize(im,(390,550))
   if panel in points:
    x,y=points[panel];cv2.circle(im,(round(x),round(y)),15,(0,220,255),2)
   else:cv2.putText(im,'Ball not confidently localized',(8,55),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
   cv2.putText(im,sid+' | '+label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1);row.append(im)
  tiles.append(np.hstack(row))
 cv2.imwrite(str(out/'normal_vs_low_flight.jpg'),np.vstack(tiles))
 print('Saved threshold report, bootstrap, plots and reviewed flight comparison')
