"""Outcome-blind feature definitions; paired waveform visualization and descriptive associations."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,sosfiltfilt
from scipy.stats import spearmanr
if __name__=='__main__':
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').set_index('shot_id');l=pd.read_csv(ROOT/'data/manifests/trackman_labels.csv').set_index('Shot ID')
 pairs=[('lucas_1_03','lucas_1_04'),('rohan_2_01','rohan_2_02'),('eduardo_1_07','eduardo_1_08')]
 fig,ax=plt.subplots(3,3,figsize=(14,10),sharex=True);cache={}
 for col,pair in enumerate(pairs):
  for sid,color in zip(pair,['#247a88','#ce5937']):
   r=m.loc[sid]
   if r.club_candidate not in cache:cache[r.club_candidate]=pd.read_csv(ROOT/r.club_candidate)
   d=cache[r.club_candidate];t=d.t_ns.to_numpy()*1e-9-r.club_peak_candidate_s;keep=abs(t)<=.4;t=t[keep];w=d.loc[keep];g=w[['gx_rad_s','gy_rad_s','gz_rad_s']].to_numpy()
   norm=np.linalg.norm(g,axis=1);grid=np.arange(t[0],t[-1],1/180);gi=np.column_stack([np.interp(grid,t,g[:,i]) for i in range(3)])
   hp=sosfiltfilt(butter(3,20,btype='highpass',fs=180,output='sos'),gi,axis=0)
   label=f'{sid}: {l.loc[sid,"Carry flat (yd)"]:.0f} yd / {l.loc[sid,"Launch angle (deg)"]:.1f} deg'
   ax[0,col].plot(t,norm,color=color,label=label);ax[1,col].plot(grid,np.linalg.norm(hp,axis=1),color=color)
   ax[2,col].step(t,w.near_accel_limit+(.05 if color=='#ce5937' else 0),where='mid',color=color)
  ax[0,col].legend(fontsize=8);ax[2,col].set_xlabel('Seconds relative to gyro peak (not verified impact)')
 for row,title in enumerate(['Gyro norm (rad/s)','20-Hz high-pass gyro magnitude','Limit flag (orange offset +0.05)']):
  ax[row,0].set_ylabel(title)
  for a in ax[row]:a.grid(alpha=.2)
 fig.suptitle('Short-carry examples versus same-golfer normal shots\nSignals are measured; chunk/thin contact type is not labeled',fontsize=14)
 fig.tight_layout();out=ROOT/'outputs/contact_quality';out.mkdir(exist_ok=True);fig.savefig(out/'waveform_pairs.png',dpi=150);plt.close(fig)
 f=pd.read_csv(ROOT/'data/processed/contact_features.csv').merge(l.reset_index().rename(columns={'Shot ID':'shot_id'})[['shot_id','Carry flat (yd)','Launch angle (deg)']],on='shot_id').dropna(subset=['Carry flat (yd)'])
 rows=[]
 for name in [c for c in f if c.startswith('contact_')]:
  for target in ['Carry flat (yd)','Launch angle (deg)']:
   z=f[[name,target]].dropna();rho,p=spearmanr(z[name],z[target]);rows.append(dict(feature=name,target=target,n=len(z),spearman_rho=rho,unadjusted_p=p,interpretation='descriptive_only_multiple_comparisons_and_golfer_confounding'))
 pd.DataFrame(rows).to_csv(ROOT/'experiments/contact_quality_v1/descriptive_associations.csv',index=False)
 print('Saved waveform plot and descriptive associations')
