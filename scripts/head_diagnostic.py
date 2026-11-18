"""Face-landmark centroid projection, confidence mask and saved measured image coordinates."""
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
from scipy.signal import savgol_filter

sid='sid_1_01';data=json.loads((ROOT/'data/processed/pose'/f'{sid}.json').read_text())
p=np.array(data['landmarks'],dtype=float);head=p[:,[0,7,8],:2].mean(axis=1)
valid=(p[:,[0,7,8],3]>=.6).all(axis=1)
t=data['clip_start_playback_s']+np.arange(len(p))/data['fps']
smooth=savgol_filter(head,9,2,axis=0)
relative=smooth-smooth[0]
df=pd.DataFrame(dict(playback_s=t,x_image_fraction=head[:,0],y_image_fraction=head[:,1],
    smooth_relative_x=relative[:,0],smooth_relative_y=relative[:,1],confidence_valid=valid))
df.loc[~valid,['smooth_relative_x','smooth_relative_y']]=np.nan
out=ROOT/'outputs/head';out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/f'{sid}_trace.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(t,df.smooth_relative_x*100,label='horizontal')
axes[0].plot(t,df.smooth_relative_y*100,label='vertical, down positive')
axes[0].set(xlabel='Video playback seconds',ylabel='Change from clip start (% image width/height)')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
axes[1].plot(df.smooth_relative_x*100,df.smooth_relative_y*100)
axes[1].scatter(relative[0,0]*100,relative[0,1]*100,color='green',label='clip start')
axes[1].set(xlabel='Horizontal (% image width)',ylabel='Vertical (% image height)')
axes[1].invert_yaxis();axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.suptitle('Sid 1 shot 1: estimated face-landmark centroid; no gaze or metric-height inference')
fig.tight_layout();fig.savefig(out/f'{sid}_trace.png',dpi=150)
print('Saved estimated centroid trace; clip start is not an annotated address event.')
