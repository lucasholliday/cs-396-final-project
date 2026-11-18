"""Scientific preview of confidence-masked head traces across sessions."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
if __name__=='__main__':
 m=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv');fig,axes=plt.subplots(3,2,figsize=(12,9))
 for ax,r in zip(axes.flat,m.groupby('session',sort=False).head(1).itertuples()):
  d=json.loads((ROOT/f'data/processed/pose_observations/{r.shot_id}.json').read_text());f=pd.DataFrame(d['frames']);t=f.frame/30
  ax.plot(t,f.head_dx_image_pct,label='Horizontal / image width',color='#227851');ax.plot(t,f.head_dy_image_pct,label='Vertical / image height (down +)',color='#b46920');ax.axhline(0,color='gray',lw=.5);ax.set(title=r.shot_id,xlabel='Review clip seconds',ylabel='Change from opening reference (%)');ax.grid(alpha=.2)
 axes.flat[0].legend(fontsize=8);fig.suptitle('Estimated head centroid: image-plane movement, not gaze or physical height\nConfidence gaps are preserved; opening reference is not verified address',fontsize=13);fig.tight_layout(rect=(0,0,1,.94));out=ROOT/'outputs/head';out.mkdir(exist_ok=True);fig.savefig(out/'six_session_traces.png',dpi=150);print(out/'six_session_traces.png')
