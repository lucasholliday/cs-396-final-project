"""Software projection preview of scene geometry, explicitly not WebGL/browser QA."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
out=ROOT/'outputs/range_scene_preview'
fig,axes=plt.subplots(1,3,figsize=(15,5),facecolor='#bddbe9')
for ax,name,title in zip(axes,['address','swing','flight'],['Completed golfer · address','Completed golfer · swing','Illustrative flight + tracer']):
 d=json.loads((out/f'{name}.json').read_text());polys=d['polygons'];ax.set_facecolor('#bddbe9');ax.add_collection(PolyCollection([p['xy'] for p in polys],facecolors=[p['color'] for p in polys],edgecolors='none',antialiaseds=False));lines=d['lines'];ax.add_collection(LineCollection([p['xy'] for p in lines],colors=[p['color'] for p in lines],linewidths=.6));ax.set(xlim=(-1,1),ylim=(-1,1),title=title);ax.axis('off')
fig.suptitle('Scene-geometry preview — software projection, not browser/WebGL verification',fontsize=12);fig.tight_layout();fig.savefig(out/'range_preview.png',dpi=150,facecolor=fig.get_facecolor());print(out/'range_preview.png')
