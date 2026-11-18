"""Contact sheets for explicit human/agent review; does not grant eligibility."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/review';out.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
for k in range(0,len(df),12):
    images=[]
    for row in df.iloc[k:k+12].itertuples():
        p=ROOT/'outputs/pose'/f'{row.shot_id}_contact.jpg'
        im=cv2.imread(str(p))
        if im is None: raise ValueError(f'Missing contact: {p}')
        cv2.putText(im,row.shot_id,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        images.append(im)
    if len(images)%2:images.append(np.zeros_like(images[0]))
    cv2.imwrite(str(out/f'page_{k//12+1}.jpg'),np.vstack([np.hstack(images[j:j+2]) for j in range(0,len(images),2)]))
