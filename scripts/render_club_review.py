"""Render source-bound assisted club anchors for inspection, without modifying originals."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/club_review';out.mkdir(exist_ok=True)
for source in (ROOT/'annotations').glob('club_keypoints_*.json'):
    review=json.loads(source.read_text());sid=review['shot_id'];video=ROOT/f'outputs/pose/{sid}_input.mp4'
    assert hashlib.sha256(video.read_bytes()).hexdigest()==review['input_sha256']
    cap=cv2.VideoCapture(str(video));tiles=[]
    for mark in review['points']:
        cap.set(cv2.CAP_PROP_POS_FRAMES,mark['frame']);ok,frame=cap.read();assert ok
        cv2.line(frame,tuple(mark['grip']),tuple(mark['head']),(255,255,0),2)
        cv2.putText(frame,f"{sid} f{mark['frame']}",(8,32),0,.65,(0,240,255),2)
        tiles.append(cv2.resize(frame,(270,480)))
    cap.release()
    while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
    for i in range(0,len(tiles),8):
        target=out/f'{sid}_{i//8}.jpg'
        cv2.imwrite(str(target),np.vstack([np.hstack(tiles[j:j+4]) for j in range(i,min(i+8,len(tiles)),4)]));print(target)
