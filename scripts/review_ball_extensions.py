"""Build contact sheets from the same native decode as the extension candidates."""
import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def review(shot, variant, row, stop_index=None):
    processed = ROOT / "data/processed"
    result = json.loads((processed / f"ball_extension_{variant}/{shot}.json").read_text())
    obs = result["observations"]
    if not obs:
        return
    prefix = result["prefix_length"]
    cut = min(prefix, len(obs))
    ids = {0, cut//2, cut-1}
    if len(obs)>cut:
        ids.update(np.linspace(cut,len(obs)-1,4,dtype=int))
    if stop_index is not None:
        cut = stop_index
        ids = {i for i in (cut-5,cut-2,cut-1,cut,cut+1,cut+4,cut+10,cut+20) if 0<=i<len(obs)}
    points = [obs[i] for i in sorted(ids)]
    cfg = json.loads((ROOT/"configs/ball_tracking_v11.json").read_text())
    x,y,w,h = cfg["native_crop"]
    start = row.video_peak_candidate_s-cfg["before_peak_s"]
    proc = subprocess.run(
        ["ffmpeg","-v","info","-ss",str(start),"-i",str(ROOT/row.video_candidate),
         "-t",str(cfg["duration_s"]),"-an","-vf",f"crop={w}:{h}:{x}:{y},showinfo",
         "-vsync","0","-pix_fmt","gray","-f","rawvideo","-"], capture_output=True, check=True)
    frames = np.frombuffer(proc.stdout,np.uint8).reshape(-1,h,w)
    raw_pts = re.findall(r"\bn:\s*\d+.*?pts_time:([\d.\-]+)",proc.stderr.decode())
    pts = np.array(raw_pts[:len(frames)],float)+start
    tiles=[]
    for i, point in zip(sorted(ids),points):
        index = int(np.argmin(abs(pts-point["playback_s"])))
        if abs(pts[index]-point["playback_s"])>.001:
            raise ValueError(f"Frame mismatch: {shot} {i}")
        u,v=int(round(point["x"]-x)),int(round(point["y"]-y))
        padded=np.pad(frames[index],60)
        patch=padded[v:v+120,u:u+120]
        patch=cv2.resize(patch,(240,240),interpolation=cv2.INTER_NEAREST)
        tile=np.zeros((270,240,3),np.uint8)
        tile[:240]=cv2.cvtColor(patch,cv2.COLOR_GRAY2BGR)
        cv2.circle(tile,(120,120),13,(80,215,255),1)
        tag=("dropped" if i>=cut else "kept") if stop_index is not None else ("added" if i>=cut else "baseline")
        cv2.putText(tile,f"{tag} {i+1}  y={point['y']:.1f}",(5,260),
                    cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        tiles.append(tile)
    while len(tiles)%4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    out=ROOT/("outputs/ball_motion_review" if stop_index is not None else "outputs/ball_extension_review")
    out.mkdir(exist_ok=True)
    assert cv2.imwrite(str(out/f"{shot}_{variant}.jpg"),sheet)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("shots",nargs="+")
    parser.add_argument("--variant",default="wide70")
    parser.add_argument("--stop-index", type=int)
    args=parser.parse_args()
    manifest=pd.read_csv(ROOT/"data/manifests/shot_manifest.csv").set_index("shot_id")
    for shot in args.shots:
        review(shot,args.variant,manifest.loc[shot],args.stop_index)
