"""Compare longer shaft edges in higher-resolution video against two background samples."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/native_shaft_highres'


def shaft_candidates(img, grip, backgrounds, detector):
    lines = detector.detect(img)[0]
    if lines is None:
        return []
    lines = lines.reshape(-1, 2, 2)
    offsets = np.linalg.norm(lines - grip, axis=2)
    reverse = offsets[:, 0] > offsets[:, 1]
    lines[reverse] = lines[reverse, ::-1]
    a, b = lines[:, 0], lines[:, 1]
    lengths = np.linalg.norm(b-a, axis=1)
    vectors = (b-a) / lengths[:, None]
    dx, dy = (a-grip).T
    perp = np.abs(dx*vectors[:, 1] - dy*vectors[:, 0])
    near = offsets.min(axis=1)
    idx = np.where((near < 115) & (perp < 34) & (lengths > 43)
                   & (np.sum((a-grip)*vectors, axis=1) > -20))[0]
    options = []
    for j in idx:
        x, y = np.rint(np.linspace(a[j], b[j], 32)[4:-2]).astype(int).T
        overlap = min(np.mean(bg[y, x] < 2.5) for bg in backgrounds)
        if overlap > .75:
            continue
        score = min(lengths[j], 220)/105 - perp[j]/23 - near[j]/150 - overlap
        if score > 0:
            options.append((score, vectors[j]))
    return sorted(options, key=lambda item: -item[0])


def run(shot):
    pose = json.loads((ROOT / f'data/processed/pose/{shot}.json').read_text())
    trace = json.loads((ROOT / f'data/processed/native_swing_shaft/{shot}.json').read_text())
    source = pd.read_csv(ROOT/'data/manifests/shot_manifest.csv').set_index('shot_id').loc[shot, 'video_candidate']
    cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-ss', str(trace['start']), '-i', str(ROOT/source),
           '-t', '3.5', '-an', '-filter_threads', '1', '-vf', 'scale=540:960', '-vsync', '0',
           '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    frames = np.frombuffer(subprocess.check_output(cmd), np.uint8).reshape(-1,960,540)
    assert len(frames) == len(trace['times'])
    hands = np.array(pose['landmarks'])[:, [15,16], :2].mean(axis=1) * (540,960)
    ts = np.arange(len(hands)) / pose['fps']
    grips = np.column_stack([np.interp(trace['times'],ts,axis) for axis in hands.T])
    backgrounds = []
    for chunk in (frames[:len(frames)//3:12], frames[-len(frames)//3::12]):
        bg = np.median(chunk,axis=0).astype(np.uint8)
        backgrounds.append(cv2.distanceTransform(255-cv2.Canny(bg,40,100),cv2.DIST_L2,3))
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE,.8)
    raw = [shaft_candidates(im, g, backgrounds, detector)[:4] for im, g in zip(frames,grips)]
    result = []
    for i, options in enumerate(raw):
        best_score = 0
        for score, vec in options:
            support = 0
            for j in range(max(0,i-3),min(len(raw),i+4)):
                if i != j and abs(trace['times'][j]-trace['times'][i]) < .025:
                    support += any(np.dot(other,vec) > .97 for _,other in raw[j])
            if support >= 2 and score + .12*support > best_score:
                best_score = score + .12*support
                best = vec
        if best_score:
            result.append(dict(source_frame=i, dir_deg=np.degrees(np.arctan2(*best[::-1])).item()))
    pd.DataFrame(result).to_csv(OUT/f'{shot}.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shots', nargs='+')
    OUT.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    for shot in parser.parse_args().shots:
        run(shot)
