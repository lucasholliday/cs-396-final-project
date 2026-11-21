"""Encode native-cadence review clips with checked source/output frame timing."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def encode(row, version, out):
    candidate = json.loads((ROOT/f"data/processed/ball_v{version}/{row.shot_id}.json").read_text())
    cfg = candidate["configuration"]
    meta = out / f"{row.shot_id}.json"
    target = out / f"{row.shot_id}.mp4"
    if meta.exists() and target.exists():
        old = json.loads(meta.read_text())
        if old["candidate_signature"] == candidate["signature"] and old.get("encoding_version") == 2:
            return len(old["source_pts"])
    start = row.video_peak_candidate_s - cfg["before_peak_s"]
    x,y,w,h = cfg["native_crop"]
    temp = target.with_suffix(".tmp.mp4")
    try:
        subprocess.run([
            "ffmpeg","-v","error","-ss",str(start),"-i",str(ROOT/row.video_candidate),
            "-frames:v",str(len(candidate["frames"])),"-an","-vf",f"crop={w}:{h}:{x}:{y}",
            "-vsync","0","-enc_time_base","-1","-video_track_timescale","60000","-c:v","libx264",
            "-bf","0","-use_editlist","0","-threads","2","-crf","20","-pix_fmt","yuv420p",
            "-movflags","+faststart","-y",str(temp)
        ],capture_output=True,check=True)
        info = json.loads(subprocess.check_output([
            "ffprobe","-v","error","-select_streams","v:0","-show_entries",
            "frame=best_effort_timestamp_time","-of","json",str(temp)
        ]))
        review_pts = [float(f["best_effort_timestamp_time"]) for f in info["frames"]]
        source_pts = [f["playback_s"] for f in candidate["frames"]]
        assert len(review_pts) == len(source_pts), (row.shot_id,len(review_pts),len(source_pts))
        cadence_error = float(np.max(abs(np.diff(review_pts)-np.diff(source_pts))))
        assert cadence_error < .0001, (row.shot_id,cadence_error)
        data = dict(encoding_version=2,shot_id=row.shot_id,source=row.video_candidate,
                    candidate_signature=candidate["signature"],review_pts=review_pts,
                    source_pts=source_pts,crop=cfg["native_crop"],cadence_error_max_s=cadence_error,
                    semantics="Source cadence checked against candidate decode by count and inter-frame PTS.")
        temp.replace(target)
        meta.write_text(json.dumps(data))
    finally:
        temp.unlink(missing_ok=True)
    return len(review_pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--shots", nargs="+")
    args = parser.parse_args()
    version = 11 if args.wide else 7
    out = ROOT / ("outputs/native_review_wide" if args.wide else "outputs/native_review")
    out.mkdir(exist_ok=True)
    manifest = pd.read_csv(ROOT/"data/manifests/shot_manifest.csv")
    if args.shots:
        manifest = manifest[manifest.shot_id.isin(args.shots)]
        if len(manifest) != len(set(args.shots)):
            parser.error("Unknown shot ID")
    records = []
    for row in manifest.itertuples():
        count = encode(row,version,out)
        records.append(dict(shot_id=row.shot_id,frames=count,status="verified_frame_count"))
        print(row.shot_id,count,flush=True)
    pd.DataFrame(records).to_csv(out/"verification.csv",index=False)


if __name__ == "__main__":
    main()
