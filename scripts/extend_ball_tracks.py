"""Evaluate confirmed extensions of a frozen launch prefix on cached native frames.

All times are source-video playback times. No outcome labels are read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ball_motion_gate import truncate_motion

ROOT = Path(__file__).resolve().parents[1]
CROPS = {"narrow35": (7, .035), "wide35": (11, .035), "wide70": (11, .070)}


def candidate_options(frame, history):
    recent = history[-6:]
    points = np.array([[p["x"], p["y"]] for p in recent])
    time = np.array([p["playback_s"] for p in recent])
    velocity = np.linalg.lstsq(
        np.column_stack((time-time[-1], np.ones(len(time)))), points, rcond=None
    )[0][0]
    dt = frame["playback_s"] - time[-1]
    predicted = points[-1] + velocity * dt
    tolerance = min(22., 4. + np.linalg.norm(velocity) * dt * .30)
    last_area = recent[-1]["area"]
    options = []
    for point in frame["candidates"]:
        xy = np.array([point["x"], point["y"]])
        error = np.linalg.norm(xy - predicted)
        ratio = point["area"] / last_area
        if error > tolerance or not .22 <= ratio <= 4.:
            continue
        # A stationary edge detection must not restart a moving track.
        displacement = xy - points[-1]
        if np.dot(displacement, velocity) <= 0 or np.linalg.norm(displacement) < 1.:
            continue
        score = error / tolerance + .18 * abs(np.log(ratio))
        options.append((score, {**point, "frame": frame["frame"],
                                "playback_s": frame["playback_s"]}))
    return sorted(options, key=lambda item: item[0])[:3]


def extend(prefix, frames, max_gap, *, motion_policy="none", launch_s=None):
    track, stop = truncate_motion(prefix,motion_policy,launch_s)
    if stop:
        return track
    if len(prefix) < 5:
        return list(prefix)
    for i, frame in enumerate(frames):
        gap = frame["playback_s"] - track[-1]["playback_s"]
        if gap <= .001:
            continue
        if gap > max_gap:
            break
        proposals = []
        for score, first in candidate_options(frame, track):
            path = [first]
            # Confirm across native frames before committing any point.
            for later in frames[i+1:i+5]:
                if later["playback_s"] - path[-1]["playback_s"] > .024:
                    break
                options = candidate_options(later, track + path)
                if options:
                    cost, point = options[0]
                    score += cost
                    path.append(point)
                if len(path) == 3:
                    proposals.append((score, path))
                    break
        proposals.sort(key=lambda item: item[0])
        if proposals and (len(proposals) == 1 or proposals[1][0] - proposals[0][0] >= .25):
            track.extend(proposals[0][1])
            if motion_policy != "none":
                track, stop = truncate_motion(track,motion_policy,launch_s)
                if stop:
                    break
    return track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", nargs="+")
    args = parser.parse_args()
    out = ROOT / "experiments/ball_extension_v1"
    out.mkdir(exist_ok=True)
    processed = ROOT / "data/processed"
    refs = pd.read_csv(ROOT / "annotations/contact_ball_points.csv")
    stats, checks = [], []
    shot_ids = sorted(pd.read_csv(ROOT / "data/manifests/shot_manifest.csv").shot_id)
    if args.shots:
        if set(args.shots) - set(shot_ids):
            parser.error("Unknown shot ID")
        shot_ids = sorted(set(args.shots))
    for sid in shot_ids:
        path = processed / "ball_launch_tracks" / f"{sid}.json"
        baseline = json.loads(path.read_text()) if path.exists() else {"seed": None, "observations": []}
        prefix = baseline["observations"]
        variants = {"baseline": prefix}
        for name, (version, gap) in CROPS.items():
            source = processed / f"ball_v{version}" / path.name
            candidates = json.loads(source.read_text())
            # Frame indices from different decodes need not correspond. Link by playback time.
            variants[name] = extend(prefix, candidates["frames"], gap)
            folder = processed / f"ball_extension_{name}"
            folder.mkdir(exist_ok=True)
            (folder / path.name).write_text(json.dumps({
                "shot_id": sid, "seed": baseline["seed"], "observations": variants[name],
                "status": "unvalidated extension", "prefix_length": len(prefix),
                "candidate_signature": candidates["signature"],
            }))
        for name, observations in variants.items():
            times = np.array([p["playback_s"] for p in observations])
            stats.append(dict(shot_id=sid, variant=name, points=len(times),
                              added=len(times)-len(prefix),
                              above_old_crop=sum(p["y"] < 650 for p in observations),
                              end_y=observations[-1]["y"] if observations else None,
                              span_s=float(np.ptp(times)) if len(times) else 0))
            for ref in refs[refs.shot_id == sid].itertuples():
                error = float("inf")
                if len(times):
                    i = np.argmin(abs(times-ref.playback_s))
                    if abs(times[i]-ref.playback_s) <= .006:
                        p = observations[i]
                        error = float(np.hypot(p["x"]-ref.x, p["y"]-ref.y))
                checks.append(dict(shot_id=sid, variant=name, panel=ref.panel,
                                   error_px=error, within16=error <= 16, within32=error <= 32))
    stats = pd.DataFrame(stats)
    checks = pd.DataFrame(checks, columns=["shot_id","variant","panel","error_px","within16","within32"])
    stats.to_csv(out / "tracks.csv", index=False)
    checks.to_csv(out / "reference_errors.csv", index=False)
    print(stats.groupby("variant").agg(points=("points", "sum"), added=("added", "sum"),
            shots_extended=("added", lambda a: (a > 0).sum()), above_crop=("above_old_crop", "sum")))
    if not checks.empty:
        print(checks.groupby("variant")[["within16", "within32"]].agg(["sum","count"]))


if __name__ == "__main__":
    main()
