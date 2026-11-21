"""Causal terminal ball-motion checks in source pixels per video playback second."""
from math import hypot

MAX_STEP_S = .035
LAUNCH_WINDOW_S = .080


def truncate_motion(points, policy, launch_s=None):
    """Return the prefix and first rejected window, without interpolating gaps."""
    if policy not in {"none", "absolute", "relative", "either", "stall"}:
        raise ValueError(f"Unknown ball-motion policy {policy}")
    if len(points) < 3 or policy == "none":
        return list(points), None
    start = points[0]["playback_s"] if launch_s is None else launch_s
    peak = 0.
    for i in range(2, len(points)):
        a, b, c = points[i-2:i+1]
        gaps = [b["playback_s"]-a["playback_s"], c["playback_s"]-b["playback_s"]]
        if any(not 0 < gap <= MAX_STEP_S for gap in gaps):
            continue
        distance = hypot(c["x"]-a["x"], c["y"]-a["y"])
        speed = distance / sum(gaps)
        if c["playback_s"]-start <= LAUNCH_WINDOW_S:
            peak = max(peak, speed)
        threshold = {"absolute": 1000., "relative": .20*peak,
                     "either": max(1000., .20*peak), "stall": 180.}[policy]
        # A nearly stationary cluster is stronger evidence of loss than low
        # apparent speed alone. Require all three points within two pixels.
        stationary = max(distance, hypot(b["x"]-a["x"], b["y"]-a["y"]),
                         hypot(c["x"]-b["x"], c["y"]-b["y"])) <= 2.
        if speed < threshold and (policy != "stall" or stationary):
            return list(points[:i-1]), dict(
                first_dropped_s=b["playback_s"], confirmed_s=c["playback_s"],
                speed_px_s=speed, threshold_px_s=threshold, peak_px_s=peak
            )
    return list(points), None



def truncate_stationary_tail(points):
    """Stop the live trace after a tight cluster persists across native observations.

    This is a display rule for negligible image movement, independent of apparent
    speed ratios during flight. Short apex motion and missing frames are allowed.
    """
    for end in range(5, len(points)):
        last = points[end]
        start = end
        while start and last["playback_s"]-points[start]["playback_s"] < .045:
            if points[start]["playback_s"]-points[start-1]["playback_s"] > MAX_STEP_S:
                break
            start -= 1
        window = points[start:end+1]
        if len(window)<6 or last["playback_s"]-window[0]["playback_s"]<.045:
            continue
        anchor = window[0]
        radius = max(hypot(p["x"]-anchor["x"],p["y"]-anchor["y"]) for p in window)
        if radius <= 1.5:
            return list(points[:start+1]), dict(
                first_dropped_s=points[start+1]["playback_s"],
                confirmed_s=last["playback_s"], radius_px=radius
            )
    return list(points), None
