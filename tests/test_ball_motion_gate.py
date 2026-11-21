import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from ball_motion_gate import truncate_motion, truncate_stationary_tail
from benchmark_ball_motion_gate import fit_early_linear
from extend_ball_tracks import extend
from launch_geometry_features import fit


def track(xs, dt=.006):
    return [dict(x=x,y=900.,playback_s=(j+1)*dt,area=10) for j,x in enumerate(xs)]


class BallMotionGateTests(unittest.TestCase):
    def test_relative_and_floor_can_reject_slow_nonstationary_motion(self):
        points = track([0,60,120,125,130,135])
        kept,stop = truncate_motion(points,"either",0.)
        self.assertEqual(len(kept),3)
        self.assertLess(stop["speed_px_s"],1000)
        self.assertEqual(extend(points,[],.035,motion_policy="either",launch_s=0.),kept)
        self.assertIsNone(truncate_stationary_tail(points)[1])

    def test_stationary_display_gate_stops_before_stale_tail(self):
        points = track([0,15,30]+[30,30.2,29.9]*5)
        kept,stop = truncate_stationary_tail(points)
        self.assertEqual(len(kept),3)
        self.assertEqual(stop["first_dropped_s"],points[3]["playback_s"])

    def test_speed_window_cannot_cross_extended_gap(self):
        points = track([0,15,16])
        points[2]["playback_s"] += .08
        self.assertIsNone(truncate_motion(points,"absolute")[1])

    def test_early_estimators_ignore_entire_late_tail(self):
        seed = dict(x=0.,y=900.,playback_s=0.)
        points = track([2*j+15 for j in range(12)])
        late = [dict(x=4500.,y=50.,playback_s=.081,area=100)]
        for fn in (fit,fit_early_linear):
            self.assertEqual(fn(points,seed,.080),fn(points+late,seed,.080))


if __name__ == "__main__":
    unittest.main()
