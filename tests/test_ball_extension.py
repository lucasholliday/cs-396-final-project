import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extend_ball_tracks import extend


def point(t):
    return dict(playback_s=t, x=700+180*t, y=710-400*t,area=7)


def frame(t, *points):
    return dict(playback_s=t,frame=round(t*1000),candidates=points)


class BallExtensionTests(unittest.TestCase):
    def setUp(self):
        self.prefix = [point(t/1000) for t in range(0,31,6)]

    def test_confirmed_reacquisition_after_longer_gap(self):
        frames = [frame(t,point(t)) for t in [.080,.086,.092]]
        self.assertEqual(extend(self.prefix,frames,.035), self.prefix)
        result = extend(self.prefix,frames,.070)
        self.assertEqual(result[:len(self.prefix)], self.prefix)
        self.assertEqual([p["playback_s"] for p in result[-3:]], [.080,.086,.092])

    def test_does_not_commit_single_plausible_hit(self):
        frames = [frame(.05,point(.05)),frame(.056),frame(.062)]
        self.assertEqual(extend(self.prefix,frames,.070),self.prefix)

    def test_does_not_follow_stationary_repeat(self):
        frozen = point(.030)
        frames = [frame(t,frozen) for t in [.048,.054,.060]]
        self.assertEqual(extend(self.prefix,frames,.070),self.prefix)


if __name__ == "__main__":
    unittest.main()
