"""Historical candidate-output contract; does NOT establish ball or club identity.
The former tests compared image angles to radar direction and were invalid as
tracking ground truth. Independent frame annotations are still required.
"""
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from video_launch_tracker import extract_shot_launch

class VideoLaunchTrackingTests(unittest.TestCase):
    def test_development_examples_have_finite_ordered_candidates(self):
        for sid in ['lucas_1_02','lucas_1_09','lucas_1_10','lucas_1_12']:
            with self.subTest(shot=sid):
                result=extract_shot_launch(sid)
                self.assertIsNotNone(result)
                track=result['ball_track']
                self.assertGreaterEqual(len(track),3)
                self.assertTrue(np.isfinite([[p['x'],p['y'],p['playback_s']] for p in track]).all())
                self.assertTrue((np.diff([p['playback_s'] for p in track])>0).all())
                self.assertTrue(np.isfinite(result['ball_angle_deg']))

if __name__=='__main__':unittest.main()
