import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_ball_proposals import moving_prefix
from phase_candidates import propose
from early_swing import pose_features

class TrackingReviewTests(unittest.TestCase):
 def test_stationary_tail_is_removed(self):
  obs=[dict(x=0,y=y,playback_s=i*.01) for i,y in enumerate([100,90,80,70,70,70,70,70])]
  self.assertEqual(len(moving_prefix(obs)),4)
 def test_moving_track_is_preserved(self):
  obs=[dict(x=0,y=100-i*10,playback_s=i*.01) for i in range(10)]
  self.assertEqual(moving_prefix(obs),obs)
 def test_phase_abstains_when_wrists_hidden(self):
  a=np.zeros((105,33,4));a[:,:,3]=1;a[:,[15,16],3]=.1
  result=propose(dict(landmarks=a.tolist(),clip_start_playback_s=0,fps=30),2.5)
  self.assertIsNone(result['top_candidate_s']);self.assertIsNone(result['impact_s'])
 def test_pose_features_ignore_low_visibility_outlier(self):
  a=np.zeros((30,33,4));a[:,:,3]=1;a[10,[0,7,8],:2]=100;a[10,[0,7,8],3]=0
  f=pose_features(a,30)
  self.assertEqual(f['early_head_range_0'],0);self.assertEqual(f['early_head_speed_p95'],0)

if __name__=='__main__':unittest.main()
