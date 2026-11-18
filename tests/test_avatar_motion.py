import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from avatar_motion import complete

class AvatarMotion(unittest.TestCase):
 def fixture(self,n=30):
  world=np.full((n,33,3),np.nan);landmarks=np.zeros((n,33,4));return dict(estimated_world_landmarks=world.tolist(),landmarks=landmarks.tolist())
 def test_missing_pose_has_finite_display_fallback(self):
  d=complete(self.fixture());a=np.array(d['frames']);self.assertTrue(np.isfinite(a).all());self.assertTrue(np.array(d['inferred']).all());self.assertIn('excluded',d['semantics'])
 def test_leg_lengths_do_not_flicker_with_missing_observations(self):
  d=complete(self.fixture());a=np.array(d['frames'])
  for parent,child in [(23,25),(25,27),(24,26),(26,28)]:
   lengths=np.linalg.norm(a[:,child]-a[:,parent],axis=1);self.assertGreater(lengths.min(),.2);self.assertLess(np.ptp(lengths),1e-8)
 def test_shared_grip_and_source_not_mutated(self):
  p=self.fixture();before=np.array(p['estimated_world_landmarks']);d=complete(p);np.testing.assert_equal(np.array(p['estimated_world_landmarks']),before);a=np.array(d['frames']);self.assertLess(np.linalg.norm(a[:,15]-a[:,16],axis=1).max(),.03)
if __name__=='__main__':unittest.main()
