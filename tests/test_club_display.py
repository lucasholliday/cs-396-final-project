import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from club_display import complete

class ClubDisplayTest(unittest.TestCase):
    def fixture(self):
        w=np.random.default_rng(42).normal(0,.2,(8,33,3));lm=np.ones((8,33,4));lm[:,:,:2]=(w[:,:,:2]*200+[270,480])/[540,960]
        p=dict(estimated_world_landmarks=w.tolist(),landmarks=lm.tolist())
        r=dict(image_size=[540,960],maximum_interpolation_gap_frames=3,occluded_frames=[2],points=[dict(frame=i,grip=[200,400],head=[200+dx,400+dy]) for i,dx,dy in [(0,80,120),(3,120,-80),(7,100,40)]])
        return p,r
    def test_no_long_gap_extrapolation_or_occlusion_fill(self):
        p,r=self.fixture();v=complete(p,r)
        self.assertIsNotNone(v['frames'][1]);self.assertIsNone(v['frames'][2])
        self.assertTrue(all(v['frames'][i] is None for i in [4,5,6]))
    def test_depth_alternatives_preserve_image_evidence_and_inputs(self):
        p,r=self.fixture();before=copy.deepcopy((p,r));v=complete(p,r);a=np.array(v['projection'])
        self.assertEqual((p,r),before)
        for f in filter(None,v['frames']):
            delta=np.array(f['head'])-f['grip']
            for key in ['direction','alternate_direction']:
                direction=np.array(f[key]);self.assertAlmostEqual(np.linalg.norm(direction),1)
                np.testing.assert_allclose(a@(direction*.85),delta,atol=1e-8)
if __name__=='__main__':unittest.main()
