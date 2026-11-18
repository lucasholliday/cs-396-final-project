import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pose_observations import masked_smooth,observations
from ball_tracking_v6 import candidates
from ball_timing_study import slope_features
import pandas as pd

class VisualObservations(unittest.TestCase):
    def test_smoothing_does_not_bridge_missing_run(self):
        x=np.array([[0.,0.],[0.,0.],[100.,100.],[50.,50.],[50.,50.]])
        y=masked_smooth(x,np.array([True,True,False,True,True]))
        self.assertTrue(np.isnan(y[2]).all());np.testing.assert_array_equal(y[:2],0);np.testing.assert_array_equal(y[3:],50)
    def test_no_late_reference_invented(self):
        a=np.zeros((20,33,4));a[12:,:,3]=1
        d=observations(dict(landmarks=a.tolist(),fps=30,clip_start_playback_s=0))
        self.assertIn('unavailable',d['reference']);self.assertTrue(np.isnan(d['frames'][15]['head_dx_image_pct']))
    def test_positive_motion_rejects_dark_ghost(self):
        b=np.full((40,40),150,np.uint8);f=b.copy();f[5:10,5:10]=20;f[20:25,20:25]=230
        cfg=dict(positive_difference=22,brightness_minimum=110,minimum_component_area=3,maximum_component_area=850,maximum_component_width=65,maximum_component_height=140)
        cs=candidates(f,b,cfg);self.assertEqual(len(cs),1);self.assertAlmostEqual(cs[0]['x'],22);self.assertAlmostEqual(cs[0]['y'],22)
    def test_dual_polarity_recovers_silhouette(self):
        b=np.full((40,40),200,np.uint8);f=b.copy();f[5:10,5:10]=50
        cfg=dict(dual_polarity=True,positive_difference=22,brightness_minimum=110,minimum_component_area=3,maximum_component_area=850,maximum_component_width=65,maximum_component_height=140)
        cs=candidates(f,b,cfg);self.assertEqual(len(cs),1);self.assertEqual(cs[0]['polarity'],-1)
    def test_interframe_slope_is_time_origin_invariant(self):
        p=pd.DataFrame(dict(playback_s=[1.,1.05,1.1],x=[100,110,120],y=[500,450,400]))
        a=slope_features(p);p.playback_s+=100;b=slope_features(p)
        for key in a:self.assertAlmostEqual(a[key],b[key],places=8)
if __name__=='__main__':unittest.main()

