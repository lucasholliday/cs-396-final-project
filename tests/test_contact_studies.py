import sys,unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from contact_features import extract_window
from carry_threshold_experiment import fit_selected
from contact_benchmark import probabilities
class ContactStudyTests(unittest.TestCase):
 def test_gyro_descriptors_are_rotation_invariant(self):
  t=np.arange(0,2,.005);g=np.column_stack([np.sin(t*15),np.cos(t*9),np.sin(t*30)])*10
  d=pd.DataFrame(dict(t_ns=t*1e9,gx_rad_s=g[:,0],gy_rad_s=g[:,1],gz_rad_s=g[:,2],ax_m_s2=0,ay_m_s2=0,az_m_s2=9.81,near_accel_limit=0,near_gyro_limit=0))
  cfg=json.loads((ROOT/'configs/contact_quality.json').read_text());a,_=extract_window(d,1,cfg)
  q=np.array([[0,-1,0],[1,0,0],[0,0,1]]);d[['gx_rad_s','gy_rad_s','gz_rad_s']]=g@q
  b,_=extract_window(d,1,cfg)
  for k in a:
   if k.startswith('contact_gyro_'):self.assertAlmostEqual(a[k],b[k],places=7)
 def test_clipped_acceleration_is_not_used_as_true_jerk(self):
  t=np.arange(0,2,.005);d=pd.DataFrame(dict(t_ns=t*1e9,gx_rad_s=np.sin(t),gy_rad_s=0,gz_rad_s=0,ax_m_s2=999,ay_m_s2=999,az_m_s2=999,near_accel_limit=1,near_gyro_limit=0))
  f,_=extract_window(d,1,json.loads((ROOT/'configs/contact_quality.json').read_text()))
  self.assertTrue(np.isnan(f['contact_accel_unclipped_jerk_p95']));self.assertEqual(f['contact_accel_unclipped_pair_fraction'],0)
 def test_single_class_gate_has_valid_probabilities(self):
  d=pd.DataFrame(dict(golfer=['a']*3,x=[1,2,3]));cfg=json.loads((ROOT/'configs/carry_thresholds.json').read_text())
  p,_,_=fit_selected(d,['x'],np.zeros(3,dtype=int),True,cfg)
  np.testing.assert_array_equal(probabilities(p,d[['x']]),[0,0,0])
 def test_new_predictions_unique(self):
  for folder,keys in [('contact_quality_v1',['scheme','task','features','shot_id']),('reviewed_ball_v1',['cohort','task','features','shot_id'])]:
   d=pd.read_csv(ROOT/'experiments'/folder/'predictions.csv');self.assertFalse(d.duplicated(keys).any());self.assertTrue(np.isfinite(d.predicted).all())
 def test_threshold_gate_retains_all_test_shots(self):
  d=pd.read_csv(ROOT/'experiments/carry_thresholds_v1/predictions.csv')
  self.assertFalse(d.duplicated(['features','threshold_yd','shot_id']).any())
  self.assertTrue((d.accepted==(d.low_probability<.5)).all())
  np.testing.assert_allclose(d.gated_prediction,np.where(d.accepted,d.high_prediction,d.low_prediction))
  expected={'video':72,'video_watch':72,'video_watch_club_contact':65,'reviewed_ball_assisted':31}
  for (name,_),g in d.groupby(['features','threshold_yd']):self.assertEqual(len(g),expected[name])
if __name__=='__main__':unittest.main()
