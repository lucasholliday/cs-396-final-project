"""Synthetic geometry and held-out evaluation checks; not ball-identity validation."""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launch_geometry_features import fit
from audited_launch_features import features
from carry_estimators import QualityRoutedRegressor, SoftCarryRegressor

class AuditedLaunchTests(unittest.TestCase):
    def motion(self, end=.14):
        seed=dict(x=500.,y=1300.,playback_s=1.)
        track=[]
        for t in np.arange(.008,end,.006):
            z=(t+.002)/(1+18*(t+.002))
            track.append(dict(x=500+800*z,y=1300-6500*z,playback_s=1+t))
        return seed,track

    def test_projective_fit_recovers_known_motion(self):
        seed,track=self.motion(); f=fit(track,seed)
        self.assertAlmostEqual(f['proj_vy']*1920,6500,delta=1)
        self.assertAlmostEqual(f['proj_vx']*1080,800,delta=1)
        self.assertAlmostEqual(f['proj_depth_rate'],18,delta=.01)
        self.assertAlmostEqual(f['proj_shift'],.002,delta=.00001)

    def test_parameters_do_not_depend_on_track_length_for_exact_model(self):
        seed,short=self.motion(.065);_,long=self.motion()
        a,b=fit(short,seed),fit(long,seed)
        for key in ['proj_vx','proj_vy','proj_depth_rate']:
            self.assertAlmostEqual(a[key],b[key],delta=.01)

    def test_short_missing_tracks_abstain(self):
        seed,track=self.motion()
        self.assertEqual(fit([],seed),{})
        self.assertEqual(fit(track[:4],seed),{})
        self.assertEqual(fit(track,None),{})

    def test_fixed_time_features_do_not_extrapolate(self):
        seed,track=self.motion(.065);f=features(track,seed,800)
        self.assertIn('rise_040',f)
        self.assertNotIn('rise_080',f)
        self.assertNotIn('rise_120',f)

    def test_routing_uses_availability_and_training_only(self):
        X=np.c_[np.arange(12.),np.arange(12.)*.2];y=np.arange(12.)*3.
        X[-2:]=np.nan; original=X.copy()
        model=QualityRoutedRegressor('ridge10','mean').fit(X,y)
        query=np.array([[2.,.4],[np.nan,.4]])
        actual=model.predict(query)
        self.assertAlmostEqual(actual[1],y.mean())
        self.assertAlmostEqual(actual[0],model.primary_.predict(query[:1])[0])
        np.testing.assert_equal(X,original)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(model.primary_.named_steps['standardscaler'].n_samples_seen_,10)

    def test_routing_with_no_valid_training_tracks(self):
        X=np.full((9,2),np.nan);y=np.arange(9.)
        model=QualityRoutedRegressor('svr10','mean').fit(X,y)
        self.assertIsNone(model.primary_)
        np.testing.assert_allclose(model.predict([[1.,2.],[np.nan,np.nan]]),[4.,4.])

    def test_single_regime_soft_model(self):
        model=SoftCarryRegressor().fit(np.ones((8,2)),np.arange(8.)+100)
        np.testing.assert_allclose(model.predict([[1.,2.]]),[103.5])

    def test_saved_oof_predictions_and_metrics(self):
        for version in range(1,8):
            folder=ROOT/f'experiments/audited_launch_v{version}'
            p=pd.read_csv(folder/'predictions.csv');m=pd.read_csv(folder/'metrics.csv')
            keys=['cohort','cutoff','features','method']
            self.assertFalse(p.duplicated(keys+['shot_id']).any())
            self.assertTrue(np.isfinite(p.predicted).all())
            expected={('all_labeled',0):72,('matched_club',0):65,('all_labeled',100):44,('matched_club',100):39}
            for group,q in p.groupby(keys):
                self.assertEqual(len(q),expected[group[:2]])
            measured=p.assign(error=abs(p.actual-p.predicted)).groupby(keys).error.mean()
            reported=m.set_index(keys).mae
            np.testing.assert_allclose(measured.sort_index(),reported.sort_index(),atol=1e-10)

if __name__=='__main__': unittest.main()
