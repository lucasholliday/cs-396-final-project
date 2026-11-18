import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark import model


class ResearchContractTests(unittest.TestCase):
    def test_preprocessing_uses_training_rows(self):
        pipe=model('ridge10',396)
        pipe.fit(np.array([[1.],[3.],[np.nan]]),np.array([2.,6.,4.]))
        self.assertAlmostEqual(pipe.named_steps['simpleimputer'].statistics_[0],2)
        pipe.predict(np.array([[100000.]]))
        self.assertAlmostEqual(pipe.named_steps['standardscaler'].mean_[0],2)

    def test_fixed_splits_keep_golfer_groups(self):
        path=ROOT/'experiments/baseline_v1/splits.csv'
        if not path.exists():self.skipTest('Benchmark not yet run')
        d=pd.read_csv(path)
        self.assertFalse(d.shot_id.duplicated().any())
        for fold in d.golfer_fold.unique():
            self.assertFalse(set(d[d.golfer_fold==fold].golfer)&set(d[d.golfer_fold!=fold].golfer))
        self.assertEqual(d.groupby('golfer').golfer_fold.nunique().max(),1)

    def test_features_exclude_targets_and_identity(self):
        path=ROOT/'experiments/baseline_v1/config.json'
        if not path.exists():self.skipTest('Benchmark not yet run')
        c=json.loads(path.read_text())
        forbidden=set(c['config']['targets'])|{'golfer','shot_id','session','shot_in_session','video_peak_candidate_s'}
        for features in c['feature_sets'].values():self.assertFalse(set(features)&forbidden)

    def test_predictions_exactly_once_per_evaluation_shot(self):
        path=ROOT/'experiments/baseline_v1/predictions.csv'
        if not path.exists():self.skipTest('Benchmark not yet run')
        d=pd.read_csv(path)
        self.assertFalse(d.duplicated(['cohort','scheme','features','target','model','shot_id']).any())
        self.assertTrue(np.isfinite(d.predicted).all())


if __name__=='__main__':unittest.main()
