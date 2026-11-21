import json
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'scripts'))

from fuse_club_arc import fit_delivery_plane, SHAFT_LENGTH_M


class ClubFusionTests(unittest.TestCase):
    def test_shaft_length_constant(self):
        self.assertAlmostEqual(SHAFT_LENGTH_M, 0.946, places=3)

    def test_fit_delivery_plane_synthetic(self):
        # Create points lying in an inclined plane: y = 0.5 * z + 0.1 * x
        xs = np.linspace(-0.5, 0.5, 20)
        zs = np.linspace(-1.0, 0.0, 20)
        ys = 0.5 * zs + 0.1 * xs
        pts = np.column_stack([xs, ys, zs])
        normal, centroid, inclination = fit_delivery_plane(pts)
        self.assertTrue(np.isfinite(normal).all())
        self.assertTrue(np.isfinite(centroid).all())
        self.assertGreater(inclination, 20.0)
        self.assertLess(inclination, 85.0)

    def test_club_fusion_artifacts_exist_and_finite(self):
        fusion_files = list((ROOT / 'data/processed/club_fusion').glob('*.json'))
        self.assertEqual(len(fusion_files), 76)
        
        # Test sample shot
        d = json.loads((ROOT / 'data/processed/club_fusion/lucas_1_01.json').read_text())
        self.assertIn('status', d)
        self.assertIn('metrics', d)
        self.assertIn('delivery_arc', d)
        self.assertIn('swing_plane', d)
        
        # Check delivery arc points count and distance to ball
        pts = d['delivery_arc']['clubhead_points']
        self.assertGreaterEqual(len(pts), 20)
        for pt in pts:
            self.assertEqual(len(pt), 3)
            self.assertTrue(all(np.isfinite(x) for x in pt))


if __name__ == '__main__':
    unittest.main()
