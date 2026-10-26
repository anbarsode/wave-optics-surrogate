"""Small calculation checks for the visualizer backend."""

import unittest
import numpy as np

from viz.engine import calculate


def request(model="PM"):
    return {
        "model": model,
        "solvers": ["bgo"],
        "parameters": {
            "impact_x": 0.5,
            "impact_y": 0.0,
            "use_external": False,
            "convergence": 0.0,
            "shear": 0.0,
            "z_lens": 0.5,
            "z_source": 1.5,
            "mass": 100.0,
            "f_min": 10.0,
            "f_max": 1000.0,
        },
        "solver_settings": {
            "frequency_samples": 48,
            "sky_resolution": 48,
            "bgo_order": 0,
            "time_samples": 64,
        },
    }


class EngineTests(unittest.TestCase):
    def test_point_mass_calculation_has_two_images_and_finite_curves(self):
        result = calculate(request("PM"))
        self.assertEqual(len(result["images"]), 2)
        self.assertEqual(result["images"][0]["delay_seconds"], 0)
        self.assertEqual(len(result["frequency_hz"]), 48)
        self.assertTrue(np.all(np.isfinite(result["curves"]["bgo"]["amplitude"])))

    def test_sis_calculation_returns_matching_sky_grid(self):
        result = calculate(request("SIS"))
        field = result["field"]
        self.assertEqual(field["resolution"], 48)
        self.assertEqual(len(field["delay"]), 48**2)
        self.assertEqual(len(field["negative"]), 48**2)

    def test_multiple_image_radius_uses_a_finite_source_plane_area(self):
        result = calculate(request("PM"))
        field = result["field"]
        self.assertFalse(field["multiple_image_infinite"])
        self.assertGreater(field["multiple_image_radius"], 0)
        self.assertGreater(result["cosmology"]["theta_mi_mas"], 0)


if __name__ == "__main__":
    unittest.main()
