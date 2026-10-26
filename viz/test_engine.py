"""Small calculation checks for the visualizer backend."""

import unittest
import numpy as np

from viz.engine import Settings, calculate


def request(model="PM"):
    return {
        "model": model,
        "solvers": ["bgo"],
        "parameters": {
            "impact_x": 0.5,
            "impact_y": 0.0,
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
        result = calculate(request("SIS"))
        field = result["field"]
        self.assertFalse(field["multiple_image_infinite"])
        self.assertGreater(field["multiple_image_radius"], 0)
        self.assertGreater(result["cosmology"]["theta_mi_mas"], 0)

    def test_point_mass_multiple_image_radius_uses_the_sampled_source_plane_area(self):
        result = calculate(request("PM"))
        self.assertFalse(result["field"]["multiple_image_infinite"])
        self.assertGreater(result["field"]["multiple_image_radius"], 0)
        self.assertFalse(result["cosmology"]["theta_mi_infinite"])

    def test_glow_returns_its_direct_nonnegative_delay_density(self):
        input_request = request("PM")
        input_request["solvers"] = ["glow"]
        input_request["solver_settings"]["glow_accuracy"] = -1
        result = calculate(input_request)
        curve = result["curves"]["glow"]
        self.assertIn("delay", curve)
        self.assertIn("delay_density", curve)
        self.assertTrue(np.all(np.asarray(curve["delay_density"]) >= 0))

    def test_external_field_keeps_far_field_eigenvalues_positive(self):
        settings = Settings.from_request({"parameters": {"convergence": .30, "shear": -.90}})
        self.assertGreater(settings.convergence, 0)
        self.assertLess(settings.shear, 0)
        self.assertGreater(1 - settings.convergence - abs(settings.shear), 0)

    def test_signed_shear_does_not_disable_the_external_potential(self):
        settings = Settings.from_request({"parameters": {"convergence": 0, "shear": -.20}})
        self.assertTrue(settings.use_external)

    def test_negative_external_convergence_represents_an_underdensity(self):
        settings = Settings.from_request({"parameters": {"convergence": -.30, "shear": .20}})
        self.assertEqual(settings.convergence, -.30)
        self.assertGreater(1 - settings.convergence - abs(settings.shear), 0)

    def test_source_lens_and_external_parameters_stay_independent(self):
        settings = Settings.from_request({"parameters": {
            "impact_x": -1.2, "impact_y": .7, "z_lens": .8, "z_source": 4.2,
            "mass": 3e5, "convergence": -.4, "shear": .5,
        }})
        np.testing.assert_allclose(settings.source, [-1.2, .7])
        self.assertEqual(settings.lens_redshift, .8)
        self.assertEqual(settings.source_redshift, 4.2)
        self.assertEqual(settings.redshifted_mass, 3e5)
        self.assertEqual(settings.convergence, -.4)
        self.assertEqual(settings.shear, .5)


if __name__ == "__main__":
    unittest.main()
