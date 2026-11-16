"""MAP -> SVI -> HMC through FittingSequence on a small noiseless EPL + shear lens."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import simulation_util as sim_util

from jaxtronomy.Data.imaging_data import ImageData
from jaxtronomy.ImSim.image_model import ImageModel
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.Workflow.fitting_sequence import FittingSequence


class TestFittingSequenceNumpyro(object):

    def setup_method(self):
        num_pix, delta_pix = 30, 0.1
        kwargs_data = sim_util.data_configure_simple(
            num_pix, delta_pix, exposure_time=1000.0, background_rms=0.05
        )
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": 0.2, "pixel_size": delta_pix}
        kwargs_numerics = {"supersampling_factor": 1}

        self.kwargs_lens = [
            {"theta_E": 1.0, "gamma": 2.0, "e1": 0.1, "e2": -0.05,
             "center_x": 0.0, "center_y": 0.0},
            {"gamma1": 0.03, "gamma2": -0.01, "ra_0": 0, "dec_0": 0},
        ]
        kwargs_source = [{"amp": 30.0, "R_sersic": 0.2, "n_sersic": 1.5,
                          "e1": 0.1, "e2": 0.0, "center_x": 0.05, "center_y": 0.02}]
        kwargs_lens_light = [{"amp": 10.0, "R_sersic": 0.8, "n_sersic": 3.0,
                              "center_x": 0.0, "center_y": 0.0}]
        lens_model_list = ["EPL", "SHEAR"]
        source_list = ["SERSIC_ELLIPSE"]
        lens_light_list = ["SERSIC"]

        image_model = ImageModel(
            ImageData(**kwargs_data),
            PSF(**kwargs_psf),
            lens_model_class=LensModel(lens_model_list),
            source_model_class=LightModel(source_list),
            lens_light_model_class=LightModel(lens_light_list),
            kwargs_numerics=kwargs_numerics,
        )
        kwargs_data["image_data"] = np.array(
            image_model.image(self.kwargs_lens, kwargs_source,
                              kwargs_lens_light=kwargs_lens_light)
        )

        def block(init, sigma, fixed, lower, upper):
            return [init, sigma, fixed, lower, upper]

        lens_params = block(
            [{"theta_E": 0.95, "gamma": 2.05, "e1": 0.08, "e2": -0.03,
              "center_x": 0.02, "center_y": -0.02},
             {"gamma1": 0.02, "gamma2": 0.0}],
            [{"theta_E": 0.05, "gamma": 0.05, "e1": 0.03, "e2": 0.03,
              "center_x": 0.03, "center_y": 0.03},
             {"gamma1": 0.01, "gamma2": 0.01}],
            [{}, {"ra_0": 0, "dec_0": 0}],
            [{"theta_E": 0.5, "gamma": 1.5, "e1": -0.5, "e2": -0.5,
              "center_x": -0.5, "center_y": -0.5},
             {"gamma1": -0.2, "gamma2": -0.2}],
            [{"theta_E": 1.5, "gamma": 2.5, "e1": 0.5, "e2": 0.5,
              "center_x": 0.5, "center_y": 0.5},
             {"gamma1": 0.2, "gamma2": 0.2}],
        )
        source_params = block(
            [{"R_sersic": 0.25, "n_sersic": 1.5, "e1": 0.05, "e2": 0.0,
              "center_x": 0.0, "center_y": 0.0}],
            [{"R_sersic": 0.05, "n_sersic": 0.3, "e1": 0.05, "e2": 0.05,
              "center_x": 0.05, "center_y": 0.05}],
            [{}],
            [{"R_sersic": 0.05, "n_sersic": 0.5, "e1": -0.5, "e2": -0.5,
              "center_x": -1.0, "center_y": -1.0}],
            [{"R_sersic": 1.0, "n_sersic": 5.0, "e1": 0.5, "e2": 0.5,
              "center_x": 1.0, "center_y": 1.0}],
        )
        lens_light_params = block(
            [{"R_sersic": 0.8, "n_sersic": 3.0, "center_x": 0.0, "center_y": 0.0}],
            [{"R_sersic": 0.1, "n_sersic": 0.3, "center_x": 0.05, "center_y": 0.05}],
            [{"n_sersic": 3.0, "center_x": 0.0, "center_y": 0.0}],
            [{"R_sersic": 0.1, "n_sersic": 0.5, "center_x": -1.0, "center_y": -1.0}],
            [{"R_sersic": 3.0, "n_sersic": 8.0, "center_x": 1.0, "center_y": 1.0}],
        )

        self.fitting_seq = FittingSequence(
            {"multi_band_list": [[kwargs_data, kwargs_psf, kwargs_numerics]],
             "multi_band_type": "single-band"},
            {"lens_model_list": lens_model_list,
             "source_light_model_list": source_list,
             "lens_light_model_list": lens_light_list},
            {"linear_solver": True},
            {"check_bounds": True},
            {"lens_model": lens_params, "source_model": source_params,
             "lens_light_model": lens_light_params},
        )

    def test_map_svi_hmc(self):
        fitting_list = [
            ["MAP", {"num_starts": 2, "maxiter": 500, "tolerance": 1e-6,
                     "rng_seed": 1}],
            ["SVI", {"num_steps": 300, "learning_rate": 1e-3, "num_samples": 200,
                     "rng_seed": 2, "progress": False}],
            ["HMC", {"num_warmup": 50, "num_samples": 50, "num_chains": 2,
                     "chain_method": "sequential", "max_tree_depth": 5,
                     "rng_seed": 3, "progress": False}],
        ]
        chain_list = self.fitting_seq.fit_sequence(fitting_list)
        assert [c[0] for c in chain_list] == ["MAP", "SVI", "HMC"]

        name, samples, param_list, logL, info = chain_list[0]
        num_param = len(param_list)
        assert samples.shape == (1, num_param)
        assert "covariance" in info
        # noiseless data: the MAP sits at the truth
        kwargs_map = self.fitting_seq.param_class.args2kwargs(samples[0])
        npt.assert_allclose(kwargs_map["kwargs_lens"][0]["theta_E"], 1.0, atol=2e-3)
        npt.assert_allclose(kwargs_map["kwargs_lens"][0]["gamma"], 2.0, atol=2e-2)
        npt.assert_allclose(kwargs_map["kwargs_lens"][1]["gamma1"], 0.03, atol=5e-3)

        name, samples, _, logL, info = chain_list[1]
        assert samples.shape == (200, num_param)
        assert logL.shape == (200,)
        assert info["covariance_unconstrained"].shape == (num_param, num_param)
        assert len(info["losses"]) == 300

        name, samples, _, logL, info = chain_list[2]
        assert samples.shape == (2, 50, num_param)
        assert logL.shape == (2, 50)
        assert info["r_hat"].shape == (num_param,)
        theta_E = samples[..., param_list.index("theta_E_lens0")]
        npt.assert_allclose(np.median(theta_E), 1.0, atol=0.01)

        # the state ends at the best HMC sample
        kwargs_best = self.fitting_seq.best_fit()
        npt.assert_allclose(kwargs_best["kwargs_lens"][0]["theta_E"], 1.0, atol=0.01)

    def test_hmc_without_covariance(self):
        chain_list = self.fitting_seq.fit_sequence(
            [["HMC", {"num_warmup": 20, "num_samples": 10, "num_chains": 1,
                      "max_tree_depth": 4, "rng_seed": 0, "progress": False}]]
        )
        assert chain_list[0][1].shape[:2] == (1, 10)

    def test_unknown_fitting_type(self):
        with pytest.raises(ValueError):
            self.fitting_seq.fit_sequence([["HMCC", {}]])


if __name__ == "__main__":
    pytest.main()
