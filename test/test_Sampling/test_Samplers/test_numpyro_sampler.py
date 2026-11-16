import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.Sampling.Samplers.numpyro_sampler import NumpyroSampler


class TestNumpyroSampler(object):
    """A correlated 2d Gaussian, well inside the bounds, so the answers are
    analytic."""

    def setup_method(self):
        self.mean = np.array([0.3, -0.2])
        self.cov = np.array([[0.01, 0.006], [0.006, 0.0064]])
        prec = jnp.asarray(np.linalg.inv(self.cov))
        mean = jnp.asarray(self.mean)

        def logL(x):
            d = x - mean
            return -0.5 * d @ prec @ d

        self.sampler = NumpyroSampler(
            logL, args_lower=np.array([-2.0, -2.0]), args_upper=np.array([2.0, 2.0])
        )

    def test_transforms(self):
        args = np.array([0.5, -1.0])
        z = self.sampler.to_unconstrained(args)
        npt.assert_allclose(self.sampler.to_constrained(z), args, rtol=1e-10)

    def test_bounds_check(self):
        with pytest.raises(ValueError):
            NumpyroSampler(lambda x: 0.0, np.array([1.0]), np.array([0.0]))

    def test_map(self):
        result = self.sampler.map(
            args_init=np.array([1.0, 1.0]),
            args_sigma=np.array([0.3, 0.3]),
            num_starts=3,
            tolerance=1e-12,
        )
        npt.assert_allclose(result["args"], self.mean, atol=1e-5)
        npt.assert_allclose(result["logL"], 0.0, atol=1e-8)
        assert len(result["logL_starts"]) == 3
        npt.assert_allclose(result["covariance"], self.cov, rtol=1e-4)

    def test_map_parallel(self):
        # the per-device path (one start per device, padded rounds) must match the
        # one-at-a-time path
        kwargs = dict(
            args_init=np.array([1.0, 1.0]),
            args_sigma=np.array([0.3, 0.3]),
            num_starts=3,
            tolerance=1e-12,
            laplace=False,
        )
        result_parallel = self.sampler.map(parallel=True, **kwargs)
        result_serial = self.sampler.map(parallel=False, **kwargs)
        npt.assert_allclose(
            result_parallel["logL_starts"], result_serial["logL_starts"], atol=1e-8
        )
        npt.assert_allclose(result_parallel["args"], self.mean, atol=1e-5)

    def test_map_from_gradient_singularity(self):
        # smooth logL written through |x|, whose gradient is NaN at exactly 0, like
        # the EPL ellipticity at e1 = e2 = 0
        def logL(x):
            r = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
            return -((r - 0.3) ** 2) / 0.01

        sampler = NumpyroSampler(logL, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        result = sampler.map(args_init=np.zeros(2), tolerance=1e-12, laplace=False)
        npt.assert_allclose(np.hypot(*result["args"]), 0.3, atol=1e-4)

    def test_map_all_nan(self):
        sampler = NumpyroSampler(
            lambda x: jnp.sqrt(-1.0 - x[0] ** 2), np.array([-1.0]), np.array([1.0])
        )
        with pytest.raises(ValueError, match="not finite"):
            sampler.map(args_init=np.zeros(1))

    def test_laplace_at_bound(self):
        # the second parameter's likelihood peaks at 1.2, outside its upper bound of
        # 1, so the MAP presses against the bound; the curvature is still positive
        def logL(x):
            return -0.5 * (x[0] - 0.3) ** 2 / 0.01 - 0.5 * (x[1] - 1.2) ** 2 / 0.04

        sampler = NumpyroSampler(logL, np.array([-1.0, 0.0]), np.array([1.0, 1.0]))
        result = sampler.map(args_init=np.array([0.0, 0.5]), tolerance=1e-10)
        assert result["args"][1] > 0.99
        npt.assert_allclose(result["covariance"], np.diag([0.01, 0.04]), rtol=1e-3)
        assert np.all(np.linalg.eigvalsh(result["covariance_unconstrained"]) > 0)

    def test_map_requires_sigma(self):
        with pytest.raises(ValueError):
            self.sampler.map(args_init=np.zeros(2), num_starts=2)

    def test_svi(self):
        result = self.sampler.svi(
            args_init=self.mean,
            num_steps=3000,
            learning_rate=1e-2,
            num_particles=4,
            num_samples=4000,
            progress=False,
        )
        assert result["samples"].shape == (4000, 2)
        assert result["logL"].shape == (4000,)
        npt.assert_allclose(result["mean"], self.mean, atol=0.02)
        npt.assert_allclose(np.cov(result["samples"].T), self.cov, rtol=0.25)

    def test_hmc(self):
        result = self.sampler.hmc(
            args_init=self.mean,
            num_warmup=300,
            num_samples=1000,
            num_chains=2,
            chain_method="sequential",
            init_covariance_unconstrained=np.eye(2) * 1e-3,
            progress=False,
        )
        samples = result["samples"]
        assert samples.shape == (2, 1000, 2)
        assert result["logL"].shape == (2, 1000)
        assert result["num_divergences"] == 0
        assert np.all(result["r_hat"] < 1.05)
        flat = samples.reshape(-1, 2)
        npt.assert_allclose(flat.mean(axis=0), self.mean, atol=0.02)
        npt.assert_allclose(np.cov(flat.T), self.cov, rtol=0.2)

    def test_hmc_until_converged(self):
        result = self.sampler.hmc(
            args_init=self.mean,
            num_warmup=300,
            num_samples=2000,
            num_chains=2,
            chain_method="vectorized",
            init_covariance_unconstrained=np.eye(2) * 1e-3,
            rhat_target=1.05,
            check_every=100,
            min_samples=100,
            progress=False,
        )
        num_draws = result["samples"].shape[1]
        # an easy Gaussian converges long before the cap, in whole blocks
        assert num_draws < 2000 and num_draws % 100 == 0
        assert np.max(result["r_hat"]) < 1.05
        assert result["logL"].shape == (2, num_draws)
        assert result["diverging"].shape == (2, num_draws)
        flat = result["samples"].reshape(-1, 2)
        npt.assert_allclose(flat.mean(axis=0), self.mean, atol=0.03)

    def test_hmc_until_converged_reaches_cap(self):
        # an unreachable target: stops at num_samples, with a shorter last block
        result = self.sampler.hmc(
            args_init=self.mean,
            num_warmup=100,
            num_samples=250,
            num_chains=2,
            chain_method="sequential",
            init_covariance_unconstrained=np.eye(2) * 1e-3,
            rhat_target=0.5,
            check_every=100,
            min_samples=100,
            progress=False,
        )
        assert result["samples"].shape == (2, 250, 2)
        assert result["num_steps"].shape == (2, 250)


if __name__ == "__main__":
    pytest.main()
