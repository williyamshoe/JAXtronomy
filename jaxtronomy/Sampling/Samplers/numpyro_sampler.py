"""Gradient-based inference with numpyro: MAP, SVI and HMC (NUTS).

All three methods share one numpyro model: the sampled parameters ``args`` have a
uniform prior between the lower and upper parameter bounds, and the jaxtronomy
log-likelihood (which already contains any Gaussian priors and custom logL terms
from ``kwargs_likelihood``) enters through ``numpyro.factor``. numpyro maps the
bounded parameters to an unbounded space, so neither the optimizer nor the
sampler ever hits the hard bound penalty of ``Likelihood.logL``.
"""

__author__ = "jaxtronomy developers"

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import constrain_fn, potential_energy, unconstrain_fn

_SITE = "args"


class NumpyroSampler(object):
    """MAP optimization, stochastic variational inference and Hamiltonian Monte Carlo
    on a jaxtronomy log-likelihood."""

    def __init__(self, logL_func, args_lower, args_upper):
        """
        :param logL_func: jax-traceable function of a 1d parameter array returning the
            log-likelihood, usually ``Likelihood.logL``
        :param args_lower: 1d array, lower bounds of the parameters
        :param args_upper: 1d array, upper bounds of the parameters
        """
        self.logL = logL_func
        self._lower = jnp.asarray(args_lower, dtype=float)
        self._upper = jnp.asarray(args_upper, dtype=float)
        if not bool(jnp.all(self._upper > self._lower)):
            raise ValueError("every upper bound must be larger than its lower bound")
        self.num_param = len(self._lower)
        self._logL_chunk = None
        self._lbfgs_pmap_cache = {}

    # ------------------------------------------------------------------
    # model and coordinate transforms
    # ------------------------------------------------------------------
    def model(self):
        """Numpyro model: flat prior within the bounds times the likelihood."""
        args = numpyro.sample(
            _SITE, dist.Uniform(self._lower, self._upper).to_event(1)
        )
        numpyro.factor("logL", self.logL(args))

    def to_unconstrained(self, args):
        """Maps parameters within the bounds to the unbounded sampling space."""
        return unconstrain_fn(self.model, (), {}, {_SITE: jnp.asarray(args)})[_SITE]

    def to_constrained(self, z):
        """Maps the unbounded sampling space back to parameters within the bounds."""
        return constrain_fn(self.model, (), {}, {_SITE: jnp.asarray(z)})[_SITE]

    def _clip_into_bounds(self, args):
        """Moves parameters that sit on or outside a bound slightly inside it, where
        the unbounded transform is finite."""
        width = self._upper - self._lower
        return jnp.clip(
            jnp.asarray(args, dtype=float),
            self._lower + 1e-6 * width,
            self._upper - 1e-6 * width,
        )

    def _neg_logL_unconstrained(self, z):
        return -self.logL(self.to_constrained(z))

    def _potential(self, z):
        """Potential energy of the model in the unbounded space: -logL minus the
        log-Jacobian of the bound transform (up to the constant uniform prior)."""
        return potential_energy(self.model, (), {}, {_SITE: z})

    def _constrain_chains(self, z_chains):
        """Maps unbounded samples (num_chains, num_draws, num_param) to the
        parameters."""
        z_chains = jnp.asarray(z_chains)
        flat = jax.vmap(self.to_constrained)(z_chains.reshape(-1, self.num_param))
        return np.asarray(flat).reshape(z_chains.shape)

    def _finite_gradient_start(self, args, rng_seed, max_tries=10, jitter=1e-4):
        """Returns ``args``, nudged if needed so that logL and its gradient are finite
        there.

        Some profiles have a gradient singularity at special parameter values that
        are natural starting guesses, e.g. EPL at exactly e1 = e2 = 0, where the
        conversion to orientation angle and axis ratio (``arctan2`` and ``sqrt``) has
        no derivative even though the lens model is smooth. Moving by a tiny fraction
        of the bound width avoids it.

        :param args: 1d array, starting parameters
        :param rng_seed: int, seed for the nudges
        :param max_tries: int, number of nudges to try
        :param jitter: float, nudge size as a fraction of each parameter's bound width
        :return: 1d array of starting parameters
        """
        value_and_grad = jax.jit(jax.value_and_grad(self.logL))
        width = self._upper - self._lower
        key = jax.random.PRNGKey(rng_seed)
        candidate = args
        for i in range(max_tries + 1):
            value, grad = value_and_grad(candidate)
            if bool(jnp.isfinite(value)) and bool(jnp.all(jnp.isfinite(grad))):
                if i > 0:
                    print(
                        "Nudged the starting point by %.0e of the bound widths: the "
                        "logL gradient is not finite at the given start." % jitter
                    )
                return candidate
            key, subkey = jax.random.split(key)
            candidate = self._clip_into_bounds(
                args + jitter * width * jax.random.normal(subkey, (self.num_param,))
            )
        raise ValueError(
            "logL or its gradient is not finite at the starting point, even after %d "
            "small nudges. Check the starting parameters." % max_tries
        )

    def logL_batch(self, samples, batch_size=64):
        """Log-likelihood of many parameter vectors, evaluated in fixed-size chunks
        from Python, so at most ``batch_size`` likelihood calls are in memory at once.
        (A single compiled loop over all samples can be restructured by the GPU
        compiler into one call over every sample, which runs out of memory for
        thousands of samples of an imaging likelihood.)

        :param samples: 2d array (num_samples, num_param)
        :param batch_size: int, number of samples evaluated at once
        :return: 1d array of log-likelihoods
        """
        samples = np.asarray(samples, dtype=float)
        num = len(samples)
        batch_size = max(1, min(int(batch_size), num))
        if self._logL_chunk is None:
            self._logL_chunk = jax.jit(jax.vmap(self.logL))
        logL = np.empty(num)
        for start in range(0, num, batch_size):
            chunk = samples[start : start + batch_size]
            n = len(chunk)
            if n < batch_size:  # pad so every chunk reuses one compiled shape
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch_size - n, 0)])
            logL[start : start + n] = np.asarray(self._logL_chunk(chunk))[:n]
        return logL

    # ------------------------------------------------------------------
    # MAP
    # ------------------------------------------------------------------
    def map(
        self,
        args_init,
        args_sigma=None,
        num_starts=1,
        maxiter=1000,
        tolerance=1e-3,
        laplace=True,
        rng_seed=0,
        parallel=None,
    ):
        """Maximum a posteriori estimate with multi-start L-BFGS.

        The posterior is the likelihood within the bounds, so this is also the maximum
        likelihood within the bounds. The first start is ``args_init``; the others are
        drawn from a normal distribution of width ``args_sigma`` around it.

        :param args_init: 1d array, starting parameters
        :param args_sigma: 1d array, spread of the additional starts. Required if
            ``num_starts > 1``
        :param num_starts: int, number of L-BFGS runs
        :param maxiter: int, maximum number of L-BFGS iterations per start
        :param tolerance: float, a run stops when the loss improves by less than this
            three iterations in a row
        :param laplace: bool, if True also returns the inverse Hessian of -logL at the
            MAP (the Laplace approximation of the posterior covariance)
        :param rng_seed: int, seed for the additional starts
        :param parallel: bool or None, run the starts at the same time, one per
            device (e.g. one per GPU), in rounds of the number of devices. None
            does so whenever there is more than one device and more than one start
        :return: dict with ``args`` (best parameters), ``logL``, ``logL_starts`` (final
            logL of every start), ``num_iterations`` and, if ``laplace``,
            ``covariance`` (in the constrained parameters; None if the Hessian is
            not positive definite) and ``covariance_unconstrained``
        """
        args_init = self._clip_into_bounds(args_init)
        starts = [args_init]
        if num_starts > 1:
            if args_sigma is None:
                raise ValueError("args_sigma is required for num_starts > 1")
            key = jax.random.PRNGKey(rng_seed)
            draws = args_init + jnp.asarray(args_sigma) * jax.random.normal(
                key, (num_starts - 1, self.num_param)
            )
            starts += [self._clip_into_bounds(d) for d in draws]

        num_devices = jax.local_device_count()
        if parallel is None:
            parallel = num_devices > 1 and len(starts) > 1
        z_starts = [
            self.to_unconstrained(self._finite_gradient_start(start, rng_seed + i))
            for i, start in enumerate(starts)
        ]
        if parallel:
            print(
                "MAP: %d starts in parallel on %d devices" % (len(starts), num_devices),
                flush=True,
            )
            runs = []
            for first in range(0, len(z_starts), num_devices):
                chunk = z_starts[first : first + num_devices]
                n = len(chunk)
                # pad the last round so every device gets a start; extras are dropped
                padded = jnp.stack(chunk + [chunk[-1]] * (num_devices - n))
                z_out, num_out = self._lbfgs_pmap(maxiter, tolerance)(padded)
                runs += [(z_out[j], num_out[j]) for j in range(n)]
        else:
            runs = (self._lbfgs(z, maxiter, tolerance) for z in z_starts)

        best = None
        logL_starts, iterations = [], []
        for i, (z, num_iter) in enumerate(runs):
            logL = float(-self._neg_logL_unconstrained(z))
            if not (np.all(np.isfinite(np.asarray(z))) and np.isfinite(logL)):
                # a NaN step; jaxtronomy's logL of NaN parameters is not NaN, so
                # check the parameters themselves
                logL = -np.inf
            logL_starts.append(logL)
            iterations.append(int(num_iter))
            print(
                "MAP start %d/%d: logL = %.4f after %d iterations"
                % (i + 1, len(starts), logL, num_iter),
                flush=True,
            )
            if np.isfinite(logL) and (best is None or logL > best[1]):
                best = (jnp.asarray(z), logL)

        if best is None:
            raise RuntimeError(
                "Every MAP start ended at non-finite parameters (NaN gradients)."
            )
        z_best, logL_best = best
        result = {
            "args": np.asarray(self.to_constrained(z_best)),
            "logL": logL_best,
            "logL_starts": np.array(logL_starts),
            "num_iterations": np.array(iterations),
        }
        if laplace:
            result.update(self.laplace_covariance(z_best))
        return result

    @partial(jax.jit, static_argnums=(0, 2))
    def _lbfgs(self, z_init, maxiter, tolerance):
        return self._lbfgs_run(z_init, maxiter, tolerance)

    def _lbfgs_pmap(self, maxiter, tolerance):
        """L-BFGS mapped over devices: one start per device, each with its own
        loop, so a start that converges early does not wait for the others."""
        key = (maxiter, float(tolerance))
        if key not in self._lbfgs_pmap_cache:
            self._lbfgs_pmap_cache[key] = jax.pmap(
                lambda z: self._lbfgs_run(z, maxiter, tolerance)
            )
        return self._lbfgs_pmap_cache[key]

    def _lbfgs_run(self, z_init, maxiter, tolerance):
        opt = optax.lbfgs()
        loss = self._neg_logL_unconstrained
        value_and_grad = optax.value_and_grad_from_state(loss)

        def step(carry):
            z, state, num_small = carry
            value, grad = value_and_grad(z, state=state)
            updates, state = opt.update(
                grad, state, z, value=value, grad=grad, value_fn=loss
            )
            z = optax.apply_updates(z, updates)
            new_value = optax.tree.get(state, "value")
            num_small = jnp.where(value - new_value < tolerance, num_small + 1, 0)
            return z, state, num_small

        def keep_going(carry):
            _, state, num_small = carry
            return (optax.tree.get(state, "count") < maxiter) & (num_small < 3)

        z, state, _ = jax.lax.while_loop(
            keep_going, step, (z_init, opt.init(z_init), 0)
        )
        return z, optax.tree.get(state, "count")

    def laplace_covariance(self, z):
        """Laplace approximation at an optimum: the inverse Hessian of -logL in the
        parameters themselves, and its linear map to the unbounded space.

        The Hessian is taken in the parameters, not in the unbounded space. Where a
        parameter sits at a bound with the likelihood still rising towards it, the
        Hessian in the unbounded space picks up a term (gradient times the curvature
        of the bound transform) that can make it indefinite, although the posterior
        itself is well curved; the parameter-space Hessian does not.

        :param z: 1d array, optimum in the unbounded space
        :return: dict with ``covariance`` and ``covariance_unconstrained`` (both None
            if the Hessian is not positive definite)
        """
        z = jnp.asarray(z)
        args = self.to_constrained(z)
        hess = jax.hessian(lambda a: -self.logL(a))(args)
        cov = _inverse_if_positive_definite(hess)
        if cov is None:
            print(
                "Laplace approximation: the Hessian at the MAP is not positive "
                "definite; no covariance returned."
            )
            return {"covariance_unconstrained": None, "covariance": None}
        # the bound transform acts element-wise, so its Jacobian dz/dargs is diagonal
        jac = np.asarray(jax.jacfwd(self.to_constrained)(z))
        inv_jac = np.linalg.inv(jac)
        cov_z = inv_jac @ cov @ inv_jac.T
        return {"covariance_unconstrained": cov_z, "covariance": cov}

    # ------------------------------------------------------------------
    # SVI
    # ------------------------------------------------------------------
    def svi(
        self,
        args_init,
        num_steps=5000,
        learning_rate=1e-3,
        num_particles=1,
        init_scale=1e-3,
        init_covariance_unconstrained=None,
        num_samples=2000,
        clip_norm=10.0,
        rng_seed=0,
        progress=True,
    ):
        """Stochastic variational inference with a full-rank Gaussian guide in the
        unbounded space (``AutoMultivariateNormal``), so correlations between
        parameters are captured.

        :param init_covariance_unconstrained: 2d array or None, initial covariance of
            the guide in the unbounded space (e.g. the Laplace covariance from
            :meth:`map`). Starting from it saves most of the steps the guide would
            otherwise spend growing from ``init_scale``. If None, the guide starts as
            an isotropic Gaussian of width ``init_scale``

        :param args_init: 1d array, initial mean of the guide (e.g. the MAP)
        :param num_steps: int, number of optimization steps
        :param learning_rate: float, Adam learning rate
        :param num_particles: int, number of guide draws per ELBO estimate; more
            particles give less noisy gradients at proportionally higher cost
        :param init_scale: float, initial standard deviation of the guide in the
            unbounded space
        :param num_samples: int, number of draws from the fitted guide to return
        :param clip_norm: float, gradients are clipped to this global norm, which keeps
            the first steps stable when the guide starts far from the posterior
        :param rng_seed: int, random seed
        :param progress: bool, show a progress bar
        :return: dict with ``samples`` (num_samples, num_param), ``logL`` of the
            samples, ``mean`` (guide mean in the parameters), ``losses`` (negative
            ELBO per step), ``loc_unconstrained`` and ``covariance_unconstrained``
            (the fitted Gaussian in the unbounded space)
        """
        args_init = self._finite_gradient_start(
            self._clip_into_bounds(args_init), rng_seed=rng_seed
        )
        guide = AutoMultivariateNormal(
            self.model,
            init_loc_fn=init_to_value(values={_SITE: args_init}),
            init_scale=init_scale,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate)
        )
        svi = SVI(
            self.model,
            guide,
            optimizer,
            loss=Trace_ELBO(num_particles=num_particles, vectorize_particles=True),
        )
        init_params = None
        if init_covariance_unconstrained is not None:
            chol = np.linalg.cholesky(np.asarray(init_covariance_unconstrained))
            init_params = {
                "auto_loc": self.to_unconstrained(args_init),
                "auto_scale_tril": jnp.asarray(chol),
            }
        key_run, key_sample = jax.random.split(jax.random.PRNGKey(rng_seed))
        result = svi.run(
            key_run, num_steps, progress_bar=progress, init_params=init_params
        )
        params = result.params

        posterior = guide.get_posterior(params)
        loc_z = posterior.loc
        cov_z = posterior.covariance_matrix
        z_samples = posterior.sample(key_sample, (num_samples,))
        samples = jax.vmap(self.to_constrained)(z_samples)
        print(
            "SVI final loss (negative ELBO): %.4f" % float(result.losses[-1]), flush=True
        )

        return {
            "samples": np.asarray(samples),
            "logL": self.logL_batch(samples),
            "mean": np.asarray(self.to_constrained(loc_z)),
            "losses": np.asarray(result.losses),
            "loc_unconstrained": np.asarray(loc_z),
            "covariance_unconstrained": np.asarray(cov_z),
        }

    # ------------------------------------------------------------------
    # HMC
    # ------------------------------------------------------------------
    def hmc(
        self,
        args_init,
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        chain_method="parallel",
        init_covariance_unconstrained=None,
        init_spread=1.0,
        dense_mass=True,
        target_accept_prob=0.8,
        max_tree_depth=10,
        thinning=1,
        rhat_target=None,
        check_every=200,
        min_samples=200,
        rng_seed=0,
        progress=True,
    ):
        """Hamiltonian Monte Carlo with the No-U-Turn sampler.

        :param args_init: 1d array, starting point (e.g. the MAP or the SVI mean)
        :param num_warmup: int, warm-up (adaptation) steps per chain, not returned
        :param num_samples: int, samples per chain
        :param num_chains: int, number of chains
        :param chain_method: 'parallel' (one chain per device, e.g. per GPU; falls
            back to 'sequential' with a warning if there are fewer devices than
            chains), 'vectorized' (all chains on one device) or 'sequential'
        :param init_covariance_unconstrained: 2d array or None, covariance of the
            posterior in the unbounded space, e.g. from SVI or the Laplace
            approximation. Used as the initial inverse mass matrix and to spread the
            chain starting points. None starts from the identity
        :param init_spread: float, chains start at ``args_init`` plus draws from the
            initial covariance scaled by this factor (only with more than one chain
            and a covariance)
        :param dense_mass: bool, adapt a dense mass matrix, which handles the strong
            correlations between lens parameters (e.g. slope and shear)
        :param target_accept_prob: float, target acceptance probability of the step
            size adaptation
        :param max_tree_depth: int, maximum tree depth; each step costs up to
            2**max_tree_depth gradient evaluations
        :param thinning: int, keep every n-th sample
        :param rhat_target: float or None, stop sampling early once the largest
            split r_hat over all parameters is below this value (e.g. 1.01). After
            warm-up the chains are then sampled in blocks of ``check_every`` draws,
            r_hat is checked after each block, and ``num_samples`` becomes the maximum
            number of draws per chain. None always draws ``num_samples``
        :param check_every: int, draws per chain between r_hat checks
        :param min_samples: int, minimum draws per chain before stopping early, so
            that r_hat is not trusted on a handful of draws
        :param rng_seed: int, random seed
        :param progress: bool, show a progress bar
        :return: dict with ``samples`` (num_chains, num_samples, num_param), ``logL``
            (num_chains, num_samples), ``diverging`` (num_chains, num_samples bool),
            ``num_steps`` (leapfrog steps per sample), ``r_hat`` and ``n_eff``
            (per parameter), ``num_divergences``, ``inverse_mass_matrix`` and
            ``step_size`` after adaptation
        """
        args_init = self._finite_gradient_start(
            self._clip_into_bounds(args_init), rng_seed=rng_seed
        )
        z_init = self.to_unconstrained(args_init)

        kwargs_nuts = {
            "dense_mass": dense_mass,
            "target_accept_prob": target_accept_prob,
            "max_tree_depth": max_tree_depth,
        }
        if init_covariance_unconstrained is not None:
            cov = jnp.asarray(init_covariance_unconstrained)
            if not dense_mass:
                cov = jnp.diag(cov)
            kwargs_nuts["inverse_mass_matrix"] = cov
        # NUTS on a plain potential rather than on the numpyro model: with
        # chain_method='parallel' a model-based kernel keeps values traced inside
        # the first pmap, which breaks continuing the chains in blocks
        kernel = NUTS(potential_fn=self._potential, **kwargs_nuts)

        key_init, key_run = jax.random.split(jax.random.PRNGKey(rng_seed))
        if num_chains > 1 and init_covariance_unconstrained is not None:
            chol = jnp.linalg.cholesky(jnp.asarray(init_covariance_unconstrained))
            noise = jax.random.normal(key_init, (num_chains, self.num_param))
            z_starts = z_init + init_spread * noise @ chol.T
            z_starts = z_starts.at[0].set(z_init)
        else:
            z_starts = jnp.tile(z_init, (num_chains, 1))
        init_params = z_starts if num_chains > 1 else z_init

        def make_mcmc(n_samples):
            return MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=n_samples,
                num_chains=num_chains,
                chain_method=chain_method,
                thinning=thinning,
                progress_bar=progress,
            )

        if rhat_target is not None:
            check_every = max(int(thinning), min(int(check_every), int(num_samples)))
            mcmc = make_mcmc(check_every)
        else:
            mcmc = make_mcmc(num_samples)
        print(
            "HMC: %d chain(s), chain_method '%s', on %s"
            % (num_chains, chain_method, jax.devices()),
            flush=True,
        )
        fields = ("diverging", "num_steps")
        if rhat_target is None:
            mcmc.run(key_run, init_params=init_params, extra_fields=fields)
            samples = self._constrain_chains(mcmc.get_samples(group_by_chain=True))
            extra = {
                k: np.asarray(v)
                for k, v in mcmc.get_extra_fields(group_by_chain=True).items()
            }
        else:
            samples, extra, mcmc = self._hmc_until_converged(
                mcmc,
                make_mcmc,
                self._constrain_chains,
                key_run,
                init_params,
                fields,
                rhat_target=rhat_target,
                check_every=check_every,
                min_samples=min_samples,
                max_samples=num_samples,
            )
        n_chain, n_draw, n_dim = samples.shape
        logL = self.logL_batch(samples.reshape(-1, n_dim)).reshape(n_chain, n_draw)

        summary = numpyro.diagnostics.summary(
            {_SITE: samples}, group_by_chain=True
        )[_SITE]
        diverging = np.asarray(extra["diverging"])
        state = mcmc.last_state
        adapt = state.adapt_state
        print(
            "HMC: %d divergences, max r_hat = %.3f, min n_eff = %.0f"
            % (diverging.sum(), np.max(summary["r_hat"]), np.min(summary["n_eff"]))
        )

        return {
            "samples": np.asarray(samples),
            "logL": logL,
            "diverging": diverging,
            "num_steps": np.asarray(extra["num_steps"]),
            "r_hat": np.asarray(summary["r_hat"]),
            "n_eff": np.asarray(summary["n_eff"]),
            "num_divergences": int(diverging.sum()),
            "inverse_mass_matrix": np.asarray(
                _first_value(adapt.inverse_mass_matrix)
            ),
            "step_size": np.asarray(adapt.step_size),
        }


    @staticmethod
    def _hmc_until_converged(
        mcmc, make_mcmc, constrain, rng_key, init_params, fields, rhat_target, check_every,
        min_samples, max_samples,
    ):
        """Warm up once, then draw blocks of ``check_every`` samples per chain,
        continuing each block from the last state, until the largest split r_hat is
        below ``rhat_target`` (after at least ``min_samples`` draws) or
        ``max_samples`` draws are reached.

        :param mcmc: numpyro MCMC set up to draw ``check_every`` samples per run
        :param make_mcmc: function of a number of samples returning a new MCMC with
            the same kernel, used for a shorter last block
        :param constrain: function mapping unbounded chains to the parameters
        :return: samples (num_chains, num_draws, num_param), dict of extra fields
            (num_chains, num_draws), the MCMC object of the last block
        """
        blocks, extras = [], {k: [] for k in fields}
        while True:
            if not blocks:
                mcmc.run(rng_key, init_params=init_params, extra_fields=fields)
            else:
                # continue from the last state: no new warm-up, the adapted step
                # size and mass matrix are kept
                mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=fields)
            blocks.append(constrain(mcmc.get_samples(group_by_chain=True)))
            for k, v in mcmc.get_extra_fields(group_by_chain=True).items():
                extras[k].append(np.asarray(v))
            samples = np.concatenate(blocks, axis=1)
            num_draws = samples.shape[1]
            r_hat = numpyro.diagnostics.summary(
                {_SITE: samples}, group_by_chain=True
            )[_SITE]["r_hat"]
            max_r_hat = float(np.nanmax(np.asarray(r_hat)))
            print(
                "HMC: %d draws per chain, max r_hat = %.4f (target %.3f)"
                % (num_draws, max_r_hat, rhat_target),
                flush=True,
            )
            if num_draws >= min_samples and max_r_hat < rhat_target:
                print("HMC: converged after %d draws per chain" % num_draws, flush=True)
                break
            if num_draws >= max_samples:
                print(
                    "HMC: reached the maximum of %d draws per chain with max r_hat "
                    "%.4f >= %.3f; consider more warm-up or a larger num_samples"
                    % (num_draws, max_r_hat, rhat_target),
                    flush=True,
                )
                break
            last_state = mcmc.last_state
            remaining = max_samples - num_draws
            if remaining < check_every:
                mcmc = make_mcmc(remaining)
            mcmc.post_warmup_state = last_state
        extra = {k: np.concatenate(v, axis=1) for k, v in extras.items()}
        return samples, extra, mcmc

def _inverse_if_positive_definite(matrix):
    matrix = np.asarray(matrix)
    matrix = 0.5 * (matrix + matrix.T)
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return None
    inv_chol = np.linalg.inv(chol)
    return inv_chol.T @ inv_chol


def _first_value(value):
    """The adapted inverse mass matrix is a dict keyed by site tuples."""
    if isinstance(value, dict):
        return next(iter(value.values()))
    return value
