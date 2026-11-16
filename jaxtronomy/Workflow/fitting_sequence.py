from jaxtronomy.Sampling.likelihood import Likelihood
from jaxtronomy.Sampling.Samplers.optax import OptaxMinimizer
from jaxtronomy.Sampling.Samplers.numpyro_sampler import NumpyroSampler
from jaxtronomy.Sampling.sampler import Sampler

from lenstronomy.Sampling.likelihood import Likelihood as Likelihood_lenstronomy
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Workflow.psf_fitting import PsfFitting
from lenstronomy.Workflow.alignment_matching import AlignmentFitting
from lenstronomy.Workflow.multi_band_manager import MultiBandUpdateManager
from lenstronomy.Sampling.Samplers.multinest_sampler import MultiNestSampler
from lenstronomy.Sampling.Samplers.polychord_sampler import DyPolyChordSampler
from lenstronomy.Sampling.Samplers.dynesty_sampler import DynestySampler
from lenstronomy.Sampling.Samplers.nautilus_sampler import NautilusSampler
from lenstronomy.Sampling.Samplers.cobaya_sampler import CobayaSampler

import copy
import jax
import numpy as np
import lenstronomy.Util.analysis_util as analysis_util

__all__ = ["FittingSequence"]


class FittingSequence(object):
    """Class to define a sequence of fitting applied, inherit the Fitting class this is
    a Workflow manager that allows to update model configurations before executing
    another step in the modelling The user can take this module as an example of how to
    create their own workflows or build their own around the FittingSequence."""

    def __init__(
        self,
        kwargs_data_joint,
        kwargs_model,
        kwargs_constraints,
        kwargs_likelihood,
        kwargs_params,
        mpi=False,
        verbose=True,
    ):
        """

        :param kwargs_data_joint: keyword argument specifying the data according to Likelihood
        :param kwargs_model: keyword arguments to describe all model components used in
         class_creator.create_class_instances()
        :param kwargs_constraints: keyword arguments of the Param() class to handle parameter constraints during the
         sampling (except upper and lower limits and sampling input mean and width)
        :param kwargs_likelihood: keyword arguments of the Likelihood() class to handle parameters and settings of the
         likelihood
        :param kwargs_params: setting of the sampling bounds and initial guess mean and spread.
         The argument is organized as:
         'lens_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'lens_light_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'point_source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'extinction_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'special': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'tracer_source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
        :param mpi: MPI option (bool), if True, will launch an MPI Pool job for the steps in the fitting sequence where
         possible
        :param verbose: bool, if True prints temporary results and indicators of the fitting process
        """
        solver_type = kwargs_constraints.get("solver_type", None)
        if solver_type is not None and solver_type != "NONE":
            raise NotImplementedError(
                "JAXtronomy has not implemented any solvers, so solver_type must be set to None or 'NONE'."
            )

        self.kwargs_data_joint = kwargs_data_joint
        self.multi_band_list = kwargs_data_joint.get("multi_band_list", [])
        self.multi_band_type = kwargs_data_joint.get("multi_band_type", "single-band")
        self._verbose = verbose
        self._mpi = mpi

        kwargs_constraints["_jax"] = True
        self._updateManager = MultiBandUpdateManager(
            kwargs_model,
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_params,
            num_bands=len(self.multi_band_list),
        )
        self._mcmc_init_samples = None
        # posterior covariance in numpyro's unbounded space from the last MAP (Laplace)
        # or SVI step, with the parameter names it belongs to; seeds the HMC mass matrix
        self._numpyro_covariance = None
        self._psf_iteration_memory = []
        self._psf_iteration_index = 0  # index of the sequence of the PSF iteration (how many times it is being run)

    @property
    def kwargs_fixed(self):
        """Returns the updated kwargs_fixed from the update manager.

        :return: list of fixed kwargs, see UpdateManager()
        """
        return self._updateManager.fixed_kwargs

    def fit_sequence(self, fitting_list):
        """

        :param fitting_list: list of [['string', {kwargs}], ..] with 'string being the specific fitting option and
         kwargs being the arguments passed to this option
        :return: fitting results
        """
        chain_list = []
        for i, fitting in enumerate(fitting_list):
            # clear the cache of compiled functions before each iteration to free up memory
            # since a new Likelihood class is created each time, functions will have to be recompiled anyways
            jax.clear_caches()

            fitting_type = fitting[0]
            kwargs = fitting[1]

            if fitting_type in [
                "PSO",
                "SIMPLEX",
                "MCMC",
                "emcee",
                "zeus",
                "Cobaya",
                "dynesty",
                "dyPolyChord",
                "MultiNest",
                "nested_sampling",
                "Nautilus",
                "MAP",
                "SVI",
                "HMC",
            ]:
                self._updateManager.check_initial_state()
            if fitting_type == "restart":
                self._updateManager.set_init_state()
                self._updateManager.check_initial_state()

            elif fitting_type == "update_settings":
                self.update_settings(**kwargs)

            elif fitting_type == "set_param_value":
                self.set_param_value(**kwargs)

            elif fitting_type == "set_amplitudes":
                self.set_amplitudes()

            elif fitting_type == "fix_not_computed":
                self.fix_not_computed(**kwargs)

            elif fitting_type == "psf_iteration":
                self.psf_iteration(**kwargs)

            elif fitting_type == "align_images":
                self.align_images(**kwargs)

            elif fitting_type == "calibrate_images":
                raise ValueError("calibrate_images not supported in jaxtronomy")

            elif fitting_type == "PSO":
                kwargs_result, chain, param = self.pso(**kwargs)
                self._updateManager.update_param_state(**kwargs_result)

                chain_list.append([fitting_type, chain, param])

            elif fitting_type == "SIMPLEX":
                kwargs_result = self.simplex(**kwargs)
                self._updateManager.update_param_state(**kwargs_result)
                chain_list.append([fitting_type, kwargs_result])

            elif fitting_type in ["MCMC", "emcee", "zeus"]:
                if fitting_type == "MCMC":
                    print("MCMC selected. Sampling with default option emcee.")
                    fitting_type = "emcee"
                if "init_samples" not in kwargs:
                    kwargs["init_samples"] = self._mcmc_init_samples
                elif kwargs["init_samples"] is None:
                    kwargs["init_samples"] = self._mcmc_init_samples
                mcmc_output = self.mcmc(**kwargs, sampler_type=fitting_type)
                kwargs_result = self._result_from_mcmc(mcmc_output)
                self._updateManager.update_param_state(**kwargs_result)
                chain_list.append(mcmc_output)

            elif fitting_type == "Cobaya":
                print("Using the Metropolis--Hastings MCMC sampler in Cobaya.")
                param_class = self.param_class
                kwargs_temp = self._updateManager.parameter_state
                mean_start = param_class.kwargs2args(**kwargs_temp)
                kwargs_sigma = self._updateManager.sigma_kwargs
                sigma_start = np.array(param_class.kwargs2args(**kwargs_sigma))
                # pass the likelihood and starting info to the sampler
                sampler = CobayaSampler(self.likelihood_class, mean_start, sigma_start)
                # run the sampler
                updated_info, sampler_type, best_fit_values = sampler.run(**kwargs)
                # change the best-fit values returned by cobaya into lenstronomy kwargs format
                best_fit_kwargs = self.param_class.args2kwargs(
                    best_fit_values, bijective=True
                )
                # collect the products
                mh_output = [updated_info, sampler_type, best_fit_kwargs]
                # append the products to the chain list
                chain_list.append(mh_output)

            elif fitting_type in [
                "dynesty",
                "dyPolyChord",
                "MultiNest",
                "nested_sampling",
            ]:
                if fitting_type == "nested_sampling":
                    print(
                        "Nested sampling selected. Sampling with default option dynesty."
                    )
                    fitting_type = "dynesty"
                ns_output = self.nested_sampling(**kwargs, sampler_type=fitting_type)
                chain_list.append(ns_output)

            elif fitting_type == "Nautilus":
                # do importance nested sampling with Nautilus
                nautilus = NautilusSampler(
                    likelihood_module=self.likelihood_class, mpi=self._mpi, **kwargs
                )
                samples, means, log_z, log_z_err, log_l, results_object = nautilus.run(
                    **kwargs
                )
                chain_list.append(
                    [
                        fitting_type,
                        samples,
                        nautilus.param_names,
                        log_l,
                        log_z,
                        log_z_err,
                        results_object,
                    ]
                )
                if kwargs.get("verbose", False):
                    print(len(samples), "number of points sampled")
                kwargs_result = self.best_fit_from_samples(
                    results_object["points"], results_object["log_l"]
                )
                self._updateManager.update_param_state(**kwargs_result)
            elif fitting_type == "optax":
                kwargs_result = self.optax(**kwargs)
                self._updateManager.update_param_state(**kwargs_result)
                chain_list.append([fitting_type, kwargs_result])
            elif fitting_type in ["MAP", "SVI", "HMC"]:
                method = {"MAP": self.map, "SVI": self.svi, "HMC": self.hmc}
                output = method[fitting_type](**kwargs)
                chain_list.append(output)
            else:
                raise ValueError(
                    "fitting_sequence {} is not supported. Please use: 'PSO', 'SIMPLEX', 'optax', "
                    "'MAP', 'SVI', 'HMC', "
                    "'MCMC' or 'emcee', 'zeus', 'Cobaya', "
                    "'dynesty', 'dyPolyChord',  'Multinest', 'Nautilus, '"
                    "'psf_iteration', 'restart', 'update_settings', 'calibrate_images' or "
                    "'align_images'".format(fitting_type)
                )

        return chain_list

    def best_fit(self, bijective=False):
        """

        :param bijective: bool, if True, the mapping of image2source_plane and the mass_scaling parameterisation
         are inverted. If you do not use those options, there is no effect.
        :return: best fit model of the current state of the FittingSequence class
        """

        return self._updateManager.best_fit(bijective=bijective)

    def update_state(self, kwargs_update):
        """Updates current best fit state to the input model keywords specified.

        :param kwargs_update: format of kwargs_result
        :return: None
        """
        self._updateManager.update_param_state(**kwargs_update)

    def best_fit_likelihood(self, verbose=False):
        """Returns the log likelihood of the best fit model of the current state of this
        class.

        :param verbose: bool, if True, prints likelihood statements
        :return: log likelihood, float
        """
        kwargs_result = self.best_fit(bijective=True)
        param_class = self.param_class
        likelihood_class = self.likelihood_class
        logL = likelihood_class.logL(
            param_class.kwargs2args(**kwargs_result), verbose=verbose
        )
        return logL

    @property
    def bic(self):
        """Bayesian information criterion (BIC) of the model.

        :return: bic value, float
        """
        num_data = self.likelihood_class.num_data
        num_param_nonlinear = self.param_class.num_param()[0]
        num_param_linear = self.param_class.num_param_linear()
        num_param = num_param_nonlinear + num_param_linear
        bic = analysis_util.bic_model(
            self.best_fit_likelihood(verbose=False), num_data, num_param
        )
        return bic

    @property
    def param_class(self):
        """

        :return: Param() class instance reflecting the current state of FittingSequence
        """
        return self._updateManager.param_class

    @property
    def likelihood_class(self):
        """

        :return: Likelihood() class instance reflecting the current state of FittingSequence
        """
        kwargs_model_copy = copy.deepcopy(self._updateManager.kwargs_model)

        # If Shapelets are present, extract n_max from kwargs_fixed, place into profile_kwargs_list to
        # set n_max at initialization. Required for jaxtronomy
        for shapelet_profile in ["SHAPELETS", "SHAPELETS_ELLIPSE"]:
            light_model_list = kwargs_model_copy.get("source_light_model_list", [])
            if shapelet_profile in light_model_list:
                index = light_model_list.index(shapelet_profile)
                num_profiles = len(light_model_list)
                profile_kwargs_list = kwargs_model_copy.get(
                    "source_light_profile_kwargs_list",
                    [{} for _ in range(num_profiles)],
                )
                n_max = self._updateManager.fixed_kwargs[1][index]["n_max"]
                profile_kwargs_list[index] = {"n_max": n_max}
                kwargs_model_copy["source_light_profile_kwargs_list"] = (
                    profile_kwargs_list
                )

            light_model_list = kwargs_model_copy.get("lens_light_model_list", [])
            if shapelet_profile in light_model_list:
                index = light_model_list.index(shapelet_profile)
                num_profiles = len(light_model_list)
                profile_kwargs_list = kwargs_model_copy.get(
                    "lens_light_profile_kwargs_list", [{} for _ in range(num_profiles)]
                )
                n_max = self._updateManager.fixed_kwargs[2][index]["n_max"]
                profile_kwargs_list[index] = {"n_max": n_max}
                kwargs_model_copy["lens_light_profile_kwargs_list"] = (
                    profile_kwargs_list
                )

        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_class = Likelihood(
            self.kwargs_data_joint,
            kwargs_model_copy,
            self.param_class,
            **kwargs_likelihood,
        )
        return likelihood_class

    @property
    def likelihood_class_lenstronomy(self):
        """

        :return: Likelihood() class instance reflecting the current state of FittingSequence
        """
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_class = Likelihood_lenstronomy(
            self.kwargs_data_joint,
            self._updateManager.kwargs_model,
            self.param_class,
            **kwargs_likelihood,
        )
        return likelihood_class

    def simplex(self, n_iterations, method="Nelder-Mead"):
        """Downhill simplex optimization using the Nelder-Mead algorithm.

        :param n_iterations: maximum number of iterations to perform
        :param method: the optimization method used, see documentation in
            scipy.optimize.minimize
        :return: result of the best fit
        """

        param_class = self.param_class
        kwargs_temp = self._updateManager.parameter_state
        init_pos = param_class.kwargs2args(**kwargs_temp)
        sampler = Sampler(likelihood_class=self.likelihood_class)
        result = sampler.simplex(init_pos, n_iterations, method)

        kwargs_result = param_class.args2kwargs(result, bijective=True)
        return kwargs_result

    def mcmc(
        self,
        n_burn,
        n_run,
        walkerRatio=None,
        n_walkers=None,
        sigma_scale=1,
        threadCount=1,
        vectorization_batch_size=None,
        init_samples=None,
        re_use_samples=True,
        sampler_type="emcee",
        progress=True,
        backend_filename=None,
        start_from_backend=False,
        flatten_chains=True,
        **kwargs_zeus,
    ):
        """MCMC routine.

        :param n_burn: number of burn in iterations (will not be saved)
        :param n_run: number of MCMC iterations that are saved
        :param walkerRatio: ratio of walkers/number of free parameters
        :param n_walkers: integer, number of walkers of emcee (optional, if set, overwrites the walkerRatio input
        :param sigma_scale: scaling of the initial parameter spread relative to the width in the initial settings
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :param vectorization_batch_size: int, only relevant for GPU, determines the number of
            particles/walkers whose logL will be computed simultaneously. None defaults to one
            particle at a time and 0 means to compute all particles/walkers simultaneously.
        :param init_samples: initial sample from where to start the MCMC process
        :param re_use_samples: bool, if True, re-uses the samples described in init_samples.nOtherwise starts from
         scratch.
        :param sampler_type: string, which MCMC sampler to be used. Options are 'emcee' and 'zeus'
        :param progress: boolean, if True shows progress bar in EMCEE
        :param backend_filename: name of the HDF5 file where sampling state is saved (through emcee backend engine)
        :type backend_filename: string
        :param start_from_backend: if True, start from the state saved in `backup_filename`.
         O therwise, create a new backup file with name `backup_filename` (any already existing file is overwritten!).
        :type start_from_backend: bool
        :param flatten_chains: If true, flattens MCMC chains to have shape [n_run * n_walkers, ndim]. Otherwise,
            unflattened MCMC chains will have shape [n_run, n_walkers, ndim]
        :type flatten_chains: bool
        :param kwargs_zeus: zeus-specific kwargs
        :return: list of output arguments, e.g. MCMC samples, parameter names, logL distances of all samples specified
         by the specific sampler used
        """
        param_class = self.param_class
        mcmc_class = Sampler(likelihood_class=self.likelihood_class)
        kwargs_temp = self._updateManager.parameter_state
        mean_start = param_class.kwargs2args(**kwargs_temp)
        kwargs_sigma = self._updateManager.sigma_kwargs
        sigma_start = np.array(param_class.kwargs2args(**kwargs_sigma)) * sigma_scale
        num_param, param_list = param_class.num_param()
        if n_walkers is None:
            if walkerRatio is None:
                raise ValueError(
                    "MCMC sampler needs either n_walkers or walkerRatio as input argument"
                )
            n_walkers = num_param * walkerRatio
        # run MCMC
        if init_samples is not None and re_use_samples is True:
            num_samples, num_param_prev = np.shape(init_samples)
            if num_param_prev == num_param:
                print("re-using previous samples to initialize the next MCMC run.")
                idxs = np.random.choice(len(init_samples), n_walkers)
                initpos = init_samples[idxs]
            else:
                raise ValueError(
                    "Can not re-use previous MCMC samples as number of parameters have changed!"
                )
        else:
            initpos = None

        if sampler_type == "zeus":
            # check if zeus is specified, if not default to emcee
            samples, dist = mcmc_class.mcmc_zeus(
                n_walkers,
                n_run,
                n_burn,
                mean_start,
                sigma_start,
                mpi=self._mpi,
                threadCount=threadCount,
                progress=progress,
                initpos=initpos,
                backend_filename=backend_filename,
                flatten_chains=flatten_chains,
                **kwargs_zeus,
            )
            output = [sampler_type, samples, param_list, dist]
        else:
            # sample with emcee
            samples, dist = mcmc_class.mcmc_emcee(
                n_walkers,
                n_run,
                n_burn,
                mean_start,
                sigma_start,
                mpi=self._mpi,
                threadCount=threadCount,
                vectorization_batch_size=vectorization_batch_size,
                progress=progress,
                initpos=initpos,
                backend_filename=backend_filename,
                start_from_backend=start_from_backend,
                flatten_chains=flatten_chains,
            )
            output = [sampler_type, samples, param_list, dist]

        self._mcmc_init_samples = samples  # overwrites previous samples to continue from there in the next MCMC run
        return output

    def optax(
        self,
        num_chains,
        maxiter,
        tolerance=0.01,
        sigma_scale=1,
        rng_seed=None,
    ):
        """Uses Optax L-BFGS gradient descent to find best-fit parameters.

        :param num_chains: int, number of minimization chains to run
        :param maxiter: int, maximum number of iterations during gradient descent
            process
        :param tolerance: float, if |logL[i] - logL[i-1]| < tol three times in a row, the gradient descent is stopped
        :param sigma_scale: scales the standard deviation of the prior distribution
        :param rng_seed: int, used to generate random initial starting point from the
            prior distribution. If None, a random seed will be used.
        """
        if rng_seed is None:
            rng_seed = int(np.random.uniform(0, 1) * 1000000)

        param_class = self.param_class
        likelihood_module = self.likelihood_class

        # Coonverts kwargs to args for the mean, sigma, lower, and upper parameter values
        kwargs_temp = self._updateManager.parameter_state
        args_mean = param_class.kwargs2args(**kwargs_temp)

        kwargs_sigma = self._updateManager.sigma_kwargs
        args_sigma = param_class.kwargs2args(**kwargs_sigma) * sigma_scale

        kwargs_lower = self._updateManager._lower_kwargs
        args_lower = param_class.kwargs2args(*kwargs_lower)

        kwargs_upper = self._updateManager._upper_kwargs
        args_upper = param_class.kwargs2args(*kwargs_upper)

        # Initialize the solver class
        minimizer = OptaxMinimizer(
            logL_func=likelihood_module.logL,
            args_mean=args_mean,
            args_sigma=args_sigma,
            args_lower=args_lower,
            args_upper=args_upper,
            maxiter=maxiter,
        )

        # Runs the minimizer
        final_params = minimizer.run(
            num_chains=num_chains, tol=tolerance, rng_seed=rng_seed
        )

        # Print results
        kwargs_result = param_class.args2kwargs(final_params)
        print("best fit log_likelihood:", likelihood_module.logL(final_params))
        print("Final parameters:", kwargs_result)

        return kwargs_result

    # ------------------------------------------------------------------
    # gradient-based inference with numpyro: MAP, SVI and HMC
    # ------------------------------------------------------------------
    def _numpyro_sampler(self):
        """NumpyroSampler on the current likelihood and parameter bounds, with the
        current parameter state, its spread, and the parameter names."""
        param_class = self.param_class
        args_lower = param_class.kwargs2args(*self._updateManager._lower_kwargs)
        args_upper = param_class.kwargs2args(*self._updateManager._upper_kwargs)
        sampler = NumpyroSampler(
            self.likelihood_class.logL, args_lower=args_lower, args_upper=args_upper
        )
        args_init = np.array(
            param_class.kwargs2args(**self._updateManager.parameter_state)
        )
        args_sigma = np.array(
            param_class.kwargs2args(**self._updateManager.sigma_kwargs)
        )
        _, param_list = param_class.num_param()
        return sampler, args_init, args_sigma, param_list

    def _stored_covariance(self, param_list, step_name):
        """Covariance left by the last MAP (Laplace) or SVI step, if it belongs to the
        same sampled parameters."""
        if self._numpyro_covariance is None:
            return None
        cov_param_list, cov = self._numpyro_covariance
        if list(cov_param_list) != list(param_list):
            print(
                "%s: the sampled parameters changed since the last MAP/SVI step; "
                "not using its covariance." % step_name
            )
            return None
        return cov

    @staticmethod
    def _rng_seed(rng_seed):
        if rng_seed is None:
            return int(np.random.randint(0, 2**31 - 1))
        return rng_seed

    def map(
        self,
        num_starts=1,
        maxiter=1000,
        tolerance=1e-3,
        sigma_scale=1,
        laplace=True,
        rng_seed=None,
        parallel=None,
    ):
        """Maximum a posteriori fit with multi-start L-BFGS in numpyro's unbounded
        parameter space. The first start is the current parameter state; the others are
        drawn around it with the initial spread (``kwargs_sigma``) times
        ``sigma_scale``.

        :param num_starts: int, number of L-BFGS runs
        :param maxiter: int, maximum number of iterations per run
        :param tolerance: float, a run stops when -logL improves by less than this
            three iterations in a row
        :param sigma_scale: float, scales the spread of the additional starts
        :param laplace: bool, also compute the Laplace (inverse Hessian) covariance,
            which is reported and seeds the HMC mass matrix if no SVI step follows
        :param rng_seed: int or None
        :param parallel: bool or None, run the starts at the same time, one per
            device (GPU); None does so whenever there is more than one device
        :return: ['MAP', samples (1, num_param), param_list, logL (1,), info dict]
        """
        sampler, args_init, args_sigma, param_list = self._numpyro_sampler()
        result = sampler.map(
            args_init,
            args_sigma=args_sigma * sigma_scale,
            num_starts=num_starts,
            maxiter=maxiter,
            tolerance=tolerance,
            laplace=laplace,
            rng_seed=self._rng_seed(rng_seed),
            parallel=parallel,
        )
        self._update_state(result["args"])
        if result.get("covariance_unconstrained") is not None:
            self._numpyro_covariance = (param_list, result["covariance_unconstrained"])

        info = {"logL_starts": result["logL_starts"]}
        for key in ["covariance", "covariance_unconstrained"]:
            if result.get(key) is not None:
                info[key] = result[key]
        return [
            "MAP",
            result["args"][np.newaxis, :],
            param_list,
            np.array([result["logL"]]),
            info,
        ]

    def svi(
        self,
        num_steps=5000,
        learning_rate=1e-3,
        num_particles=1,
        init_scale=1e-3,
        use_init_covariance=True,
        num_samples=2000,
        clip_norm=10.0,
        rng_seed=None,
        progress=True,
    ):
        """Stochastic variational inference with a full-rank Gaussian guide in
        numpyro's unbounded space, started at the current parameter state (e.g. the
        MAP). Updates the parameter state to the guide mean and stores the fitted
        covariance to seed the HMC mass matrix.

        :param use_init_covariance: bool, start the guide from the Laplace covariance
            of a previous MAP step on the same parameters, if there is one; otherwise
            from an isotropic Gaussian of width ``init_scale``
        See :meth:`NumpyroSampler.svi` for the other arguments.

        :return: ['SVI', samples (num_samples, num_param), param_list, logL, info dict]
        """
        sampler, args_init, _, param_list = self._numpyro_sampler()
        init_cov = (
            self._stored_covariance(param_list, "SVI") if use_init_covariance else None
        )
        result = sampler.svi(
            args_init,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_particles=num_particles,
            init_scale=init_scale,
            init_covariance_unconstrained=init_cov,
            num_samples=num_samples,
            clip_norm=clip_norm,
            rng_seed=self._rng_seed(rng_seed),
            progress=progress,
        )
        self._update_state(result["mean"])
        self._numpyro_covariance = (param_list, result["covariance_unconstrained"])

        info = {
            key: result[key]
            for key in ["losses", "mean", "loc_unconstrained", "covariance_unconstrained"]
        }
        return ["SVI", result["samples"], param_list, result["logL"], info]

    def hmc(
        self,
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        chain_method="parallel",
        use_init_covariance=True,
        init_spread=1.0,
        dense_mass=True,
        target_accept_prob=0.8,
        max_tree_depth=10,
        thinning=1,
        rhat_target=None,
        check_every=200,
        min_samples=200,
        rng_seed=None,
        progress=True,
    ):
        """Hamiltonian Monte Carlo (NUTS), started at the current parameter state. If a
        previous MAP or SVI step on the same parameters left a covariance, it seeds the
        mass matrix and spreads the chain starting points. Updates the parameter state
        to the sample with the highest likelihood.

        :param use_init_covariance: bool, use the covariance from a previous MAP or SVI
            step; otherwise adaptation starts from the identity
        See :meth:`NumpyroSampler.hmc` for the other arguments. With
        ``chain_method='parallel'`` each chain runs on its own device (GPU). With
        ``rhat_target`` (e.g. 1.01), sampling stops once the chains have converged and
        ``num_samples`` is the maximum number of draws per chain.

        :return: ['HMC', samples (num_chains, num_samples, num_param), param_list,
            logL (num_chains, num_samples), info dict]
        """
        sampler, args_init, _, param_list = self._numpyro_sampler()
        init_cov = (
            self._stored_covariance(param_list, "HMC") if use_init_covariance else None
        )
        result = sampler.hmc(
            args_init,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            init_covariance_unconstrained=init_cov,
            init_spread=init_spread,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            thinning=thinning,
            rhat_target=rhat_target,
            check_every=check_every,
            min_samples=min_samples,
            rng_seed=self._rng_seed(rng_seed),
            progress=progress,
        )
        samples, logL = result["samples"], result["logL"]
        kwargs_result = self.best_fit_from_samples(samples, logL)
        self._updateManager.update_param_state(**kwargs_result)
        self._mcmc_init_samples = samples.reshape(-1, samples.shape[-1])

        info = {
            key: result[key]
            for key in [
                "diverging",
                "num_steps",
                "r_hat",
                "n_eff",
                "inverse_mass_matrix",
                "step_size",
            ]
        }
        return ["HMC", samples, param_list, logL, info]

    def pso(
        self,
        n_particles,
        n_iterations,
        sigma_scale=1,
        print_key="PSO",
        threadCount=1,
        vectorization_batch_size=None,
    ):
        """Particle Swarm Optimization.

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param sigma_scale: scaling of the initial parameter spread relative to the
            width in the initial settings
        :param print_key: string, printed text when executing this routine
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :param rng_seed: int, seed used for randomness in PSO. If None, a random seed is
            generated.
        :param vectorization_batch_size: int, only relevant for GPU, determines the
            number of particles/walkers whose logL will be computed simultaneously. None
            defaults to one particle at a time and 0 means to compute all
            particles/walkers simultaneously.
        :return: result of the best fit, the PSO chain of the best fit parameter after
            each iteration [lnlikelihood, parameters, velocities], list of parameters in
            same order as in chain
        """

        param_class = self.param_class
        kwargs_temp = self._updateManager.parameter_state
        init_pos = param_class.kwargs2args(**kwargs_temp)
        kwargs_sigma = self._updateManager.sigma_kwargs
        sigma_start = param_class.kwargs2args(**kwargs_sigma)
        lower_start = np.array(init_pos) - np.array(sigma_start) * sigma_scale
        upper_start = np.array(init_pos) + np.array(sigma_start) * sigma_scale

        lower_limit = param_class.kwargs2args(**self._updateManager.lower_kwargs)
        upper_limit = param_class.kwargs2args(**self._updateManager.upper_kwargs)
        lower_start = np.maximum(lower_start, lower_limit)
        upper_start = np.minimum(upper_start, upper_limit)

        num_param, param_list = param_class.num_param()
        # run PSO
        sampler = Sampler(likelihood_class=self.likelihood_class)
        result, chain = sampler.pso(
            n_particles,
            n_iterations,
            lower_start,
            upper_start,
            init_pos=init_pos,
            threadCount=threadCount,
            vectorization_batch_size=vectorization_batch_size,
            mpi=self._mpi,
            print_key=print_key,
            verbose=self._verbose,
        )
        kwargs_result = param_class.args2kwargs(result, bijective=True)
        return kwargs_result, chain, param_list

    def nested_sampling(
        self,
        sampler_type="dynesty",
        kwargs_run={},
        prior_type="uniform",
        width_scale=1,
        sigma_scale=1,
        output_basename="chain",
        remove_output_dir=True,
        dypolychord_dynamic_goal=0.8,
        polychord_settings={},
        dypolychord_seed_increment=200,
        output_dir="nested_sampling_chains",
        dynesty_bound="multi",
        dynesty_sample="auto",
    ):
        """Run (Dynamic) Nested Sampling algorithms, depending on the type of algorithm.

        :param sampler_type: 'MULTINEST', 'DYPOLYCHORD', 'DYNESTY'
        :param kwargs_run: keywords passed to the core sampling method
        :param prior_type: 'uniform' of
            'gaussian', for converting the unit hypercube to param cube :param width_scale:
            scale the width (lower/upper limits) of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by
            this factor
        :param output_basename: name of the folder in which the core
            MultiNest/PolyChord code will save output files
        :param remove_output_dir: if True, the above folder is removed after completion
        :param dypolychord_dynamic_goal: dynamic goal for DyPolyChord (trade-off between
            evidence (0) and posterior (1) computation) :param polychord_settings: settings
            dictionary to send to pypolychord. Check dypolychord documentation for details.
        :param dypolychord_seed_increment: seed increment for dypolychord with MPI.
            Check dypolychord documentation for details.
        :param dynesty_bound: see https://dynesty.readthedocs.io
        :param sampler_type: 'MULTINEST', 'DYPOLYCHORD', 'DYNESTY'
        :param kwargs_run: keywords passed to the core sampling method
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to
            param cube
        :param width_scale: scale the width (lower/upper limits) of the parameters space
            by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by
            this factor
        :param output_basename: name of the folder in which the core MultiNest/PolyChord
            code will save output files
        :param remove_output_dir: if True, the above folder is removed after completion
        :param dypolychord_dynamic_goal: dynamic goal for DyPolyChord (trade-off between
            evidence (0) and posterior (1) computation)
        :param polychord_settings: settings dictionary to send to pypolychord. Check
            dypolychord documentation for details.
        :param dypolychord_seed_increment: seed increment for dypolychord with MPI.
            Check dypolychord documentation for details.
        :param dynesty_bound: see https://dynesty.readthedocs.io for details
        :param dynesty_sample: see https://dynesty.readthedocs.io for details
        :return: list of output arguments : samples, mean inferred values, log-
            likelihood, log-evidence, error on log-evidence for each sample
        """
        mean_start, sigma_start = self._prepare_sampling(prior_type)

        if sampler_type == "dyPolyChord":
            if "resume_dyn_run" in kwargs_run and kwargs_run["resume_dyn_run"] is True:
                # resume_dyn_run = True
                raise ValueError("resume_dyn_run feature doesn't work")
            else:
                resume_dyn_run = False
            sampler = DyPolyChordSampler(
                self.likelihood_class,
                prior_type=prior_type,
                prior_means=mean_start,
                prior_sigmas=sigma_start,
                width_scale=width_scale,
                sigma_scale=sigma_scale,
                output_dir=output_dir,
                output_basename=output_basename,
                polychord_settings=polychord_settings,
                remove_output_dir=remove_output_dir,
                resume_dyn_run=resume_dyn_run,
                use_mpi=self._mpi,
            )
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(
                dypolychord_dynamic_goal, kwargs_run
            )

        elif sampler_type == "MultiNest":
            sampler = MultiNestSampler(
                self.likelihood_class,
                prior_type=prior_type,
                prior_means=mean_start,
                prior_sigmas=sigma_start,
                width_scale=width_scale,
                sigma_scale=sigma_scale,
                output_dir=output_dir,
                output_basename=output_basename,
                remove_output_dir=remove_output_dir,
                use_mpi=self._mpi,
            )
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(
                kwargs_run
            )
        else:
            sampler = DynestySampler(
                self.likelihood_class,
                prior_type=prior_type,
                prior_means=mean_start,
                prior_sigmas=sigma_start,
                width_scale=width_scale,
                sigma_scale=sigma_scale,
                bound=dynesty_bound,
                sample=dynesty_sample,
                use_mpi=self._mpi,
            )
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(
                kwargs_run
            )

        # update current best fit values
        self._update_state(samples[-1])

        output = [
            sampler_type,
            samples,
            sampler.param_names,
            logL,
            logZ,
            logZ_err,
            results_object,
        ]

        return output

    def psf_iteration(self, compute_bands=None, **kwargs_psf_iter):
        """Iterative PSF reconstruction.

        :param compute_bands: bool list, if multiple bands, this process can be limited
            to a subset of bands
        :param kwargs_psf_iter: keyword arguments as used or available in
            PSFIteration.update_iterative() definition
        :return: 0, updated PSF is stored in self.multi_band_list
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_mask_list = kwargs_likelihood.get("image_likelihood_mask_list", None)
        kwargs_pixelbased = kwargs_likelihood.get("kwargs_pixelbased", None)
        kwargs_temp = self.best_fit(bijective=False)

        # Since psf_iteration happens through lenstronomy, we have to convert
        # all JAX arrays back into numpy arrays to avoid conflict with numba
        kwargs_temp_numpy = jax.tree.map(jax.device_get, kwargs_temp)

        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for band_index in range(len(self.multi_band_list)):
            if compute_bands[band_index] is True:
                kwargs_psf = self.multi_band_list[band_index][1]
                kwargs_psf_before = copy.deepcopy(kwargs_psf)
                image_model = SingleBandMultiModel(
                    self.multi_band_list,
                    kwargs_model,
                    likelihood_mask_list=likelihood_mask_list,
                    band_index=band_index,
                    kwargs_pixelbased=kwargs_pixelbased,
                )
                psf_iter = PsfFitting(image_model_class=image_model)

                kwargs_psf = psf_iter.update_iterative(
                    kwargs_psf, kwargs_params=kwargs_temp_numpy, **kwargs_psf_iter
                )
                self.multi_band_list[band_index][1] = kwargs_psf
                self._psf_iteration_memory.append(
                    {
                        "sequence": self._psf_iteration_index,
                        "band": band_index,
                        "psf_before": kwargs_psf_before,
                        "psf_after": kwargs_psf,
                    }
                )
        self._psf_iteration_index += 1
        return 0

    def align_images(
        self,
        n_particles=10,
        n_iterations=10,
        align_offset=True,
        align_rotation=False,
        threadCount=1,
        compute_bands=None,
        delta_shift=0.2,
        delta_rot=0.1,
    ):
        """Aligns the coordinate systems of different exposures within a fixed model
        parameterisation by executing a PSO with relative coordinate shifts as free
        parameters.

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param align_offset: aligns shift in Ra and Dec
        :type align_offset: boolean
        :param align_rotation: aligns coordinate rotation
        :type align_rotation: boolean
        :param delta_shift: astrometric shift tolerance
        :param delta_rot: rotation angle tolerance [in radian]
        :param compute_bands: bool list, if multiple bands, this process can be limited
            to a subset of bands for which the coordinate system is being fit for best
            alignment to the model parameters
        :return: 0, updated coordinate system for the band(s)
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_mask_list = kwargs_likelihood.get("image_likelihood_mask_list", None)
        kwargs_temp = self.best_fit(bijective=False)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for i in range(len(self.multi_band_list)):
            if compute_bands[i] is True:
                alignmentFitting = AlignmentFitting(
                    self.multi_band_list,
                    kwargs_model,
                    kwargs_temp,
                    band_index=i,
                    likelihood_mask_list=likelihood_mask_list,
                    align_offset=align_offset,
                    align_rotation=align_rotation,
                )

                kwargs_data, chain = alignmentFitting.pso(
                    n_particles=n_particles,
                    n_iterations=n_iterations,
                    delta_shift=delta_shift,
                    delta_rot=delta_rot,
                    threadCount=threadCount,
                    mpi=self._mpi,
                    print_key="Alignment fitting for band %s ..." % i,
                )
                print("Align completed for band %s." % i)
                print(
                    "ra_shift: %s,  dec_shift: %s, phi_rot: %s"
                    % (
                        kwargs_data.get("ra_shift", 0),
                        kwargs_data.get("dec_shift", 0),
                        kwargs_data.get("phi_rot", 0),
                    )
                )
                self.multi_band_list[i][0] = kwargs_data
        return 0

    def update_settings(
        self,
        kwargs_model=None,
        kwargs_constraints=None,
        kwargs_likelihood=None,
        lens_add_fixed=None,
        source_add_fixed=None,
        lens_light_add_fixed=None,
        ps_add_fixed=None,
        special_add_fixed=None,
        tracer_source_add_fixed=None,
        lens_remove_fixed=None,
        source_remove_fixed=None,
        lens_light_remove_fixed=None,
        ps_remove_fixed=None,
        special_remove_fixed=None,
        tracer_source_remove_fixed=None,
        change_source_lower_limit=None,
        change_source_upper_limit=None,
        change_lens_lower_limit=None,
        change_lens_upper_limit=None,
        change_sigma_lens=None,
        change_sigma_source=None,
        change_sigma_lens_light=None,
    ):
        """Updates lenstronomy settings "on the fly".

        :param kwargs_model: kwargs, specified keyword arguments overwrite the existing ones
        :param kwargs_constraints: kwargs, specified keyword arguments overwrite the existing ones
        :param kwargs_likelihood: kwargs, specified keyword arguments overwrite the existing ones
        :param lens_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param source_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param ps_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param special_add_fixed: ['param1', 'param2',...]
        :param special_add_fixed: ['param1', 'param2',...]
        :param tracer_source_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param source_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param ps_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param special_remove_fixed: ['param1', 'param2',...]
        :param special_remove_fixed: ['param1', 'param2',...]
        :param tracer_source_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param change_lens_lower_limit: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_lens_upper_limit: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_source_lower_limit: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_source_upper_limit: [[i_model, [''param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_sigma_lens: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_sigma_source: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_sigma_lens_light: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :return: 0, the settings are overwritten for the next fitting step to come
        """
        self._updateManager.update_options(
            kwargs_model, kwargs_constraints, kwargs_likelihood
        )
        self._updateManager.update_fixed(
            lens_add_fixed,
            source_add_fixed,
            lens_light_add_fixed,
            ps_add_fixed,
            special_add_fixed,
            tracer_source_add_fixed,
            lens_remove_fixed,
            source_remove_fixed,
            lens_light_remove_fixed,
            ps_remove_fixed,
            special_remove_fixed,
            tracer_source_remove_fixed,
        )
        self._updateManager.update_limits(
            change_source_lower_limit,
            change_source_upper_limit,
            change_lens_lower_limit,
            change_lens_upper_limit,
        )
        self._updateManager.update_sigmas(
            change_sigma_lens=change_sigma_lens,
            change_sigma_source=change_sigma_source,
            change_sigma_lens_light=change_sigma_lens_light,
        )
        return 0

    def set_amplitudes(self):
        """Overwrites all current amplitude parameters with values obtained from calling
        the linear solver on the current parameter state.

        For multi-band fitting, any light models present in multiple bands will have its
        amplitude parameter set to the result from calling the linear solver on the last
        band that it is present in.
        """

        # Extract kwargs from current parameter state (and convert JAX arrays to numpy to avoid conflicts with numba)
        kwargs_temp = self.best_fit()
        kwargs_temp_numpy = jax.tree.map(jax.device_get, kwargs_temp)

        kwargs_lens = kwargs_temp_numpy["kwargs_lens"]
        kwargs_source = kwargs_temp_numpy["kwargs_source"]
        kwargs_lens_light = kwargs_temp_numpy["kwargs_lens_light"]
        kwargs_ps = kwargs_temp_numpy["kwargs_ps"]
        kwargs_special = kwargs_temp_numpy["kwargs_special"]
        kwargs_extinction = kwargs_temp_numpy["kwargs_extinction"]
        kwargs_tracer_source = kwargs_temp_numpy["kwargs_tracer_source"]

        # We have to use lenstronomy's likelihood class since the jit-compiled JAX
        # version does not do in-place updates on external dictionaries
        im_sim_class = self.likelihood_class_lenstronomy.image_likelihood.imSim

        # Call linear solver to obtain amplitudes; the function update_linear_kwargs
        # is called inside, which automatically updates all dictionaries
        im_sim_class.image_linear_solve(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
        )
        self._updateManager.update_param_state(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_special,
            kwargs_extinction,
            kwargs_tracer_source,
        )
        print("updated kwargs:", self.best_fit())

    def set_param_value(self, **kwargs):
        """Set a parameter to a specific value. `kwargs` are below.

        :param lens: [[i_model, ['param1', 'param2',...], [...]]
        :type lens:
        :param source: [[i_model, ['param1', 'param2',...], [...]]
        :type source:
        :param lens_light: [[i_model, ['param1', 'param2',...], [...]]
        :type lens_light:
        :param ps: [[i_model, ['param1', 'param2',...], [...]]
        :type ps:
        :return: 0, the value of the param is overwritten
        :rtype:
        """
        self._updateManager.update_param_value(**kwargs)

    def fix_not_computed(self, free_bands):
        """Fixes lens model parameters of imaging bands/frames that are not computed and
        frees the parameters of the other lens models to the initial kwargs_fixed
        options.

        :param free_bands: bool list of length of imaging bands in order of imaging
            bands, if False: set fixed lens model
        :return: None
        """
        self._updateManager.fix_not_computed(free_bands=free_bands)

    def _prepare_sampling(self, prior_type):
        if prior_type == "gaussian":
            mean_start = self.param_class.kwargs2args(
                **self._updateManager.parameter_state
            )
            sigma_start = self.param_class.kwargs2args(
                **self._updateManager.sigma_kwargs
            )
            mean_start = np.array(mean_start)
            sigma_start = np.array(sigma_start)
        else:
            mean_start, sigma_start = None, None
        return mean_start, sigma_start

    def _update_state(self, result):
        """

        :param result: array of parameters being sampled (e.g. result of MCMC chain)
        :return: None, updates the parameter state of the class instance
        """
        kwargs_result = self.param_class.args2kwargs(result, bijective=True)
        self._updateManager.update_param_state(**kwargs_result)

    def _result_from_mcmc(self, mcmc_output):
        """

        :param mcmc_output: list returned by self.mcmc()
        :return: kwargs_result like returned by self.pso(), from best logL MCMC sample
        """
        _, samples, _, logL_values = mcmc_output
        return self.best_fit_from_samples(samples, logL_values)

    def best_fit_from_samples(self, samples, logl):
        """Return best fit (max likelihood) value of samples in lenstronomy conventions.

        :param samples: samples of multi-dimensional parameter space
        :param logl: likelihood values for each sample
        :return: kwargs_result in lenstronomy convention
        """
        samples = np.asarray(samples)
        logl = np.asarray(logl)

        # get index of best logL sample, in the shape of logl
        best_fit_index = np.unravel_index(np.argmax(logl), logl.shape)
        # for flat input this is a 1-tuple (i,), for unflattened a 2-tuple (step, walker);
        # indexing samples with the tuple returns the n_dim vector in both cases
        best_fit_sample = samples[best_fit_index]
        best_fit_result = best_fit_sample.tolist()
        # get corresponding kwargs
        kwargs_result = self.param_class.args2kwargs(best_fit_result, bijective=True)
        return kwargs_result

    @property
    def psf_iteration_memory(self):
        """
        returns all the psf iterations performed in the FittingSequence
        It stores in a list of dictionaries:
        "sequence": what PSF sequence it is (0, 1 etc)
        "band": index of the imaging band that is being corrected
        "psf_before" kwargs_psf prior to the iteration
        "psf_after" kwargs_psf as a result of the iteration

        :return: list of all psf corrections
        """
        return self._psf_iteration_memory
