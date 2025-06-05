import numpy as np
import sys
from io import StringIO
import arviz
from astropy import table
try:
    import jax
    import numpyro
    import numpyro_ext
    from numpyro.infer import MCMC, NUTS
except ModuleNotFoundError:
    # jax hasn't been installed
    pass


from .likelihood import computeRedChiSq, update_uncertainty
from . import plots_s5 as plots
from .fitters import group_variables, load_old_fitparams, save_fit
from ..lib.split_channels import get_trim

from .likelihood import lnprob

def jaxoptfitter(lc, model, meta, log, calling_function='jaxopt',
                 **kwargs):
    """Perform sampling using numpyro_ext.optim.optimize.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.jax_models.CompositeJaxModel
        The composite model to fit
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    calling_function: str, optional
        The fitter that is being run (e.g., may be 'emcee' if running lsqfitter
        to initialize emcee walkers). Defailts to 'jaxopt'.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.jax_models.CompositeJaxModel
        The composite model after fitting
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars, prior1, prior2, priortype, indep_vars = \
        group_variables(model)
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'jaxopt')

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    # Set the model parameters to their starting values
    model.update(freepars)

    start_lnprob = lnprob(freepars, lc, model, prior1, prior2, priortype,
                          freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    # Plot starting point
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta,
                       fitter=calling_function+'StartingPoint')
        # Plot GP starting point
        if model.GP:
            plots.plot_GP_components(lc, model, meta,
                                     fitter=calling_function+'StartingPoint')

    # Plot star spots starting point
    if 'fleck_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_fleck_star(lc, model, meta,
                              fitter=calling_function+'StartingPoint')
    if 'jaxoplanet' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        if 'spotrad' in model.longparamlist[0]:
            plots.plot_starry_star(lc, model, meta,
                                   fitter=calling_function+'StartingPoint')

    # Plot Harmonica string starting point
    if 'harmonica_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_harmonica_string(lc, model, meta,
                                    fitter=calling_function+'StartingPoint')

    # Add some extra fitting controls
    optimizer = numpyro_ext.optim.JAXOptMinimize(
        method=meta.lsq_method, tol=meta.lsq_tol,
        maxiter=meta.lsq_maxiter,
    )
    # Initialize the optimizer
    run_optim = numpyro_ext.optim.optimize(
        model.setup,
        init_strategy=numpyro.infer.init_to_value(values=start),
        optimizer=optimizer
    )

    log.writelog('Running jaxopt optimizer...')
    map_soln = run_optim(jax.random.PRNGKey(0),
                         lc.time, lc.flux, lc.unc, freepars)

    # Get the best fit params
    fit_params = np.array([map_soln[name] for name in freenames])
    model.update(fit_params)
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    t_results = table.Table([freenames, fit_params],
                            names=("Parameter", "Mean"))

    # Save the fit ASAP
    save_fit(meta, lc, model, calling_function, t_results, freenames)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype,
                        freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nJAXOPT RESULTS:')
    for i in range(len(freenames)):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_ch')[-1].split('_')[0]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            trim1, trim2 = get_trim(meta.nints, chan)
            unc = np.ma.median(lc.unc[trim1:trim2])
            scatter_ppm = 1e6*fit_params[i]*unc
            log.writelog(f'{freenames[i]}: {fit_params[i]}; {scatter_ppm} ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]}')
    if meta.pixelsampling:
        log.writelog('\nSpherical Harmonic Basis:')

        for chan in range(model.nchannel_fitted):
            if chan == 0:
                chankey = ''
            else:
                chankey = f'_ch{chan}'
            if model.nchannel_fitted > 1:
                log.writelog(f'  Channel {chan+1}:')
                padding = '    '
            else:
                padding = '  '
            log.writelog(f'{padding}fp: {map_soln["fp"+chankey]}')
            for ell in range(1, meta.ydeg+1):
                for m in range(-ell, ell+1):
                    name = f'Y{ell}{m}'
                    log.writelog(f'{padding}{name}: '
                                 f'{map_soln[name+chankey][0]}')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter=calling_function)

    # Plot star spots
    if 'fleck_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_fleck_star(lc, model, meta,
                              fitter=calling_function)
    if 'jaxoplanet' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        if 'spotrad' in model.longparamlist[0]:
            plots.plot_starry_star(lc, model, meta,
                                   fitter=calling_function)

    # Plot Harmonica string
    if 'harmonica_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_harmonica_string(lc, model, meta, fitter=calling_function)

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter=calling_function)

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('Y10' in freenames or 'Y11' in freenames
                                 or 'sinusoid_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter=calling_function)

    # FINDME: Not yet implemented
    # if meta.pixelsampling and meta.isplots_S5 >= 1:
    #     eclipse_maps = map_soln['map'][np.newaxis]
    #     plots.plot_eclipse_map(lc, eclipse_maps, meta,
    #                            fitter=calling_function)

    # Make RMS time-averaging plot
    if meta.isplots_S5 >= 3 and calling_function == 'jaxopt':
        # This plot is only really useful if you're actually using the
        # jaxopt fitter, otherwise don't make it
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    # Plot residuals distribution
    if meta.isplots_S5 >= 3 and calling_function == 'jaxopt':
        plots.plot_res_distr(lc, model, meta, fitter=calling_function)

    # Make a new model instance
    model.chi2red = chi2red
    model.fit_params = fit_params

    return model


def nutsfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using numpyro's NUTS sampler.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta : eureka.lib.readECF.MetaClass
        The metadata object
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    **kwargs : dict
        Arbitrary keyword arguments.

    Returns
    -------
    best_model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars = group_variables(model)[0]
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'nuts')
    ndim = len(freenames)

    # Set the model parameters to their starting values
    model.update(freepars)

    if meta.jaxopt_first:
        # Only call exoplanet fitter first if asked
        log.writelog('\nCalling jaxoptfitter first...')
        # RUN exoplanet optimizer
        exo_sol = jaxoptfitter(lc, model, meta, log,
                               calling_function='nuts_jaxopt', **kwargs)

        freepars = exo_sol.fit_params
        model.update(freepars)

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    # Plot starting point
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta,
                       fitter='nutsStartingPoint')
        # Plot GP starting point
        if model.GP:
            plots.plot_GP_components(lc, model, meta,
                                     fitter='nutsStartingPoint')

    # Plot star spots starting point
    if 'fleck_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_fleck_star(lc, model, meta,
                              fitter='nutsStartingPoint')
    if 'jaxoplanet' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        if 'spotrad' in model.longparamlist[0]:
            plots.plot_starry_star(lc, model, meta,
                                   fitter='nutsStartingPoint')

    # Plot Harmonica string starting point
    if 'harmonica_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_harmonica_string(lc, model, meta,
                                    fitter='nutsStartingPoint')

    log.writelog('Running numpyro\'s NUTS sampler...')
    numpyro.set_host_device_count(meta.ncpu)
    kernel = NUTS(model.setup, dense_mass=meta.dense_mass,
                  init_strategy=numpyro.infer.init_to_value(values=start))
    mcmc = MCMC(sampler=kernel, num_warmup=meta.run_nburn,
                num_samples=meta.run_nsteps, num_chains=meta.chains,
                chain_method="parallel")
    mcmc.run(jax.random.PRNGKey(1), lc.time, lc.flux, lc.unc, freepars)
    samples = mcmc.get_samples()
    # Wait for multi-threaded execution to finish before continuing
    jax.block_until_ready(samples)

    # Log detailed convergence and sampling statistics
    sys.stderr.flush()
    sys.stdout.flush()
    log.writelog('\n\nNUTS sampling statistics:', mute=(not meta.verbose),
                 end='')
    # Need to temporarily redirect stdout since mcmc.print_summary() prints
    # to stdout rather than returning a string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    mcmc.print_summary()
    sys.stdout = old_stdout
    log.writelog(mystdout.getvalue(), mute=(not meta.verbose))

    # FINDME: Currently unused code, but could be used to make some other
    # potentially helpful figures
    # (see https://python.arviz.org/en/latest/api/plots.html)
    # posterior_predictive = numpyro.infer.Predictive(model.setup, samples)(
    #     jax.random.PRNGKey(2), lc.time, lc.flux, lc.unc, freepars)
    # prior = numpyro.infer.Predictive(model.setup, num_samples=500)(
    #     jax.random.PRNGKey(3), lc.time, lc.flux, lc.unc, freepars)
    # trace_az = arviz.from_numpyro(
    #     mcmc, prior=prior, posterior_predictive=posterior_predictive)
    trace_az = arviz.from_numpyro(mcmc)
    samples = np.hstack([samples[name].reshape(-1, 1) for name in freenames])

    # Record median + percentiles
    q = np.percentile(samples, [16, 50, 84], axis=0)
    fit_params = q[1]  # median
    mean_params = np.mean(samples, axis=0)
    errs = np.std(samples, axis=0)

    # Create table of results
    t_results = table.Table([freenames, mean_params, q[0]-q[1], q[2]-q[1],
                             q[0], fit_params, q[2]],
                            names=("Parameter", "Mean", "-1sigma", "+1sigma",
                                   "16th", "50th", "84th"))

    upper_errs = q[2]-q[1]
    lower_errs = q[1]-q[0]

    model.update(fit_params)
    model.errs = dict(zip(freenames, errs))
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    # Save the fit ASAP so plotting errors don't make you lose everything
    save_fit(meta, lc, model, 'nuts', t_results, freenames, samples)

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nNUTS RESULTS:')
    for i in range(ndim):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_ch')[-1].split('_')[0]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            trim1, trim2 = get_trim(meta.nints, chan)
            unc = np.ma.median(lc.unc[trim1:trim2])
            scatter_ppm = 1e6*fit_params[i]*unc
            scatter_ppm_upper = 1e6*upper_errs[i]*unc
            scatter_ppm_lower = 1e6*lower_errs[i]*unc
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]}); {scatter_ppm} '
                         f'(+{scatter_ppm_upper}, -{scatter_ppm_lower}) ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]})')
    if meta.pixelsampling:
        log.writelog('\nSpherical Harmonic Basis:')

        for chan in range(model.nchannel_fitted):
            if chan == 0:
                chankey = ''
            else:
                chankey = f'_ch{chan}'
            if model.nchannel_fitted > 1:
                log.writelog(f'  Channel {chan+1}:')
                padding = '    '
            else:
                padding = '  '

            othernames = ['fp', ]
            for ell in range(1, meta.ydeg+1):
                for m in range(-ell, ell+1):
                    othernames.append(f'Y{ell}{m}')

            for name in othernames:
                values = trace_az.posterior.stack(sample=("chain", "draw")
                                                  )[name+chankey][:]
                q = np.percentile(values, [16, 50, 84])
                medval = q[1]  # median
                uppererr = q[2]-q[1]
                lowererr = q[1]-q[0]
                log.writelog(f'{padding}{name}: {medval} (+{uppererr},'
                             f' -{lowererr})')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='nuts')

    # Plot star spots starting point
    if 'fleck_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_fleck_star(lc, model, meta, fitter='nuts')
    if 'jaxoplanet' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        if 'spotrad' in model.longparamlist[0]:
            plots.plot_starry_star(lc, model, meta, fitter='nuts')

    # Plot Harmonica string
    if 'harmonica_tr' in meta.run_myfuncs and meta.isplots_S5 >= 3:
        plots.plot_harmonica_string(lc, model, meta, fitter='nuts')

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='nuts')

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('Y10' in freenames or 'Y11' in freenames
                                 or 'sinusoid_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter='nuts')

    # FINDME: Not yet implemented
    # Show the inferred planetary brightness map
    # if meta.pixelsampling and meta.isplots_S5 >= 1:
    #     eclipse_maps = np.transpose(trace_az.posterior.stack(
    #         sample=("chain", "draw"))['map'][:], [2, 0, 1])
    #     plots.plot_eclipse_map(lc, eclipse_maps, meta, fitter='nuts')

    # Make RMS time-averaging plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='nuts')

    if meta.isplots_S5 >= 3:
        # Plot residuals distribution
        plots.plot_res_distr(lc, model, meta, fitter='nuts')

        # Plot trace evolution
        plots.plot_trace(trace_az, model, lc, freenames, meta)

    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='nuts')

        if meta.pixelsampling:
            freenames_temp = np.copy(freenames)
            samples_temp = np.copy(samples)
            for chan in range(model.nchannel_fitted):
                if chan == 0:
                    chankey = ''
                else:
                    chankey = f'_ch{chan}'

                # Grab all the Ylm values
                ylm_names = []
                for ell in range(1, meta.ydeg+1):
                    for m in range(-ell, ell+1):
                        ylm_names.append(f'Y{ell}{m}{chankey}')
                ylms = []
                for name in ylm_names:
                    # Grab all the Ylm values from the trace
                    ylms.append(trace_az.posterior.stack(
                        sample=("chain", "draw")
                    )[name][:].to_numpy().flatten())

                # Replace pixel values with Ylms for a second corner plot
                freenames_ylm = []
                samples_ylm = []
                for i, freename in enumerate(freenames_temp):
                    if f'pixel{chankey}' == freename:
                        # Replace pixel values with Ylm values
                        freenames_ylm.extend(ylm_names)
                        samples_ylm.extend(ylms)
                    elif (f'pixel{chankey}' not in freename or
                            (chan == 0 and model.nchannel_fitted > 1
                             and 'pixel_ch' in freename)):
                        # Keep non-pixel values (or pixel values from
                        # upcoming channels) as they were
                        freenames_ylm.append(freename)
                        samples_ylm.append(samples_temp[:, i])
                samples_ylm = np.array(samples_ylm).T

                # Update temp variables to prepare for next channel
                # (if relevant)
                freenames_temp = np.copy(freenames_ylm)
                samples_temp = np.copy(samples_ylm)

            plots.plot_corner(samples_ylm, lc, meta, freenames_ylm,
                              fitter='Ylm_nuts')

    # Make a new model instance
    model.chi2red = chi2red
    model.fit_params = fit_params

    return model
