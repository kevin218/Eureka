import numpy as np
try:
    import pymc3 as pm
    import pymc3_ext as pmx
    import arviz
except ModuleNotFoundError:
    # PyMC3 hasn't been installed
    pass
from astropy import table

from .likelihood import computeRedChiSq, update_uncertainty
from . import plots_s5 as plots
from .fitters import group_variables, load_old_fitparams, save_fit
from ..lib.split_channels import get_trim


def exoplanetfitter(lc, model, meta, log, calling_function='exoplanet',
                    **kwargs):
    """Perform sampling using exoplanet.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    calling_function: str, optional
        The fitter that is being run (e.g. may be 'emcee' if running lsqfitter
        to initialize emcee walkers). Defailts to 'exoplanet'.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting

    Notes
    -----
    History:

    - April 6, 2022 Taylor Bell
        Initial version.
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars = group_variables(model)[0]
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'exoplanet')

    model.setup(lc.time, lc.flux, lc.unc, freepars)
    model.update(freepars)

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    # Plot starting point
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta,
                       fitter=calling_function+'StartingPoint')
        # Plot GP starting point
        if model.GP:
            plots.plot_GP_components(lc, model, meta,
                                     fitter=calling_function+'StartingPoint')

    # Plot star spots
    if 'spotrad' in model.longparamlist[0] and meta.isplots_S5 >= 3:
        plots.plot_starry_star(lc, model, meta,
                               fitter=calling_function+'StartingPoint')

    log.writelog('Running exoplanet optimizer...')
    with model.model:
        map_soln = pmx.optimize(start=start)

    # Get the best fit params
    fit_params = np.array([map_soln[name] for name in freenames])
    model.update(fit_params)
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    t_results = table.Table([freenames, fit_params],
                            names=("Parameter", "Mean"))

    # Save the fit ASAP
    save_fit(meta, lc, model, calling_function, t_results, freenames)

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nEXOPLANET RESULTS:')
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
    if 'spotrad' in model.longparamlist[0] and meta.isplots_S5 >= 3:
        plots.plot_starry_star(lc, model, meta, fitter=calling_function)

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter=calling_function)

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('Y10' in freenames or 'Y11' in freenames
                                 or 'sinusoid_pc' in meta.run_myfuncs
                                 or 'poet_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter=calling_function)

    if meta.pixelsampling and meta.isplots_S5 >= 1:
        eclipse_maps = map_soln['map'][np.newaxis]
        plots.plot_eclipse_map(lc, eclipse_maps, meta, fitter=calling_function)

    # Make RMS time-averaging plot
    if meta.isplots_S5 >= 3 and calling_function == 'exoplanet':
        # This plot is only really useful if you're actually using the
        # exoplanet fitter, otherwise don't make it
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    # Plot residuals distribution
    if meta.isplots_S5 >= 3 and calling_function == 'exoplanet':
        plots.plot_res_distr(lc, model, meta, fitter=calling_function)

    # Make a new model instance
    model.chi2red = chi2red
    model.fit_params = fit_params

    return model


def nutsfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using PyMC3 NUTS sampler.

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

    Notes
    -----
    History:

    - October 5, 2022 Taylor Bell
        Initial version.
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars = group_variables(model)[0]
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'nuts')
    ndim = len(freenames)

    model.setup(lc.time, lc.flux, lc.unc, freepars)
    model.update(freepars)

    if meta.exoplanet_first:
        # Only call exoplanet fitter first if asked
        log.writelog('\nCalling exoplanetfitter first...')
        # RUN exoplanet optimizer
        exo_sol = exoplanetfitter(lc, model, meta, log,
                                  calling_function='nuts_exoplanet', **kwargs)

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

    # Plot star spots
    if 'spotrad' in model.longparamlist[0] and meta.isplots_S5 >= 3:
        plots.plot_starry_star(lc, model, meta, fitter='nutsStartingPoint')

    log.writelog('Running PyMC3 NUTS sampler...')
    with model.model:
        trace = pmx.sample(tune=meta.tune, draws=meta.draws, start=start,
                           target_accept=meta.target_accept,
                           chains=meta.chains, cores=meta.ncpu)
        print()

        # Log detailed convergence and sampling statistics
        log.writelog('\nPyMC3 sampling statistics:', mute=(not meta.verbose))
        log.writelog(pm.summary(trace, var_names=freenames),
                     mute=(not meta.verbose))
        log.writelog('', mute=(not meta.verbose))

    trace_az = arviz.from_pymc3(trace, model=model.model)
    samples = np.hstack([trace[name].reshape(-1, 1) for name in freenames])

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

    log.writelog('\nPYMC3 NUTS RESULTS:')
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

    # Plot star spots
    if 'spotrad' in model.longparamlist[0] and meta.isplots_S5 >= 3:
        plots.plot_starry_star(lc, model, meta, fitter='nuts')

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='nuts')

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('Y10' in freenames or 'Y11' in freenames
                                 or 'sinusoid_pc' in meta.run_myfuncs
                                 or 'poet_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter='nuts')

    # Show the inferred planetary brightness map
    if meta.pixelsampling and meta.isplots_S5 >= 1:
        eclipse_maps = np.transpose(trace_az.posterior.stack(
            sample=("chain", "draw"))['map'][:], [2, 0, 1])
        plots.plot_eclipse_map(lc, eclipse_maps, meta, fitter='nuts')

    # Make RMS time-averaging plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='nuts')

    if meta.isplots_S5 >= 3:
        # Plot residuals distribution
        plots.plot_res_distr(lc, model, meta, fitter='nuts')

        # Plot trace evolution
        plots.plot_trace(trace, model, lc, freenames, meta)

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
