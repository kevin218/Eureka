import numpy as np
try:
    import jax
    import numpyro
    import numpyro_ext
    from numpyro import handlers
except ModuleNotFoundError:
    # jax hasn't been installed
    pass
from astropy import table

from .likelihood import computeRedChiSq, update_uncertainty
from . import plots_s5 as plots
from .fitters import group_variables, load_old_fitparams, save_fit
from ..lib.split_channels import get_trim


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
    freepars = group_variables(model)[0]
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'jaxopt')

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    # Set the model parameters to their starting values
    with handlers.seed(rng_seed=0):
        model.setup(lc.time, lc.flux, lc.unc, freepars)
    model.update(freepars)

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
