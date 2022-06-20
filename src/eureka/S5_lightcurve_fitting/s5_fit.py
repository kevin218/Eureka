import numpy as np
import os
import time as time_pkg
import astraeus.xarrayIO as xrio
from ..lib import manageevent as me
from ..lib import readECF
from ..lib import util, logedit
from ..lib.readEPF import Parameters
from . import lightcurve as lc
from . import models as m


def fitlc(eventlabel, ecf_path=None, s4_meta=None):
    '''Fits 1D spectra with various models and fitters.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored.
        Defaults to None which resolves to './'.
    s4_meta : eureka.lib.readECF.MetaClass; optional
        The metadata object from Eureka!'s S4 step (if running S4 and S5
        sequentially). Defaults to None.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object with attributes added by S5.

    Notes
    -----
    History:

    - November 12-December 15, 2021 Megan Mansfield
        Original version
    - December 17-20, 2021 Megan Mansfield
        Connecting S5 to S4 outputs
    - December 17-20, 2021 Taylor Bell
        Increasing connectedness of S5 and S4
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - January - February, 2022 Eva-Maria Ahrer
        Adding GP functionality
    - April 2022 Kevin Stevenson
        Enabled Astraeus
    '''
    print("\nStarting Stage 5: Light Curve Fitting\n")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S5_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel
    meta.datetime = time_pkg.strftime('%Y-%m-%d')

    if s4_meta is None:
        # Locate the old MetaClass savefile, and load new ECF into
        # that old MetaClass
        s4_meta, meta.inputdir, meta.inputdir_raw = \
            me.findevent(meta, 'S4', allowFail=False)
    else:
        # Running these stages sequentially, so can safely assume
        # the path hasn't changed
        meta.inputdir = s4_meta.outputdir
        meta.inputdir_raw = meta.inputdir[len(meta.topdir):]

    meta = me.mergeevents(meta, s4_meta)

    if not meta.allapers:
        # The user indicated in the ecf that they only want to consider one
        # aperture in which case the code will consider only the one which
        # made s4_meta. Alternatively, if S4 was run without allapers, S5
        # will already only consider that one
        meta.spec_hw_range = [meta.spec_hw, ]
        meta.bg_hw_range = [meta.bg_hw, ]

    if meta.testing_S5:
        # Only fit a single channel while testing unless doing a shared fit,
        # then do two
        chanrng = 1
    else:
        chanrng = meta.nspecchan

    # Create directories for Stage 5 outputs
    meta.run_s5 = None
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            meta.run_s5 = util.makedirectory(meta, 'S5', meta.run_s5,
                                             ap=spec_hw_val, bg=bg_hw_val)

    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            # Load in the S4 metadata used for this particular aperture pair
            meta = load_specific_s4_meta_info(meta)

            lc = xrio.readXR(meta.filename_S4_LCData)

            # Get the directory for Stage 5 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S5', meta.run_s5,
                                                ap=spec_hw_val, bg=bg_hw_val)

            # Copy existing S4 log file and resume log
            meta.s5_logname = meta.outputdir + 'S5_' + meta.eventlabel + ".log"
            log = logedit.Logedit(meta.s5_logname, read=meta.s4_logname)
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ECF
            log.writelog('Copying S5 control file', mute=(not meta.verbose))
            meta.copy_ecf()

            # Set the intial fitting parameters
            params = Parameters(meta.folder, meta.fit_par)
            # Copy EPF
            log.writelog('Copying S5 parameter control file',
                         mute=(not meta.verbose))
            params.write(meta.outputdir)
            sharedp = False
            for arg, val in params.dict.items():
                if 'shared' in val:
                    sharedp = True
            meta.sharedp = sharedp

            if meta.sharedp and meta.testing_S5:
                chanrng = min([2, meta.nspecchan])

            # Subtract off the user provided time value to avoid floating
            # point precision problems when fitting for values like t0
            offset = params.time_offset.value
            time = lc.time.values - offset
            if offset != 0:
                time_units = lc.data.attrs['time_units']+f' - {offset}'
            else:
                time_units = lc.data.attrs['time_units']
            meta.time = lc.time.values

            if sharedp:
                # Make a long list of parameters for each channel
                longparamlist, paramtitles = make_longparamlist(meta, params,
                                                                chanrng)

                log.writelog(f"\nStarting Shared Fit of {chanrng} Channels\n")

                flux = np.ma.masked_array([])
                flux_err = np.ma.masked_array([])
                for channel in range(chanrng):
                    # FINDME: need to consider optmask
                    flux = np.ma.append(flux,
                                        (lc.data.values[channel, :] /
                                         np.nanmean(
                                             lc.data.values[channel, :])))
                    flux_err = np.ma.append(flux_err,
                                            (lc.err.values[channel, :] /
                                             np.nanmean(
                                                 lc.data.values[channel, :])))

                meta = fit_channel(meta, time, flux, 0, flux_err, eventlabel,
                                   sharedp, params, log, longparamlist,
                                   time_units, paramtitles, chanrng)

                # Save results
                log.writelog('Saving results')
                me.saveevent(meta, (meta.outputdir+'S5_'+meta.eventlabel +
                                    "_Meta_Save"), save=[])
            else:
                for channel in range(chanrng):
                    # Make a long list of parameters for each channel
                    longparamlist, paramtitles = make_longparamlist(meta,
                                                                    params,
                                                                    chanrng)

                    log.writelog(f"\nStarting Channel {channel+1} of "
                                 f"{chanrng}\n")

                    # Get the flux and error measurements for
                    # the current channel
                    flux = lc.data.values[channel, :]
                    flux_err = lc.err.values[channel, :]

                    # Normalize flux and uncertainties to avoid large
                    # flux values
                    flux_err = flux_err/np.nanmean(flux)
                    flux = flux/np.nanmean(flux)

                    meta = fit_channel(meta, time, flux, channel, flux_err,
                                       eventlabel, sharedp, params, log,
                                       longparamlist, time_units, paramtitles,
                                       chanrng)

                    # Save results
                    log.writelog('Saving results', mute=(not meta.verbose))
                    me.saveevent(meta, (meta.outputdir+'S5_'+meta.eventlabel +
                                        "_Meta_Save"), save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return meta


def fit_channel(meta, time, flux, chan, flux_err, eventlabel, sharedp, params,
                log, longparamlist, time_units, paramtitles, chanrng):
    """Run a fit for one channel or perform a shared fit.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    time : ndarray
        The time array.
    flux : ndarray
        The flux array.
    chan : int
        The current channel number.
    flux_err : ndarray
        The uncertainty on each data point.
    eventlabel : str
        The unique identifier for this analysis.
    sharedp : bool
        Whether or not this is a shared fit.
    params : eureka.lib.readEPF.Parameters
        The Parameters object containing the fitted parameters
        and their priors.
    log : logedit.Logedit
        The current log in which to output messages from this current stage.
    longparamlist : list
        The long list of all parameters relevant to this fit.
    time_units : str
        The units of the time array.
    paramtitles : list
        The names of the fitted parameters.
    chanrng : int
        The number of fitted channels.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    # Load the relevant values into the LightCurve model object
    lc_model = lc.LightCurve(time, flux, chan, chanrng, log, longparamlist,
                             unc=flux_err, time_units=time_units,
                             name=eventlabel, share=sharedp)

    if hasattr(meta, 'testing_model') and meta.testing_model:
        # FINDME: Use this area to add systematics into the data
        # when testing new systematics models. In this case, I'm
        # introducing an exponential ramp to test m.ExpRampModel().
        log.writelog('***Adding exponential ramp systematic to light curve***')
        fakeramp = m.ExpRampModel(parameters=params, name='ramp', fmt='r--',
                                  log=log,
                                  longparamlist=lc_model.longparamlist,
                                  nchan=lc_model.nchannel_fitted,
                                  paramtitles=paramtitles)
        fakeramp.coeffs = (np.array([-1, 40, -3, 0, 0, 0]).reshape(1, -1)
                           * np.ones(lc_model.nchannel_fitted))
        flux *= fakeramp.eval(time=time)
        lc_model.flux = flux

    # Make the astrophysical and detector models
    modellist = []
    if 'batman_tr' in meta.run_myfuncs:
        t_transit = m.BatmanTransitModel(parameters=params, name='transit',
                                         fmt='r--', log=log,
                                         longparamlist=lc_model.longparamlist,
                                         nchan=lc_model.nchannel_fitted,
                                         paramtitles=paramtitles)
        modellist.append(t_transit)
    if 'batman_ecl' in meta.run_myfuncs:
        t_eclipse = m.BatmanEclipseModel(parameters=params, name='eclipse',
                                         fmt='r--', log=log,
                                         longparamlist=lc_model.longparamlist,
                                         nchan=lc_model.nchannel_fitted,
                                         paramtitles=paramtitles)
        modellist.append(t_eclipse)
    if 'sinusoid_pc' in meta.run_myfuncs:
        model_names = np.array([model.name for model in modellist])
        t_model = None
        e_model = None
        # Nest any transit and/or eclipse models inside of the
        # phase curve model
        if 'transit' in model_names:
            t_model = modellist.pop(np.where(model_names == 'transit')[0][0])
            model_names = np.array([model.name for model in modellist])
        if'eclipse' in model_names:
            e_model = modellist.pop(np.where(model_names == 'eclipse')[0][0])
            model_names = np.array([model.name for model in modellist])
        t_phase = \
            m.SinusoidPhaseCurveModel(parameters=params, name='phasecurve',
                                      fmt='r--', log=log,
                                      longparamlist=lc_model.longparamlist,
                                      nchan=lc_model.nchannel_fitted,
                                      paramtitles=paramtitles,
                                      transit_model=t_model,
                                      eclipse_model=e_model)
        modellist.append(t_phase)
    if 'polynomial' in meta.run_myfuncs:
        t_polynom = m.PolynomialModel(parameters=params, name='polynom',
                                      fmt='r--', log=log,
                                      longparamlist=lc_model.longparamlist,
                                      nchan=lc_model.nchannel_fitted,
                                      paramtitles=paramtitles)
        modellist.append(t_polynom)
    if 'expramp' in meta.run_myfuncs:
        t_ramp = m.ExpRampModel(parameters=params, name='ramp', fmt='r--',
                                log=log,
                                longparamlist=lc_model.longparamlist,
                                nchan=lc_model.nchannel_fitted,
                                paramtitles=paramtitles)
        modellist.append(t_ramp)
    if 'GP' in meta.run_myfuncs:
        t_GP = m.GPModel(meta.kernel_class, meta.kernel_inputs, lc_model,
                         parameters=params, name='GP', fmt='r--', log=log,
                         gp_code=meta.GP_package,
                         longparamlist=lc_model.longparamlist,
                         nchan=lc_model.nchannel_fitted,
                         paramtitles=paramtitles)
        modellist.append(t_GP)
    model = m.CompositeModel(modellist, nchan=lc_model.nchannel_fitted)

    # Fit the models using one or more fitters
    log.writelog("=========================")
    if 'lsq' in meta.fit_method:
        log.writelog("Starting lsq fit.")
        model.fitter = 'lsq'
        lc_model.fit(model, meta, log, fitter='lsq')
        log.writelog("Completed lsq fit.")
        log.writelog("-------------------------")
    if 'emcee' in meta.fit_method:
        log.writelog("Starting emcee fit.")
        model.fitter = 'emcee'
        lc_model.fit(model, meta, log, fitter='emcee')
        log.writelog("Completed emcee fit.")
        log.writelog("-------------------------")
    if 'dynesty' in meta.fit_method:
        log.writelog("Starting dynesty fit.")
        model.fitter = 'dynesty'
        lc_model.fit(model, meta, log, fitter='dynesty')
        log.writelog("Completed dynesty fit.")
        log.writelog("-------------------------")
    if 'lmfit' in meta.fit_method:
        log.writelog("Starting lmfit fit.")
        model.fitter = 'lmfit'
        lc_model.fit(model, meta, log, fitter='lmfit')
        log.writelog("Completed lmfit fit.")
        log.writelog("-------------------------")
    log.writelog("=========================")

    # Plot the results from the fit(s)
    if meta.isplots_S5 >= 1:
        lc_model.plot(meta)

    return meta


def make_longparamlist(meta, params, chanrng):
    """Make a long list of all relevant parameters.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    params : eureka.lib.readEPF.Parameters
        The Parameters object containing the fitted parameters
        and their priors.
    chanrng : int
        The number of fitted channels.

    Returns
    -------
    longparamlist : list
        The long list of all parameters relevant to this fit.
    paramtitles : list
        The names of the fitted parameters.
    """
    if meta.sharedp:
        nspecchan = chanrng
    else:
        nspecchan = 1

    longparamlist = [[] for i in range(nspecchan)]
    tlist = list(params.dict.keys())
    for param in tlist:
        if 'free' in params.dict[param]:
            longparamlist[0].append(param)
            for c in np.arange(nspecchan-1):
                title = param+'_'+str(c+1)
                params.__setattr__(title, params.dict[param])
                longparamlist[c+1].append(title)
        elif 'shared' in params.dict[param]:
            for c in np.arange(nspecchan):
                longparamlist[c].append(param)
        else:
            for c in np.arange(nspecchan):
                longparamlist[c].append(param)
    paramtitles = longparamlist[0]

    return longparamlist, paramtitles


def load_specific_s4_meta_info(meta):
    """Load the specific S4 MetaClass object used to make this aperture pair.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.

    Returns
    -------
    eureka.lib.readECF.MetaClass
        The current metadata object with values from the old MetaClass.
    """
    inputdir = os.sep.join(meta.inputdir.split(os.sep)[:-2]) + os.sep
    # Get directory containing S4 outputs for this aperture pair
    inputdir += f'ap{meta.spec_hw}_bg{meta.bg_hw}'+os.sep
    # Locate the old MetaClass savefile, and load new ECF into
    # that old MetaClass
    meta.inputdir = inputdir
    s4_meta, meta.inputdir, meta.inputdir_raw = \
        me.findevent(meta, 'S4', allowFail=False)
    # Merge S5 meta into old S4 meta
    meta = me.mergeevents(meta, s4_meta)

    return meta
