import numpy as np
import os
import time as time_pkg
from glob import glob
from copy import deepcopy
import astraeus.xarrayIO as xrio

from .s5_meta import S5MetaClass
from . import lightcurve
from . import models as m
try:
    from . import differentiable_models as dm
except:
    # PyMC3 hasn't been installed
    dm = None
from ..lib import manageevent as me
from ..lib import util, logedit
from ..lib.readEPF import Parameters
from ..version import version


def fitlc(eventlabel, ecf_path=None, s4_meta=None, input_meta=None):
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
    input_meta : eureka.lib.readECF.MetaClass; optional
        An optional input metadata object, so you can manually edit the meta
        object without having to edit the ECF file.

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
    s4_meta = deepcopy(s4_meta)
    input_meta = deepcopy(input_meta)

    if input_meta is None:
        # Load Eureka! control file and store values in Event object
        ecffile = 'S5_' + eventlabel + '.ecf'
        meta = S5MetaClass(ecf_path, ecffile)
    else:
        meta = S5MetaClass(**input_meta.__dict__)

    meta.version = version
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

    meta = S5MetaClass(**me.mergeevents(meta, s4_meta).__dict__)
    meta.set_defaults()

    # Check to make sure that dm is accessible if using dm models/fitters
    if (dm is None and ('starry' in meta.fit_method or
                        'exoplanet' in meta.fit_method)):
        raise AssertionError(f"fit_method is set to {meta.fit_method}, but "
                             "could not import starry and/or pymc3 related "
                             "packages. Ensure that you have installed the "
                             "pymc3-related packages when installing Eureka!.")

    if meta.testing_S5:
        # Only fit a single channel while testing unless doing a shared fit,
        # then do two
        chanrng = 1
    elif meta.multwhite:
        chanrng = int(len(meta.inputdirlist) + 1)
    else:
        chanrng = meta.nspecchan

    # Create directories for Stage 5 outputs
    meta.run_s5 = None
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            if not isinstance(bg_hw_val, str):
                # Only divide if value is not a string (spectroscopic modes)
                bg_hw_val //= meta.expand
            meta.run_s5 = util.makedirectory(meta, 'S5', meta.run_s5,
                                             ap=spec_hw_val//meta.expand,
                                             bg=bg_hw_val)

    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            # Load in the S4 metadata used for this particular aperture pair
            meta = load_specific_s4_meta_info(meta)
            filename_S4_hold = meta.filename_S4_LCData.split(os.sep)[-1]
            lc = xrio.readXR(meta.inputdir+os.sep+filename_S4_hold)

            # Get the number of integrations in this lightcurve so
            # that we know how to split the flattened arrays
            meta.nints = [len(lc.time.values), ]

            if meta.multwhite:
                # Need to normalize each one if doing a joint fit
                lc_whites = [lc, ]

                for p in range(len(meta.inputdirlist)):
                    # Specify where glob should search for the save file
                    path = os.path.join(meta.topdir, meta.inputdirlist[p])
                    # Search
                    path = glob(path+os.sep+f'**{os.sep}*LCData.h5',
                                recursive=True)
                    if len(path) == 0:
                        raise AssertionError(
                            'Unable to find any LCData save files at '
                            f'{path}')
                    elif len(path) > 1:
                        print(f'WARNING: Found {len(path)} LCData save '
                              f'files... Using {path[0]}')
                    # Use the first file found
                    path = path[0]
                    lc_hold = xrio.readXR(path)
                    meta.wave_low = np.append(meta.wave_low,
                                              lc_hold.wave_low.values)
                    meta.wave_hi = np.append(meta.wave_hi,
                                             lc_hold.wave_hi.values)

                    # Get the number of integrations in this white lightcurve
                    # so that we know how to split the flattened arrays
                    meta.nints = np.append(meta.nints,
                                           len(lc_hold.time.values))

                    lc_whites.append(lc_hold)

            # Directory structure should not use expanded HW values
            spec_hw_val //= meta.expand
            if not isinstance(bg_hw_val, str):
                # Only divide if value is not a string (spectroscopic modes)
                bg_hw_val //= meta.expand
            # Get the directory for Stage 5 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S5', meta.run_s5,
                                                ap=spec_hw_val,
                                                bg=bg_hw_val)

            # Copy existing S4 log file and resume log
            meta.s5_logname = meta.outputdir + 'S5_' + meta.eventlabel + ".log"
            log = logedit.Logedit(meta.s5_logname, read=meta.s4_logname)
            log.writelog("\nStarting Stage 5: Light Curve Fitting\n")
            log.writelog(f"Eureka! Version: {meta.version}", mute=True)
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
            meta.sharedp = False
            meta.whitep = False
            for arg, val in params.dict.items():
                if 'shared' in val:
                    meta.sharedp = True
                if 'white_free' in val or 'white_fixed' in val:
                    meta.whitep = True

            if meta.sharedp and meta.testing_S5:
                chanrng = min([2, meta.nspecchan])

            if meta.manual_clip is not None:
                # Remove requested data points
                if meta.multwhite:
                    for p in range(len(meta.inputdirlist)+1):
                        meta, lc_whites[p], log = \
                            util.manual_clip(lc_whites[p], meta, log)
                        meta.nints[p] = len(lc_whites[p].time.values)
                else:
                    meta, lc, log = util.manual_clip(lc, meta, log)

            # Subtract off the user provided time value to avoid floating
            # point precision problems when fitting for values like t0
            offset = params.time_offset.value
            time = lc.time.values - offset
            if offset != 0:
                time_units = lc.data.attrs['time_units']+f' - {offset}'
            else:
                time_units = lc.data.attrs['time_units']
            # Record units for Stage 6
            meta.time_units = time_units
            meta.wave_units = lc.data.attrs['wave_units']

            # Collect the covariates for potential decorrelation
            centroid_param_list = []
            for centroid_param in ['centroid_x', 'centroid_sx',
                                   'centroid_y', 'centroid_sy']:
                if hasattr(lc, centroid_param):
                    centroid_param_list.append(
                        np.ma.masked_invalid(
                            getattr(lc, centroid_param).values))
                else:
                    centroid_param_list.append(
                        np.ma.zeros(lc.time.values.shape))
            xpos, xwidth, ypos, ywidth = centroid_param_list

            # make citations for current stage
            util.make_citations(meta, 5)

            # If any of the parameters' ptypes are set to 'white_free', enforce
            # a Gaussian prior based on a white-light light curve fit. If any
            # are 'white_fixed' freeze them to the white-light curve best fit
            if meta.whitep:
                if meta.use_generate_ld:
                    # Load limb-darkening coefficients made in Stage 4
                    ld_str = meta.use_generate_ld
                    if not hasattr(lc, ld_str + '_lin'):
                        raise Exception("Limb-darkening coefficients were "
                                        "not calculated in Stage 4.")
                    log.writelog("\nUsing limb-darkening coefficients "
                                 f"generated by {ld_str} \n")
                    ld_coeffs = [lc[ld_str + '_lin_white'].values,
                                 lc[ld_str + '_quad_white'].values,
                                 lc[ld_str + '_nonlin_3para_white'].values,
                                 lc[ld_str + '_nonlin_4para_white'].values]
                elif meta.ld_file:
                    # Load limb-darkening coefficients from a custom file
                    ld_fix_file = str(meta.ld_file_white)
                    try:
                        ld_coeffs = np.genfromtxt(ld_fix_file)
                    except FileNotFoundError:
                        raise Exception("The limb-darkening file "
                                        f"{ld_fix_file} could not be found.")
                    if len(ld_coeffs.shape) == 1:
                        ld_coeffs = ld_coeffs[np.newaxis, :]
                else:
                    ld_coeffs = None

                # Make a long list of parameters for each channel
                longparamlist, paramtitles, freenames = \
                    make_longparamlist(meta, params, 1)

                log.writelog("\nStarting Fit of White-light Light Curve\n")

                # Get the flux and error measurements for
                # the current channel
                mask = lc.mask_white.values
                flux = np.ma.masked_where(mask, lc.flux_white.values)
                flux_err = np.ma.masked_where(mask, lc.err_white.values)

                # Normalize flux and uncertainties to avoid large
                # flux values
                flux, flux_err = util.normalize_spectrum(
                    meta, flux, flux_err, scandir=getattr(lc, 'scandir', None))

                meta, params = fit_channel(meta, time, flux, 0, flux_err,
                                           eventlabel, params, log,
                                           longparamlist, time_units,
                                           paramtitles, freenames, 1,
                                           ld_coeffs, xpos, ypos,
                                           xwidth, ywidth, True)

                # Save results
                log.writelog('Saving results', mute=(not meta.verbose))
                me.saveevent(meta, meta.outputdir+'S5_'+meta.eventlabel +
                             "_white_Meta_Save", save=[])

            if meta.use_generate_ld:
                # Load limb-darkening coefficients made in Stage 4
                ld_str = meta.use_generate_ld
                if meta.multwhite:
                    nwhitechan = len(meta.inputdirlist) + 1
                    lin_c1 = np.zeros((nwhitechan, 1))
                    quad = np.zeros((nwhitechan, 2))
                    nonlin_3 = np.zeros((nwhitechan, 3))
                    nonlin_4 = np.zeros((nwhitechan, 4))
                    # Load LD coefficient from each lc
                    for p in range(nwhitechan):
                        if not hasattr(lc_whites[p], ld_str + '_lin'):
                            raise Exception("Exotic-ld coefficients have not" +
                                            " been calculated in Stage 4")
                        log.writelog("\nUsing generated limb-darkening " +
                                     f"coefficients with {ld_str} \n")
                        lin_c1[p] = lc_whites[p][ld_str + '_lin'].values[0]
                        quad[p] = lc_whites[p][ld_str + '_quad'].values[0]
                        nonlin_3[p] = lc_whites[p][ld_str + '_nonlin_3para'] \
                            .values[0]
                        nonlin_4[p] = lc_whites[p][ld_str + '_nonlin_4para'] \
                            .values[0]
                    ld_coeffs = [lin_c1, quad, nonlin_3, nonlin_4]
                else:
                    if not hasattr(lc, ld_str + '_lin'):
                        raise Exception("Exotic-ld coefficients have not" +
                                        " been calculated in Stage 4")
                    log.writelog("\nUsing generated limb-darkening " +
                                 f"coefficients with {ld_str} \n")
                    ld_coeffs = [lc[ld_str + '_lin'].values,
                                 lc[ld_str + '_quad'].values,
                                 lc[ld_str + '_nonlin_3para'].values,
                                 lc[ld_str + '_nonlin_4para'].values]
            elif meta.ld_file:
                # Load limb-darkening coefficients from a custom file
                ld_fix_file = str(meta.ld_file)
                try:
                    ld_coeffs = np.genfromtxt(ld_fix_file)
                except FileNotFoundError:
                    raise Exception("The limb-darkening file " + ld_fix_file +
                                    " could not be found.")
                if len(ld_coeffs.shape) == 1:
                    ld_coeffs = ld_coeffs[np.newaxis, :]
            else:
                ld_coeffs = None

            # Make a long list of parameters for each channel
            longparamlist, paramtitles, freenames = \
                make_longparamlist(meta, params, chanrng)

            # Joint White Light Fits (may have different time axis)
            if meta.multwhite:
                log.writelog("\nStarting Shared Fit of White Lights\n")

                flux = np.ma.masked_array([])
                flux_err = np.ma.masked_array([])
                time = np.ma.masked_array([])
                xpos = np.ma.masked_array([])
                ypos = np.ma.masked_array([])
                xwidth = np.ma.masked_array([])
                ywidth = np.ma.masked_array([])

                for pi in range(len(meta.inputdirlist)+1):
                    mask = lc_whites[pi].mask.values[0, :]
                    time_temp = lc_whites[pi].time.values - offset
                    time_temp = np.ma.masked_where(mask, time_temp)
                    flux_temp = np.ma.masked_where(
                        mask, lc_whites[pi].data.values[0, :])
                    err_temp = np.ma.masked_where(
                        mask, lc_whites[pi].err.values[0, :])
                    flux_temp, err_temp = util.normalize_spectrum(
                        meta, flux_temp, err_temp, mask)
                    flux = np.ma.append(flux, flux_temp)
                    flux_err = np.ma.append(flux_err, err_temp)
                    time = np.ma.append(time, time_temp)

                    if hasattr(lc_whites[pi], 'centroid_x'):
                        xpos_temp = np.ma.masked_invalid(
                            lc_whites[pi].centroid_x.values)
                        xpos_temp = np.ma.masked_where(mask, xpos_temp)
                    else:
                        xpos_temp = None
                    if hasattr(lc_whites[pi], 'centroid_x'):
                        xwidth_temp = np.ma.masked_invalid(
                            lc_whites[pi].centroid_sx.values)
                        xwidth_temp = np.ma.masked_where(mask, xwidth_temp)
                    else:
                        xwidth_temp = None
                    if hasattr(lc_whites[pi], 'centroid_y'):
                        ypos_temp = np.ma.masked_invalid(
                            lc_whites[pi].centroid_y.values)
                        ypos_temp = np.ma.masked_where(mask, ypos_temp)
                    else:
                        ypos_temp = None
                    if hasattr(lc_whites[pi], 'centroid_y'):
                        ywidth_temp = np.ma.masked_invalid(
                            lc_whites[pi].centroid_sy.values)
                        ywidth_temp = np.ma.masked_where(mask, ywidth_temp)
                    else:
                        ywidth_temp = None

                    xpos = np.ma.append(xpos, xpos_temp)
                    ypos = np.ma.append(ypos, ypos_temp)
                    xwidth = np.ma.append(xwidth, xwidth_temp)
                    ywidth = np.ma.append(ywidth, ywidth_temp)

                meta, params = fit_channel(meta, time, flux, 0, flux_err,
                                           eventlabel, params, log,
                                           longparamlist, time_units,
                                           paramtitles, freenames, chanrng,
                                           ld_coeffs, xpos, ypos,
                                           xwidth, ywidth)

                # Save results
                log.writelog('Saving results')
                me.saveevent(meta, (meta.outputdir+'S5_'+meta.eventlabel +
                                    "_Meta_Save"), save=[])

            # Now fit the multi-wavelength light curves
            elif meta.sharedp and not meta.multwhite:
                # Get the number of exposures in this lightcurve so
                # that we know how to split the flattened arrays
                size = len(lc.time.values)
                meta.nints = np.ones(chanrng, dtype=int)*size

                log.writelog(f"\nStarting Shared Fit of {chanrng} Channels\n")

                flux = np.ma.masked_array([])
                flux_err = np.ma.masked_array([])
                for channel in range(chanrng):
                    mask = lc.mask.values[channel, :]
                    flux_temp = np.ma.masked_where(mask,
                                                   lc.data.values[channel, :])
                    err_temp = np.ma.masked_where(mask,
                                                  lc.err.values[channel, :])
                    flux_temp, err_temp = util.normalize_spectrum(
                        meta, flux_temp, err_temp,
                        scandir=getattr(lc, 'scandir', None))
                    flux = np.ma.append(flux, flux_temp)
                    flux_err = np.ma.append(flux_err, err_temp)

                meta, params = fit_channel(meta, time, flux, 0, flux_err,
                                           eventlabel, params, log,
                                           longparamlist, time_units,
                                           paramtitles, freenames, chanrng,
                                           ld_coeffs, xpos, ypos,
                                           xwidth, ywidth)

                # Save results
                log.writelog('Saving results')
                me.saveevent(meta, (meta.outputdir+'S5_'+meta.eventlabel +
                                    "_Meta_Save"), save=[])
            else:
                for channel in range(chanrng):
                    log.writelog(f"\nStarting Channel {channel} of "
                                 f"{chanrng}\n")

                    # Get the flux and error measurements for
                    # the current channel
                    mask = lc.mask.values[channel, :]
                    flux = np.ma.masked_where(mask,
                                              lc.data.values[channel, :])
                    flux_err = np.ma.masked_where(mask,
                                                  lc.err.values[channel, :])
                    time_temp = np.ma.masked_where(mask, time)

                    # Normalize flux and uncertainties to avoid large
                    # flux values
                    flux, flux_err = util.normalize_spectrum(
                        meta, flux, flux_err,
                        scandir=getattr(lc, 'scandir', None))

                    meta, params = fit_channel(meta, time_temp, flux, channel,
                                               flux_err, eventlabel, params,
                                               log, longparamlist, time_units,
                                               paramtitles, freenames, chanrng,
                                               ld_coeffs, xpos, ypos,
                                               xwidth, ywidth)

                    # Save results
                    log.writelog('Saving results', mute=(not meta.verbose))
                    me.saveevent(meta, (meta.outputdir+'S5_'+meta.eventlabel +
                                        "_Meta_Save"), save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return meta


def fit_channel(meta, time, flux, chan, flux_err, eventlabel, params,
                log, longparamlist, time_units, paramtitles, freenames,
                chanrng, ldcoeffs, xpos, ypos, xwidth, ywidth, white=False):
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
        The generic names of the fitted parameters.
    freenames : list
        The specific names of all fitted parameters (e.g., including _ch#)
    chanrng : int
        The number of fitted channels.
    ldcoeffs : list
        Limb-darkening coefficients if used from Stage 4, otherwise None.
    white : bool; optional
        Is this a white-light fit? Defaults to False.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    if np.all(flux.mask):
        # All of the data for this channel are masked - skip it
        log.writelog("=========================")
        log.writelog(f"All data for channel {chan} are masked, so skipping "
                     "this channel.")
        log.writelog("=========================")
        return meta, params

    # Load the relevant values into the LightCurve model object
    lc_model = lightcurve.LightCurve(time, flux, chan, chanrng, log,
                                     longparamlist, params, freenames,
                                     unc=flux_err, time_units=time_units,
                                     name=eventlabel, share=meta.sharedp,
                                     white=white, multwhite=meta.multwhite,
                                     nints=meta.nints)

    nchannel_fitted = lc_model.nchannel_fitted
    fitted_channels = lc_model.fitted_channels

    if meta.testing_model:
        # FINDME: Use this area to add systematics into the data
        # when testing new systematics models. In this case, I'm
        # introducing an exponential ramp to test m.ExpRampModel().
        log.writelog('***Adding exponential ramp systematic to light curve***')
        fakeramp = m.ExpRampModel(parameters=params, name='ramp', fmt='r--',
                                  log=log, time=time,
                                  longparamlist=lc_model.longparamlist,
                                  nchannel=chanrng,
                                  nchannel_fitted=nchannel_fitted,
                                  fitted_channels=fitted_channels,
                                  paramtitles=paramtitles)
        fakeramp.coeffs = (np.array([-1, 40, -3, 0, 0, 0]).reshape(1, -1)
                           * np.ones(nchannel_fitted))
        flux *= fakeramp.eval(time=time)
        lc_model.flux = flux

    if 'starry' in meta.run_myfuncs:
        use_starry = True
        StarryModel = dm.StarryModel
        SinusoidModel = dm.SinusoidPhaseCurveModel
        QuasiLambertianPhaseCurve = dm.QuasiLambertianPhaseCurve
        PolynomialModel = dm.PolynomialModel
        StepModel = dm.StepModel
        ExpRampModel = dm.ExpRampModel
        HSTRampModel = dm.HSTRampModel
        CentroidModel = dm.CentroidModel
        GPModel = dm.GPModel
        AstroModel = dm.AstroModel
        CompositeModel = dm.CompositePyMC3Model
    else:
        use_starry = False
        BatmanTransitModel = m.BatmanTransitModel
        BatmanEclipseModel = m.BatmanEclipseModel
        PoetTransitModel = m.PoetTransitModel
        PoetEclipseModel = m.PoetEclipseModel
        PoetPCModel = m.PoetPCModel
        SinusoidModel = m.SinusoidPhaseCurveModel
        QuasiLambertianPhaseCurve = m.QuasiLambertianPhaseCurve
        DampedOscillatorModel = m.DampedOscillatorModel
        LorentzianModel = m.LorentzianModel
        PolynomialModel = m.PolynomialModel
        StepModel = m.StepModel
        ExpRampModel = m.ExpRampModel
        HSTRampModel = m.HSTRampModel
        CentroidModel = m.CentroidModel
        GPModel = m.GPModel
        AstroModel = m.AstroModel
        CompositeModel = m.CompositeModel

    freenames = []
    for key in params.dict:
        if params.dict[key][1] in ['free', 'shared', 'white_free',
                                   'white_fixed']:
            freenames.append(key)
    freenames = np.array(freenames)

    # Make the astrophysical and detector models
    modellist = []
    if use_starry:
        # Fixed any masked uncertainties
        masked = np.logical_or(np.ma.getmaskarray(flux),
                               np.ma.getmaskarray(flux_err))
        lc_model.unc[masked] = np.ma.median(lc_model.unc)
        lc_model.unc_fit[masked] = np.ma.median(lc_model.unc_fit)
        lc_model.unc.mask = False
        lc_model.unc_fit.mask = False

        t_starry = StarryModel(parameters=params,
                               fmt='r--', log=log,
                               time=time, time_units=time_units,
                               freenames=freenames,
                               longparamlist=lc_model.longparamlist,
                               nchannel=chanrng,
                               nchannel_fitted=nchannel_fitted,
                               fitted_channels=fitted_channels,
                               paramtitles=paramtitles,
                               ld_from_S4=meta.use_generate_ld,
                               ld_from_file=meta.ld_file,
                               ld_coeffs=ldcoeffs,
                               recenter_ld_prior=meta.recenter_ld_prior,
                               compute_ltt=meta.compute_ltt,
                               multwhite=lc_model.multwhite,
                               nints=lc_model.nints,
                               num_planets=meta.num_planets,
                               mutualOccultations=meta.mutualOccultations)
        modellist.append(t_starry)
        meta.ydeg = t_starry.ydeg
    if 'batman_tr' in meta.run_myfuncs:
        t_transit = BatmanTransitModel(parameters=params,
                                       fmt='r--', log=log, time=time,
                                       time_units=time_units,
                                       freenames=freenames,
                                       longparamlist=lc_model.longparamlist,
                                       nchannel=chanrng,
                                       nchannel_fitted=nchannel_fitted,
                                       fitted_channels=fitted_channels,
                                       paramtitles=paramtitles,
                                       ld_from_S4=meta.use_generate_ld,
                                       ld_from_file=meta.ld_file,
                                       ld_coeffs=ldcoeffs,
                                       recenter_ld_prior=meta.recenter_ld_prior,  # noqa: E501
                                       compute_ltt=meta.compute_ltt,
                                       multwhite=lc_model.multwhite,
                                       nints=lc_model.nints,
                                       num_planets=meta.num_planets)
        modellist.append(t_transit)
    if 'batman_ecl' in meta.run_myfuncs:
        t_eclipse = BatmanEclipseModel(parameters=params,
                                       fmt='r--', log=log, time=time,
                                       time_units=time_units,
                                       freenames=freenames,
                                       longparamlist=lc_model.longparamlist,
                                       nchannel=chanrng,
                                       nchannel_fitted=nchannel_fitted,
                                       fitted_channels=fitted_channels,
                                       paramtitles=paramtitles,
                                       compute_ltt=meta.compute_ltt,
                                       multwhite=lc_model.multwhite,
                                       nints=lc_model.nints,
                                       num_planets=meta.num_planets)
        modellist.append(t_eclipse)
    if 'poet_tr' in meta.run_myfuncs:
        t_poet_tr = PoetTransitModel(parameters=params,
                                     fmt='r--', log=log, time=time,
                                     time_units=time_units,
                                     freenames=freenames,
                                     longparamlist=lc_model.longparamlist,
                                     nchannel=chanrng,
                                     nchannel_fitted=nchannel_fitted,
                                     fitted_channels=fitted_channels,
                                     paramtitles=paramtitles,
                                     ld_from_S4=meta.use_generate_ld,
                                     ld_from_file=meta.ld_file,
                                     ld_coeffs=ldcoeffs,
                                     recenter_ld_prior=meta.recenter_ld_prior,
                                     compute_ltt=meta.compute_ltt,
                                     multwhite=lc_model.multwhite,
                                     nints=lc_model.nints,
                                     num_planets=meta.num_planets)
        modellist.append(t_poet_tr)
    if 'poet_ecl' in meta.run_myfuncs:
        t_poet_ecl = PoetEclipseModel(parameters=params,
                                      fmt='r--', log=log, time=time,
                                      time_units=time_units,
                                      freenames=freenames,
                                      longparamlist=lc_model.longparamlist,
                                      nchannel=chanrng,
                                      nchannel_fitted=nchannel_fitted,
                                      fitted_channels=fitted_channels,
                                      paramtitles=paramtitles,
                                      compute_ltt=meta.compute_ltt,
                                      multwhite=lc_model.multwhite,
                                      nints=lc_model.nints,
                                      num_planets=meta.num_planets)
        modellist.append(t_poet_ecl)
    if 'poet_pc' in meta.run_myfuncs:
        t_poet_pc = PoetPCModel(parameters=params,
                                fmt='r--', log=log, time=time,
                                time_units=time_units,
                                freenames=freenames,
                                longparamlist=lc_model.longparamlist,
                                nchannel=chanrng,
                                nchannel_fitted=nchannel_fitted,
                                fitted_channels=fitted_channels,
                                paramtitles=paramtitles,
                                force_positivity=meta.force_positivity,
                                multwhite=lc_model.multwhite,
                                nints=lc_model.nints,
                                num_planets=meta.num_planets)
        modellist.append(t_poet_pc)
    if 'sinusoid_pc' in meta.run_myfuncs and use_starry:
        t_phase = SinusoidModel(parameters=params,
                                fmt='r--', log=log, time=time,
                                time_units=time_units,
                                freenames=freenames,
                                longparamlist=lc_model.longparamlist,
                                nchannel=chanrng,
                                nchannel_fitted=nchannel_fitted,
                                fitted_channels=fitted_channels,
                                paramtitles=paramtitles,
                                multwhite=lc_model.multwhite,
                                nints=lc_model.nints,
                                num_planets=meta.num_planets)
        modellist.append(t_phase)
    elif 'sinusoid_pc' in meta.run_myfuncs:
        t_phase = SinusoidModel(parameters=params,
                                fmt='r--', log=log, time=time,
                                time_units=time_units,
                                freenames=freenames,
                                longparamlist=lc_model.longparamlist,
                                nchannel=chanrng,
                                nchannel_fitted=nchannel_fitted,
                                fitted_channels=fitted_channels,
                                paramtitles=paramtitles,
                                force_positivity=meta.force_positivity,
                                multwhite=lc_model.multwhite,
                                nints=lc_model.nints,
                                num_planets=meta.num_planets)
        modellist.append(t_phase)
    if 'quasilambert_pc' in meta.run_myfuncs:
        t_phase = \
            QuasiLambertianPhaseCurve(parameters=params,
                                      fmt='r--', log=log, time=time,
                                      time_units=time_units,
                                      freenames=freenames,
                                      longparamlist=lc_model.longparamlist,
                                      nchannel=chanrng,
                                      nchannel_fitted=nchannel_fitted,
                                      fitted_channels=fitted_channels,
                                      paramtitles=paramtitles,
                                      multwhite=lc_model.multwhite,
                                      nints=lc_model.nints,
                                      num_planets=meta.num_planets)
        modellist.append(t_phase)
    if 'damped_osc' in meta.run_myfuncs:
        t_osc = DampedOscillatorModel(parameters=params,
                                      fmt='r--', log=log, time=time,
                                      time_units=time_units,
                                      freenames=freenames,
                                      longparamlist=lc_model.longparamlist,
                                      nchannel=chanrng,
                                      nchannel_fitted=nchannel_fitted,
                                      fitted_channels=fitted_channels,
                                      paramtitles=paramtitles,
                                      multwhite=lc_model.multwhite,
                                      nints=lc_model.nints)
        modellist.append(t_osc)
    if 'lorentzian' in meta.run_myfuncs:
        t_lorentzian = LorentzianModel(parameters=params,
                                       fmt='r--', log=log, time=time,
                                       time_units=time_units,
                                       freenames=freenames,
                                       longparamlist=lc_model.longparamlist,
                                       nchannel=chanrng,
                                       nchannel_fitted=nchannel_fitted,
                                       fitted_channels=fitted_channels,
                                       paramtitles=paramtitles,
                                       multwhite=lc_model.multwhite,
                                       nints=lc_model.nints)
        modellist.append(t_lorentzian)
    if 'polynomial' in meta.run_myfuncs:
        t_polynom = PolynomialModel(parameters=params,
                                    fmt='r--', log=log, time=time,
                                    time_units=time_units,
                                    freenames=freenames,
                                    longparamlist=lc_model.longparamlist,
                                    nchannel=chanrng,
                                    nchannel_fitted=nchannel_fitted,
                                    fitted_channels=fitted_channels,
                                    paramtitles=paramtitles,
                                    multwhite=lc_model.multwhite,
                                    nints=lc_model.nints)
        modellist.append(t_polynom)
    if 'step' in meta.run_myfuncs:
        t_step = StepModel(parameters=params, fmt='r--',
                           log=log, time=time, time_units=time_units,
                           freenames=freenames,
                           longparamlist=lc_model.longparamlist,
                           nchannel=chanrng,
                           nchannel_fitted=nchannel_fitted,
                           fitted_channels=fitted_channels,
                           paramtitles=paramtitles,
                           multwhite=lc_model.multwhite,
                           nints=lc_model.nints)
        modellist.append(t_step)
    if 'expramp' in meta.run_myfuncs:
        t_expramp = ExpRampModel(parameters=params, fmt='r--',
                                 log=log, time=time, time_units=time_units,
                                 freenames=freenames,
                                 longparamlist=lc_model.longparamlist,
                                 nchannel=chanrng,
                                 nchannel_fitted=nchannel_fitted,
                                 fitted_channels=fitted_channels,
                                 paramtitles=paramtitles,
                                 multwhite=lc_model.multwhite,
                                 nints=lc_model.nints)
        modellist.append(t_expramp)
    if 'hstramp' in meta.run_myfuncs:
        t_hstramp = HSTRampModel(parameters=params, fmt='r--',
                                 log=log, time=time, time_units=time_units,
                                 freenames=freenames,
                                 longparamlist=lc_model.longparamlist,
                                 nchannel=chanrng,
                                 nchannel_fitted=nchannel_fitted,
                                 fitted_channels=fitted_channels,
                                 paramtitles=paramtitles,
                                 multwhite=lc_model.multwhite,
                                 nints=lc_model.nints)
        modellist.append(t_hstramp)
    if 'xpos' in meta.run_myfuncs:
        t_cent = CentroidModel(parameters=params, fmt='r--',
                               log=log, time=time, time_units=time_units,
                               freenames=freenames,
                               longparamlist=lc_model.longparamlist,
                               nchannel=chanrng,
                               nchannel_fitted=nchannel_fitted,
                               fitted_channels=fitted_channels,
                               paramtitles=paramtitles,
                               axis='xpos', centroid=xpos,
                               multwhite=lc_model.multwhite,
                               nints=lc_model.nints)
        modellist.append(t_cent)
    if 'xwidth' in meta.run_myfuncs:
        t_cent = CentroidModel(parameters=params, fmt='r--',
                               log=log, time=time, time_units=time_units,
                               freenames=freenames,
                               longparamlist=lc_model.longparamlist,
                               nchannel=chanrng,
                               nchannel_fitted=nchannel_fitted,
                               fitted_channels=fitted_channels,
                               paramtitles=paramtitles,
                               axis='xwidth', centroid=xwidth,
                               multwhite=lc_model.multwhite,
                               nints=lc_model.nints)
        modellist.append(t_cent)
    if 'ypos' in meta.run_myfuncs:
        t_cent = CentroidModel(parameters=params, fmt='r--',
                               log=log, time=time, time_units=time_units,
                               freenames=freenames,
                               longparamlist=lc_model.longparamlist,
                               nchannel=chanrng,
                               nchannel_fitted=nchannel_fitted,
                               fitted_channels=fitted_channels,
                               paramtitles=paramtitles,
                               axis='ypos', centroid=ypos,
                               multwhite=lc_model.multwhite,
                               nints=lc_model.nints)
        modellist.append(t_cent)
    if 'ywidth' in meta.run_myfuncs:
        t_cent = CentroidModel(parameters=params, fmt='r--',
                               log=log, time=time, time_units=time_units,
                               freenames=freenames,
                               longparamlist=lc_model.longparamlist,
                               nchannel=chanrng,
                               nchannel_fitted=nchannel_fitted,
                               fitted_channels=fitted_channels,
                               paramtitles=paramtitles,
                               axis='ywidth', centroid=ywidth,
                               multwhite=lc_model.multwhite,
                               nints=lc_model.nints)
        modellist.append(t_cent)
    if 'GP' in meta.run_myfuncs:
        t_GP = GPModel(meta.kernel_class, meta.kernel_inputs, lc_model,
                       parameters=params, fmt='r--', log=log,
                       time=time, time_units=time_units,
                       gp_code=meta.GP_package,
                       useHODLR=meta.useHODLR,
                       freenames=freenames,
                       longparamlist=lc_model.longparamlist,
                       nchannel=chanrng,
                       nchannel_fitted=nchannel_fitted,
                       fitted_channels=fitted_channels,
                       paramtitles=paramtitles,
                       multwhite=lc_model.multwhite,
                       nints=lc_model.nints)
        modellist.append(t_GP)

    # Combine all physical models into an AstroModel
    physical_models = [model for model in modellist
                       if model.modeltype == 'physical']
    modellist = [model for model in modellist
                 if model.modeltype != 'physical']
    astroModel = AstroModel(components=physical_models,
                            parameters=params,
                            fmt='r--', log=log, time=time,
                            time_units=time_units,
                            freenames=freenames,
                            longparamlist=lc_model.longparamlist,
                            nchannel=chanrng,
                            nchannel_fitted=nchannel_fitted,
                            fitted_channels=fitted_channels,
                            paramtitles=paramtitles,
                            multwhite=lc_model.multwhite,
                            nints=lc_model.nints,
                            num_planets=meta.num_planets)
    modellist.append(astroModel)

    # Combine all models into a composite model
    model = CompositeModel(modellist, parameters=params,
                           log=log, time=time, time_units=time_units,
                           freenames=freenames,
                           longparamlist=lc_model.longparamlist,
                           nchannel=chanrng,
                           nchannel_fitted=nchannel_fitted,
                           fitted_channels=fitted_channels,
                           paramtitles=paramtitles,
                           multwhite=lc_model.multwhite,
                           nints=lc_model.nints,
                           num_planets=meta.num_planets)

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
    if 'exoplanet' in meta.fit_method:
        log.writelog("Starting exoplanet fit.")
        model.fitter = 'exoplanet'
        lc_model.fit(model, meta, log, fitter='exoplanet')
        log.writelog("Completed exoplanet fit.")
        log.writelog("-------------------------")
    if 'nuts' in meta.fit_method:
        log.writelog("Starting PyMC3 NUTS fit.")
        model.fitter = 'nuts'
        lc_model.fit(model, meta, log, fitter='nuts')
        log.writelog("Completed PyMC3 NUTS fit.")
        log.writelog("-------------------------")
    log.writelog("=========================")

    # Plot the results from the fit(s)
    if meta.isplots_S5 >= 1:
        lc_model.plot(meta)

    if white:
        # Update the params to the values and uncertainties from
        # this white-light light curve fit
        best_model = None
        for model in lc_model.results:
            # Non-gradient based models: dynesty > emcee > lsq
            if model.fitter == 'dynesty':
                best_model = model
            elif (model.fitter == 'emcee' and
                  (best_model is None or best_model.fitter == 'lsq')):
                best_model = model
            elif model.fitter == 'lsq' and best_model is None:
                best_model = model
            # Gradient based models: nuts > exoplanet
            elif model.fitter == 'nuts':
                best_model = model
            elif model.fitter == 'exoplanet' and best_model is None:
                best_model = model
        if best_model is None:
            raise AssertionError('Unable to find fitter results')
        for key in params.params:
            ptype = getattr(params, key).ptype
            if 'white' in ptype:
                value = getattr(best_model.components[0].parameters,
                                key).value
                if best_model.fitter in ['lsq', 'exoplanet']:
                    ptype = 'fixed'
                    priorpar1 = None
                    priorpar2 = None
                    prior = None
                else:
                    ptype = ptype[6:]  # Remove 'white_'
                    priorpar1 = value
                    priorpar2 = best_model.errs[key]
                    prior = 'N'
                par = [value, ptype, priorpar1, priorpar2, prior]
                setattr(params, key, par)

    return meta, params


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
    if meta.multwhite:
        nspecchan = int(len(meta.inputdirlist)+1)
    elif meta.sharedp:
        nspecchan = chanrng
    else:
        nspecchan = 1

    longparamlist = [[] for i in range(nspecchan)]

    order = dict([[par, i] for i, par in enumerate(params.dict.keys())])
    paramtitles = sorted(np.unique([key.split('_ch')[0]
                                    for key in params.dict.keys()]),
                         key=order.get)

    for param in paramtitles:
        for c in range(nspecchan):
            name = param
            if c > 0:
                name += f'_ch{c}'
            longparamlist[c].append(name)
            if (name not in params.dict.keys() and
                    getattr(params, param).ptype not in ['shared',
                                                         'independent']):
                # Set this parameter based on channel 0
                params.__setattr__(name, params.dict[param])

    freenames = [key for key in params.dict.keys()
                 if getattr(params, key).ptype in
                 ['free', 'shared', 'white_fixed', 'white_free']]
    # Sort the list based on the order input by the user
    freenames_sorted = []
    for name in paramtitles:
        for c in range(nspecchan):
            key = ''
            if c > 0:
                key += f'_ch{c}'
            if name+key in freenames:
                freenames_sorted.append(name+key)

    return longparamlist, paramtitles, freenames_sorted


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
    if not isinstance(meta.bg_hw, str):
        # Only divide if value is not a string (spectroscopic modes)
        bg_hw = meta.bg_hw//meta.expand
    else:
        bg_hw = meta.bg_hw
    inputdir += f'ap{meta.spec_hw//meta.expand}_bg{bg_hw}'+os.sep
    # Locate the old MetaClass savefile, and load new ECF into
    # that old MetaClass
    meta.inputdir = inputdir
    s4_meta, meta.inputdir, meta.inputdir_raw = \
        me.findevent(meta, 'S4', allowFail=False)
    filename_S4_LCData = s4_meta.filename_S4_LCData
    # Merge S5 meta into old S4 meta
    meta = S5MetaClass(**me.mergeevents(meta, s4_meta).__dict__)

    # Make sure the filename_S4_LCData is kept
    meta.filename_S4_LCData = filename_S4_LCData

    return meta
