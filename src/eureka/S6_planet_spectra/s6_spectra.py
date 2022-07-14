import numpy as np
import pandas as pd
from astropy import units, constants
import os
import time as time_pkg
from ..lib import manageevent as me
from ..lib import readECF
from ..lib import util, logedit
from . import plots_s6 as plots
from ..lib import astropytable


def plot_spectra(eventlabel, ecf_path=None, s5_meta=None):
    '''Gathers together different wavelength fits and makes
    transmission/emission spectra.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored.
        Defaults to None which resolves to './'.
    s5_meta : eureka.lib.readECF.MetaClass; optional
        The metadata object from Eureka!'s S5 step (if running S5
        and S6 sequentially). Defaults to None.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object with attributes added by S6.

    Notes
    -----
    History:

    - Feb 14, 2022 Taylor Bell
        Original version
    '''
    print("\nStarting Stage 6: Light Curve Fitting\n")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S6_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel
    meta.datetime = time_pkg.strftime('%Y-%m-%d')

    if s5_meta is None:
        # Locate the old MetaClass savefile, and load new ECF into
        # that old MetaClass
        s5_meta, meta.inputdir, meta.inputdir_raw = \
            me.findevent(meta, 'S5', allowFail=False)
    else:
        # Running these stages sequentially, so can safely assume
        # the path hasn't changed
        meta.inputdir = s5_meta.outputdir
        meta.inputdir_raw = meta.inputdir[len(meta.topdir):]

    meta = me.mergeevents(meta, s5_meta)

    if not meta.allapers:
        # The user indicated in the ecf that they only want to consider one
        # aperture in which case the code will consider only the one which
        # made s5_meta. Alternatively, if S4 or S5 was run without allapers,
        # S6 will already only consider that one
        meta.spec_hw_range = [meta.spec_hw, ]
        meta.bg_hw_range = [meta.bg_hw, ]

    # Create directories for Stage 6 outputs
    meta.run_s6 = None
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            meta.run_s6 = util.makedirectory(meta, 'S6', meta.run_s6,
                                             ap=spec_hw_val, bg=bg_hw_val)

    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta.spec_hw = spec_hw_val
            meta.bg_hw = bg_hw_val

            # Load in the S5 metadata used for this particular aperture pair
            meta = load_specific_s5_meta_info(meta)

            # Get the directory for Stage 6 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S6', meta.run_s6,
                                                ap=spec_hw_val, bg=bg_hw_val)

            # Copy existing S5 log file and resume log
            meta.s6_logname = meta.outputdir+'S6_'+meta.eventlabel+'.log'
            log = logedit.Logedit(meta.s6_logname, read=meta.s5_logname)
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ecf
            log.writelog('Copying S6 control file')
            meta.copy_ecf()

            # Get the wavelength values
            meta.wave_low = np.array(meta.wave_low)
            meta.wave_hi = np.array(meta.wave_hi)
            meta.wavelengths = np.mean(np.append(meta.wave_low.reshape(1, -1),
                                       meta.wave_hi.reshape(1, -1), axis=0),
                                       axis=0)
            meta.wave_errs = (meta.wave_hi-meta.wave_low)/2

            # Convert to the user-provided x-axis unit if needed
            if hasattr(meta, 'x_unit'):
                x_unit = getattr(units, meta.x_unit)
            else:
                log.writelog('Assuming a wavelength unit of microns')
                x_unit = units.um
            # FINDME: For now this is assuming that the data is in units of
            # microns We should add something to S3 that notes what the units
            # of the wavelength were in the FITS file
            meta.wavelengths *= units.um.to(x_unit,
                                            equivalencies=units.spectral())
            meta.wave_errs *= units.um.to(x_unit,
                                          equivalencies=units.spectral())
            physical_type = str(x_unit.physical_type).title()
            if physical_type == 'Length':
                physical_type = 'Wavelength'
            label_unit = x_unit.name
            if label_unit == 'um':
                label_unit = r'$\mu$m'
            xlabel = physical_type+' ('+label_unit+')'

            fit_methods = meta.fit_method.strip('[').strip(']').strip()
            fit_methods = fit_methods.split(',')

            accepted_y_units = ['Rp/Rs', 'Rp/R*', '(Rp/Rs)^2', '(Rp/R*)^2',
                                'Fp/Fs', 'Fp/F*']
            if 'rp' in meta.y_unit.lower():
                y_param = 'rp'
            elif 'fp' in meta.y_unit.lower():
                y_param = 'fp'
            else:
                raise AssertionError(f'Unknown y_unit {meta.y_unit} is none of'
                                     ' ['+', '.join(accepted_y_units)+']')

            # Read in S5 fitted values
            if meta.sharedp:
                meta.spectrum_median, meta.spectrum_err = \
                    parse_s5_saves(meta, fit_methods, y_param, 'shared')
            else:
                meta.spectrum_median = np.zeros(0)
                meta.spectrum_err = np.zeros((2, 0))
                for channel in range(meta.nspecchan):
                    ch_number = str(channel).zfill(len(str(meta.nspecchan)))
                    channel_key = f'ch{ch_number}'
                    median, err = parse_s5_saves(meta, fit_methods, y_param,
                                                 channel_key)
                    meta.spectrum_median = np.append(meta.spectrum_median,
                                                     median, axis=0)
                    meta.spectrum_err = np.append(meta.spectrum_err, err,
                                                  axis=1)

            # Convert the y-axis unit to the user-provided value if needed
            if meta.y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2']:
                if not np.all(np.isnan(meta.spectrum_err)):
                    lower = np.abs((meta.spectrum_median -
                                    meta.spectrum_err[0, :])**2 -
                                   meta.spectrum_median**2)
                    upper = np.abs((meta.spectrum_median +
                                    meta.spectrum_err[1, :])**2 -
                                   meta.spectrum_median**2)
                    meta.spectrum_err = np.append(lower.reshape(1, -1),
                                                  upper.reshape(1, -1), axis=0)
                meta.spectrum_median *= meta.spectrum_median
                ylabel = r'$(R_{\rm p}/R_{\rm *})^2$'
            elif meta.y_unit in ['Rp/Rs', 'Rp/R*']:
                ylabel = r'$R_{\rm p}/R_{\rm *}$'
            elif meta.y_unit in ['Fp/Fs', 'Fp/F*']:
                ylabel = r'$F_{\rm p}/F_{\rm *}$'
            else:
                raise AssertionError(f'Unknown y_unit {meta.y_unit} is none of'
                                     ' ['+', '.join(accepted_y_units)+']')

            # Convert to percent, ppm, etc. if requested
            if not hasattr(meta, 'y_scalar'):
                meta.y_scalar = 1

            if meta.y_scalar == 1e6:
                ylabel += ' (ppm)'
            elif meta.y_scalar == 100:
                ylabel += ' (%)'
            elif meta.y_scalar != 1:
                ylabel += f' * {meta.y_scalar}'

            if meta.model_spectrum is not None:
                model_path = os.path.join(meta.topdir,
                                          *meta.model_spectrum.split(os.sep))
                model_x, model_y = np.loadtxt(model_path,
                                              delimiter=meta.model_delimiter).T
                # Convert model_x_unit to x_unit if needed
                model_x_unit = getattr(units, meta.model_x_unit)
                model_x_unit = model_x_unit.to(x_unit,
                                               equivalencies=units.spectral())
                model_x *= model_x_unit
                # Figure out if model needs to be converted to Rp/Rs
                sqrt_model = (meta.model_y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2']
                              and meta.model_y_unit != meta.y_unit)
                # Figure out if model needs to be converted to (Rp/Rs)^2
                sq_model = (meta.model_y_unit in ['Rp/Rs', 'Rp/R*']
                            and meta.model_y_unit != meta.y_unit)
                if sqrt_model:
                    model_y = np.sqrt(model_y)
                elif sq_model:
                    model_y *= model_y
                elif meta.model_y_unit not in accepted_y_units:
                    raise AssertionError('Unknown model_y_unit '
                                         f'{meta.model_y_unit} is none of ['
                                         ', '.join(accepted_y_units)+']')
                elif meta.model_y_unit != meta.y_unit:
                    raise AssertionError('Unknown conversion between y_unit '
                                         f'{meta.y_unit} and model_y_unit '
                                         f'{meta.model_y_unit}')

                if not hasattr(meta, 'model_y_scalar'):
                    meta.model_y_scalar = 1

                # Convert the model y-units if needed to match the data
                # y-units requested
                if meta.model_y_scalar != 1:
                    model_y *= meta.model_y_scalar
            else:
                model_x = None
                model_y = None

            # Make the spectrum plot
            if meta.isplots_S6 >= 1:
                plots.plot_spectrum(meta, model_x, model_y, meta.y_scalar,
                                    ylabel, xlabel)

            # Should we also make the scaleHeight version of the figure?
            has_requirements = np.all([hasattr(meta, val) for val in
                                       ['planet_Teq', 'planet_mu',
                                        'planet_Rad', 'planet_Mass',
                                        'star_Rad', 'planet_R0']])
            make_fig6301 = (meta.isplots_S6 >= 3 and y_param == 'rp'
                            and has_requirements)
            if make_fig6301:
                # Make the spectrum plot
                if meta.planet_Rad is None:
                    meta.planet_Rad = meta.spectrum_median
                    if meta.y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2']:
                        meta.planet_Rad = np.sqrt(meta.planet_Rad)
                    meta.planet_Rad = np.mean(meta.planet_Rad)
                    meta.planet_Rad *= (meta.star_Rad*constants.R_sun /
                                        constants.R_jup).si.value
                if meta.planet_R0 is not None:
                    meta.planet_R0 *= (constants.R_jup/(meta.star_Rad *
                                                        constants.R_sun)
                                       ).si.value
                meta.planet_g = ((constants.G * meta.planet_Mass *
                                  constants.M_jup) /
                                 (meta.planet_Rad*constants.R_jup)**2).si.value
                log.writelog(f'Calculated g={np.round(meta.planet_g,2)} m/s^2 '
                             f'with Rp={np.round(meta.planet_Rad, 2)} R_jup '
                             f'and Mp={meta.planet_Mass} M_jup')
                scaleHeight = (constants.k_B*(meta.planet_Teq*units.K) /
                               ((meta.planet_mu*units.u) *
                                (meta.planet_g*units.m/units.s**2)))
                scaleHeight = scaleHeight.si.to('km')
                log.writelog(f'Calculated H={np.round(scaleHeight,2)} with '
                             f'g={np.round(meta.planet_g, 2)} m/s^2, '
                             f'Teq={meta.planet_Teq} K, and '
                             f'mu={meta.planet_mu} u')
                scaleHeight /= meta.star_Rad*constants.R_sun
                scaleHeight = scaleHeight.si.value
                plots.plot_spectrum(meta, model_x, model_y, meta.y_scalar,
                                    ylabel, xlabel, scaleHeight,
                                    meta.planet_R0)

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.writelog('Saving results as astropy table')
            event_ap_bg = (meta.eventlabel+"_ap"+str(spec_hw_val)+'_bg'
                           + str(bg_hw_val))
            meta.tab_filename_s6 = (meta.outputdir+'S6_'+event_ap_bg
                                    + "_Table_Save.txt")
            wavelengths = np.mean(np.append(meta.wave_low.reshape(1, -1),
                                            meta.wave_hi.reshape(1, -1),
                                            axis=0), axis=0)
            wave_errs = (meta.wave_hi-meta.wave_low)/2
            if meta.y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2']:
                tr_depth = meta.spectrum_median
                tr_depth_err = meta.spectrum_err
            elif meta.y_unit in ['Rp/Rs', 'Rp/R*']:
                tr_depth = meta.spectrum_median**2
                tr_depth_err = meta.spectrum_err**2
            else:
                tr_depth = np.ones_like(meta.spectrum_median)*np.nan
                tr_depth_err = np.ones_like(meta.spectrum_err)*np.nan
            if meta.y_unit in ['Fp/Fs', 'Fp/F*']:
                ecl_depth = meta.spectrum_median
                ecl_depth_err = meta.spectrum_err
            else:
                ecl_depth = np.ones_like(meta.spectrum_median)*np.nan
                ecl_depth_err = np.ones_like(meta.spectrum_err)*np.nan
            astropytable.savetable_S6(meta.tab_filename_s6, wavelengths,
                                      wave_errs, tr_depth, tr_depth_err,
                                      ecl_depth, ecl_depth_err)

            # Save results
            log.writelog('Saving results')
            fname = meta.outputdir+'S6_'+meta.eventlabel+"_Meta_Save"
            me.saveevent(meta, fname, save=[])

            log.closelog()

    return meta


def parse_s5_saves(meta, fit_methods, y_param, channel_key='shared'):
    """Load in the S5 save file.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    fit_methods : list
        The fitting methods used in S5.
    y_param : str
        The parameter to plot.
    channel_key : str; optional
        A string describing the current channel (e.g. ch0),
        by default 'shared'.

    Returns
    -------
    medians
        The best-fit values from S5.
    errs
        The uncertainties from a sampling algorithm like dynesty or emcee.

    Raises
    ------
    AssertionError
        The y_param was not found in the list of fitted parameter names.
    """
    for fitter in fit_methods:
        if fitter in ['dynesty', 'emcee']:
            fname = f'S5_{fitter}_fitparams_{channel_key}.csv'
            fitted_values = pd.read_csv(meta.inputdir+fname, escapechar='#',
                                        skipinitialspace=True)
            full_keys = list(fitted_values["Parameter"])

            fname = f'S5_{fitter}_samples_{channel_key}'

            if y_param == 'fp':
                keys = [key for key in full_keys if 'fp' in key]
            else:
                keys = [key for key in full_keys if 'rp' in key]

            if len(keys) == 0:
                raise AssertionError(f'Parameter {y_param} was not in the list'
                                     ' of fitted parameters which includes: '
                                     ', '.join(full_keys))

            lowers = []
            uppers = []
            medians = []

            for i, key in enumerate(keys):
                ind = np.where(fitted_values["Parameter"] == key)[0][0]
                lowers.append(np.abs(fitted_values["-1sigma"][ind]))
                uppers.append(np.abs(fitted_values["+1sigma"][ind]))
                medians.append(np.abs(fitted_values["50th"][ind]))

            errs = np.array([lowers, uppers])
            medians = np.array(medians)

        else:
            fname = f'S5_{fitter}_fitparams_{channel_key}.csv'
            fitted_values = pd.read_csv(meta.inputdir+fname, escapechar='#',
                                        skipinitialspace=True)
            full_keys = list(fitted_values["Parameter"])
            if y_param == 'fp':
                keys = [key for key in full_keys if 'fp' in key]
            else:
                keys = [key for key in full_keys if 'rp' in key]
            if len(keys) == 0:
                raise AssertionError(f'Parameter {y_param} was not in the list'
                                     ' of fitted parameters which includes: '
                                     ', '.join(full_keys))
            
            medians = []
            for i, key in enumerate(keys):
                ind = np.where(fitted_values["Parameter"] == key)[0][0]
                if "50th" in fitted_values.keys():
                    medians.append(fitted_values["50th"][ind])
                else:
                    medians.append(fitted_values["Mean"][ind])
            medians = np.array(medians)

            # if lsq (no uncertainties)
            errs = np.ones((2, len(medians)))*np.nan

    return medians, errs


def load_specific_s5_meta_info(meta):
    """Load in the MetaClass object from the particular aperture pair being
    used.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object with values from earlier stages.
    """
    inputdir = os.sep.join(meta.inputdir.split(os.sep)[:-2]) + os.sep
    # Get directory containing S5 outputs for this aperture pair
    inputdir += f'ap{meta.spec_hw}_bg{meta.bg_hw}'+os.sep
    # Locate the old MetaClass savefile, and load new ECF into
    # that old MetaClass
    meta.inputdir = inputdir
    s5_meta, meta.inputdir, meta.inputdir_raw = \
        me.findevent(meta, 'S5', allowFail=False)
    # Merge S6 meta into old S5 meta
    meta = me.mergeevents(meta, s5_meta)

    return meta
