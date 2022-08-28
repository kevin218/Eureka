import numpy as np
import pandas as pd
from astropy import units, constants
import os
import time as time_pkg
from copy import copy
from glob import glob
import re
from ..lib import manageevent as me
from ..lib import readECF
from ..lib import util, logedit
from . import plots_s6 as plots
from ..lib import astropytable

import sys


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

    for meta.spec_hw_val in meta.spec_hw_range:
        for meta.bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            # Load in the S5 metadata used for this particular aperture pair
            meta = load_specific_s5_meta_info(meta)

            # Get the directory for Stage 6 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S6', meta.run_s6,
                                                ap=meta.spec_hw_val,
                                                bg=meta.bg_hw_val)

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
            meta.xlabel = physical_type+' ('+label_unit+')'

            fit_methods = meta.fit_method.strip('[').strip(']').strip()
            fit_methods = fit_methods.split(',')

            # Make sure these are lists even if it's just one item
            if (isinstance(meta.y_scalars, int) or
                    isinstance(meta.y_scalars, float)):
                meta.y_scalars = [meta.y_scalars]
            if isinstance(meta.y_params, str):
                meta.y_params = [meta.y_params]
            if not hasattr(meta, 'y_labels') or meta.y_labels is None:
                meta.y_labels = [None for _ in range(len(meta.y_params))]
            elif isinstance(meta.y_labels, str):
                meta.y_labels = [meta.y_labels]
            if (not hasattr(meta, 'y_label_units') or
                    meta.y_label_units is None):
                meta.y_label_units = [None for _ in range(len(meta.y_params))]
            elif isinstance(meta.y_label_units, str):
                meta.y_label_units = [meta.y_label_units]

            zipped_vals = zip(meta.y_params, meta.y_scalars, meta.y_labels,
                              meta.y_label_units)
            for vals in zipped_vals:
                (meta.y_param, meta.y_scalar,
                 meta.y_label, meta.y_label_unit) = vals

                log.writelog(f'Plotting {meta.y_param}...')

                # Read in S5 fitted values
                if meta.sharedp:
                    meta.spectrum_median, meta.spectrum_err = \
                        parse_s5_saves(meta, log, fit_methods, 'shared')
                else:
                    meta = parse_unshared_saves(meta, log, fit_methods)
                
                if all(x is None for x in meta.spectrum_median):
                    # The parameter could not be found - skip it
                    continue

                # Manipulate fitted values if needed
                if meta.y_param == 'rp^2':
                    meta = compute_transit_depth(meta)
                elif meta.y_param in ['1/r1', '1/r4']:
                    meta = compute_timescale(meta)

                if meta.y_label is None:
                    # Provide some default formatting
                    if meta.y_param == 'rp^2':
                        # Transit depth
                        meta.y_label = r'$(R_{\rm p}/R_{\rm *})^2$'
                    elif meta.y_param == 'rp':
                        # Radius ratio
                        meta.y_label = r'$R_{\rm p}/R_{\rm *}$'
                    elif meta.y_param == 'fp':
                        # Eclipse depth
                        meta.y_label = r'$F_{\rm p}/F_{\rm *}$'
                    elif meta.y_param in [f'u{i}' for i in range(1, 5)]:
                        # Limb darkening parameter
                        meta.y_label = r'$u_{\rm '+meta.y_param[-1]+'}$'
                        # Figure out which limb darkening law was used
                        epf_name = glob(meta.inputdir+'*.epf')[0]
                        with open(epf_name, 'r') as file:
                            for line in file.readlines():
                                if line.startswith('limb_dark'):
                                    limb_law = line.split()[1]
                        # Remove the quotation marks
                        limb_law = limb_law[1:-1]
                        if limb_law == 'kipping2013':
                            limb_law = 'Kipping (2013)'
                        meta.y_label += ' for '+limb_law
                    elif meta.y_param == 't0':
                        # Time of transit
                        meta.y_label = r'$t_{\rm 0}$'
                    elif meta.y_param == 'AmpSin1':
                        # Sine amplitude
                        meta.y_label = (r'Amplitude of $\sin(\phi)$')
                    elif meta.y_param == 'AmpSin2':
                        # Sine2 amplitude
                        meta.y_label = (r'Amplitude of $\sin(2\phi)$')
                    elif meta.y_param == 'AmpCos1':
                        # Cosine amplitude
                        meta.y_label = (r'Amplitude of $\cos(\phi)$')
                    elif meta.y_param == 'AmpCos2':
                        # Cosine2 amplitude
                        meta.y_label = (r'Amplitude of $\cos(2\phi)$')
                    elif meta.y_param in [f'c{i}' for i in range(0, 10)]:
                        # Polynomial in time coefficient
                        meta.y_label = r'$c_{\rm '+meta.y_param[1:]+'}$'
                    elif meta.y_param in [f'r{i}' for i in range(6)]:
                        # Exponential ramp parameters
                        meta.y_label = r'$r_{\rm '+meta.y_param[1:]+'}$'
                    elif meta.y_param in ['1/r1', '1/r4']:
                        # Exponential ramp timescales
                        meta.y_label = r'$1/r_{\rm '+meta.y_param[-1]+'}$'
                    else:
                        meta.y_label = meta.y_param

                # Convert to percent, ppm, etc. if requested
                if not hasattr(meta, 'y_scalar'):
                    meta.y_scalar = 1

                if meta.y_label_unit is None:
                    if meta.y_scalar == 1e6:
                        meta.y_label_unit = ' (ppm)'
                    elif meta.y_scalar == 100:
                        meta.y_label_unit = ' (%)'
                    elif meta.y_scalar != 1:
                        meta.y_label_unit = f' * {meta.y_scalar}'
                    else:
                        meta.y_label_unit = ''
                elif meta.y_label_unit[0] != ' ':
                    # Make sure there's a leading space for proper formatting
                    meta.y_label_unit = ' '+meta.y_label_unit

                # Add any units to the y label
                meta.y_label += meta.y_label_unit

                if meta.model_spectrum is not None:
                    model_x, model_y = load_model(meta, log, x_unit)
                else:
                    model_x = None
                    model_y = None

                # Make the spectrum plot
                if meta.isplots_S6 >= 1:
                    plots.plot_spectrum(meta, model_x, model_y, meta.y_scalar,
                                        meta.y_label, meta.xlabel)

                # Should we also make the scale_height version of the figure?
                has_requirements = np.all([hasattr(meta, val) for val in
                                           ['planet_Teq', 'planet_mu',
                                            'planet_Rad', 'planet_Mass',
                                            'star_Rad', 'planet_R0']])
                make_fig6301 = (meta.isplots_S6 >= 3 and has_requirements and
                                meta.y_param in ['rp', 'rp^2'])
                if make_fig6301:
                    # Make spectrum plot with scale height on the 2nd y-axis
                    scale_height = compute_scale_height(meta, log)
                    plots.plot_spectrum(meta, model_x, model_y, meta.y_scalar,
                                        meta.y_label, meta.xlabel,
                                        scale_height, meta.planet_R0)

                log.writelog('Saving results as astropy table')
                save_table(meta)

            # Store citations to relevant dependencies in the meta file
            # pass in list of currently imported modules to search for 
            # citations
            mods = np.unique([mod.split('.')[0] for mod in sys.modules.keys()])
            util.make_citations(meta, mods)

            # Save results
            log.writelog('Saving results')
            fname = meta.outputdir+'S6_'+meta.eventlabel+"_Meta_Save"
            me.saveevent(meta, fname, save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return meta


def parse_s5_saves(meta, log, fit_methods, channel_key='shared'):
    """Load in an S5 save file.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    fit_methods : list
        The fitting methods used in S5.
    channel_key : str; optional
        A string describing the current channel (e.g. ch0),
        by default 'shared'.

    Returns
    -------
    medians
        The best-fit values from S5.
    errs
        The uncertainties from a sampling algorithm like dynesty or emcee.
    """
    if meta.y_param == 'rp^2':
        y_param = 'rp'
    elif meta.y_param in ['1/r1', '1/r4']:
        y_param = meta.y_param[2:]
    else:
        y_param = meta.y_param

    for fitter in fit_methods:
        if fitter in ['dynesty', 'emcee']:
            fname = f'S5_{fitter}_fitparams_{channel_key}.csv'
            fitted_values = pd.read_csv(meta.inputdir+fname, escapechar='#',
                                        skipinitialspace=True)
            full_keys = list(fitted_values["Parameter"])

            fname = f'S5_{fitter}_samples_{channel_key}'

            keys = [key for key in full_keys if y_param in key]
            if len(keys) == 0:
                log.writelog(f'Parameter {y_param} was not in the list of '
                             'fitted parameters which includes:\n['
                             + ', '.join(full_keys)+']')
                log.writelog(f'Skipping {y_param}')
                return None, None

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
            keys = [key for key in full_keys if y_param in key]
            if len(keys) == 0:
                log.writelog(f'Parameter {y_param} was not in the list of '
                             'fitted parameters which includes:\n['
                             + ', '.join(full_keys)+']')
                log.writelog(f'Skipping {y_param}')
                return None, None

            medians = []
            for i, key in enumerate(keys):
                ind = np.where(fitted_values["Parameter"] == key)[0][0]
                if "50th" in fitted_values.keys():
                    medians.append(fitted_values["50th"][ind])
                else:
                    medians.append(fitted_values["Mean"][ind])
            medians = np.array(medians)

            # if lsq, no uncertainties
            errs = np.ones((2, len(medians)))*np.nan

    return medians, errs


def parse_unshared_saves(meta, log, fit_methods):
    """Load in the many S5 save files for an unshared fit.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    fit_methods : list
        The fitting methods used in S5.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated meta data object.
    """
    meta.spectrum_median = []
    meta.spectrum_err = []
    for channel in range(meta.nspecchan):
        ch_number = str(channel).zfill(len(str(meta.nspecchan)))
        channel_key = f'ch{ch_number}'
        median, err = parse_s5_saves(meta, log, fit_methods, channel_key)
        if median is None:
            # Parameter was found, so don't keep looking for it
            meta.spectrum_median = [None for _ in range(meta.nspecchan)]
            meta.spectrum_err = [None for _ in range(meta.nspecchan)]
            return meta
        meta.spectrum_median.extend(median)
        meta.spectrum_err.extend(err.T)
    
    meta.spectrum_median = np.array(meta.spectrum_median)
    meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_transit_depth(meta):
    """Convert the fitted rp values to transit depth.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated meta data object.
    """
    if not np.all(np.isnan(meta.spectrum_err)):
        lower = np.abs((meta.spectrum_median-meta.spectrum_err[0, :])**2 -
                       meta.spectrum_median**2)
        upper = np.abs((meta.spectrum_median+meta.spectrum_err[1, :])**2 -
                       meta.spectrum_median**2)
        meta.spectrum_err = np.append(lower.reshape(1, -1),
                                      upper.reshape(1, -1), axis=0)
    meta.spectrum_median *= meta.spectrum_median

    return meta


def compute_timescale(meta):
    """Convert the fitted r1 or r4 value to a timescale.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated meta data object.
    """
    median = meta.spectrum_median
    if not np.all(np.isnan(meta.spectrum_err)):
        lower = meta.spectrum_err[0, :]
        upper = meta.spectrum_err[1, :]
        lower = np.abs(1/(median-lower) - 1/median)
        upper = np.abs(1/(median+upper) - 1/median)
        meta.spectrum_err = np.append(lower.reshape(1, -1),
                                      upper.reshape(1, -1), axis=0)
    meta.spectrum_median = 1/median

    return meta


def compute_scale_height(meta, log):
    """Compute the atmospheric scale height for a planet.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated meta data object.
    """
    if meta.planet_Rad is None:
        meta.planet_Rad = meta.spectrum_median
        if meta.y_param == 'rp^2':
            meta.planet_Rad = np.sqrt(meta.planet_Rad)
        meta.planet_Rad = np.mean(meta.planet_Rad)
        meta.planet_Rad *= (meta.star_Rad*constants.R_sun /
                            constants.R_jup).si.value
    if meta.planet_R0 is not None:
        meta.planet_R0 *= (constants.R_jup/(meta.star_Rad *
                                            constants.R_sun)).si.value
    meta.planet_g = ((constants.G*meta.planet_Mass*constants.M_jup) /
                     (meta.planet_Rad*constants.R_jup)**2).si.value
    log.writelog(f'Calculated g={np.round(meta.planet_g,2)} m/s^2 '
                 f'with Rp={np.round(meta.planet_Rad, 2)} R_jup '
                 f'and Mp={meta.planet_Mass} M_jup')
    scale_height = (constants.k_B*(meta.planet_Teq*units.K) /
                    ((meta.planet_mu*units.u) *
                     (meta.planet_g*units.m/units.s**2)))
    scale_height = scale_height.si.to('km')
    log.writelog(f'Calculated H={np.round(scale_height,2)} with '
                 f'g={np.round(meta.planet_g, 2)} m/s^2, '
                 f'Teq={meta.planet_Teq} K, and '
                 f'mu={meta.planet_mu} u')
    scale_height /= meta.star_Rad*constants.R_sun
    scale_height = scale_height.si.value

    return scale_height


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


def load_model(meta, log, x_unit):
    """Load in a model to plot above/below the fitted data.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    x_unit : str
        The astropy.units unit that will be used in the plot.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated meta data object.

    Raises
    ------
    AssertionError
        Unknown conversion between y_param and model_y_param.
    """
    if meta.model_y_param != meta.y_param:
        # Need to make sure this model is relevant for this current plot
        model_y_param = copy(meta.model_y_param)
        y_param = copy(meta.y_param)
        # Strip off any unimportant characters
        for key in ['(', ')', '^2', '{', '}', '[', ']']:
            model_y_param = model_y_param.replace(key, '')
            y_param = y_param.replace(key, '')
        if model_y_param != y_param:
            # This model is not relevant for this plot, so skipping it
            log.writelog(f'The model_y_param ({meta.model_y_param}) does not '
                         f'match the current y_param ({meta.y_param}), so not'
                         'using the model for this plot')
            return meta, None, None

    model_path = os.path.join(meta.topdir, *meta.model_spectrum.split(os.sep))
    model_x, model_y = np.loadtxt(model_path, delimiter=meta.model_delimiter).T
    # Convert model_x_unit to x_unit if needed
    model_x_unit = getattr(units, meta.model_x_unit)
    model_x_unit = model_x_unit.to(x_unit, equivalencies=units.spectral())
    model_x *= model_x_unit
    # Figure out if model needs to be converted to Rp/Rs
    sqrt_model = (meta.param == 'rp^2' and meta.model_y_param != meta.y_param)
    # Figure out if model needs to be converted to (Rp/Rs)^2
    sq_model = (meta.param == 'rp' and meta.model_y_param != meta.y_param)
    if sqrt_model:
        model_y = np.sqrt(model_y)
    elif sq_model:
        model_y *= model_y
    elif meta.model_y_param != meta.y_param:
        raise AssertionError('Unknown conversion between y_param '
                             f'{meta.y_param} and model_y_param '
                             f'{meta.model_y_param}')

    if not hasattr(meta, 'model_y_scalar'):
        meta.model_y_scalar = 1

    # Convert the model y-units if needed to match the data
    # y-units requested
    if meta.model_y_scalar != 1:
        model_y *= meta.model_y_scalar

    return model_x, model_y


def save_table(meta):
    """Clean y_param for filenames and save the table of values.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    """
    event_ap_bg = (meta.eventlabel+"_ap"+str(meta.spec_hw_val)+'_bg' +
                   str(meta.bg_hw_val))
    clean_y_param = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", meta.y_param)
    meta.tab_filename_s6 = (meta.outputdir+'S6_'+event_ap_bg+'_' +
                            clean_y_param+"_Table_Save.txt")
    wavelengths = np.mean(np.append(meta.wave_low.reshape(1, -1),
                                    meta.wave_hi.reshape(1, -1),
                                    axis=0), axis=0)
    wave_errs = (meta.wave_hi-meta.wave_low)/2
    astropytable.savetable_S6(meta.tab_filename_s6, meta.y_param, wavelengths,
                              wave_errs, meta.spectrum_median,
                              meta.spectrum_err)
    return
