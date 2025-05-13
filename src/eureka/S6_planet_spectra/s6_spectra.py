import numpy as np
from copy import deepcopy
import pandas as pd
from astropy import units, constants
import os
import time as time_pkg
from copy import copy
from glob import glob
from tqdm import tqdm
import re
from matplotlib.pyplot import rcParams
from astraeus import xarrayIO as xrio

try:
    import starry
except ModuleNotFoundError:
    # starry hasn't been installed
    pass
try:
    from harmonica import HarmonicaTransit
except ModuleNotFoundError:
    # Harmonica hasn't been installed
    pass

from .s6_meta import S6MetaClass
from . import plots_s6 as plots
from ..lib import manageevent as me
from ..lib import util, logedit, astropytable
from ..version import version


def plot_spectra(eventlabel, ecf_path=None, s5_meta=None, input_meta=None):
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
    input_meta : eureka.lib.readECF.MetaClass; optional
        An optional input metadata object, so you can manually edit the meta
        object without having to edit the ECF file.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object with attributes added by S6.
    lc : Astreaus object
        Data object of time-like arrays (light curve).
    '''
    s5_meta = deepcopy(s5_meta)
    input_meta = deepcopy(input_meta)

    if input_meta is None:
        # Load Eureka! control file and store values in Event object
        ecffile = 'S6_' + eventlabel + '.ecf'
        meta = S6MetaClass(ecf_path, ecffile)
    else:
        meta = S6MetaClass(**input_meta.__dict__)

    meta.version = version
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

    meta = S6MetaClass(**me.mergeevents(meta, s5_meta).__dict__)
    meta.set_defaults()

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
            if not isinstance(bg_hw_val, str):
                # Only divide if value is not a string (spectroscopic modes)
                bg_hw_val //= meta.expand
            meta.run_s6 = util.makedirectory(meta, 'S6', meta.run_s6,
                                             ap=spec_hw_val//meta.expand,
                                             bg=bg_hw_val)

    for meta.spec_hw_val in meta.spec_hw_range:
        for meta.bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            # Load in the S5 metadata used for this particular aperture pair
            meta = load_specific_s5_meta_info(meta)

            # Directory structure should not use expanded HW values
            meta.spec_hw_val //= meta.expand
            if not isinstance(meta.bg_hw_val, str):
                # Only divide if value is not a string (spectroscopic modes)
                meta.bg_hw_val //= meta.expand
            # Get the directory for Stage 6 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S6', meta.run_s6,
                                                ap=meta.spec_hw_val,
                                                bg=meta.bg_hw_val)

            # Copy existing S5 log file and resume log
            meta.s6_logname = meta.outputdir+'S6_'+meta.eventlabel+'.log'
            log = logedit.Logedit(meta.s6_logname, read=meta.s5_logname)
            log.writelog("\nStarting Stage 6: Plot Spectra\n")
            log.writelog(f"Eureka! Version: {meta.version}", mute=True)
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

            # FINDME: For now this is assuming that the data is in units of
            # microns We should add something to S3 that notes what the units
            # of the wavelength were in the FITS file
            meta.wavelengths *= units.um.to(meta.x_unit,
                                            equivalencies=units.spectral())
            meta.wave_errs *= units.um.to(meta.x_unit,
                                          equivalencies=units.spectral())
            physical_type = str(meta.x_unit.physical_type).title()
            if physical_type == 'Length':
                physical_type = 'Wavelength'
            label_unit = meta.x_unit.name
            if label_unit == 'um':
                label_unit = r'$\mu$m'
            meta.xlabel = physical_type+' ('+label_unit+')'

            fit_methods = meta.fit_method.strip('[').strip(']').strip()
            fit_methods = fit_methods.split(',')

            zipped_vals = zip(meta.y_params, meta.y_scalars, meta.y_labels,
                              meta.y_label_units)
            for vals in zipped_vals:
                (meta.y_param, meta.y_scalar,
                 meta.y_label, meta.y_label_unit) = vals
                log.writelog(f'Plotting {meta.y_param}...')

                meta.y_param_basic = meta.y_param.split('_pl')[0]
                meta.y_param_basic = meta.y_param_basic.split('_ch')[0]
                meta.y_param_basic = meta.y_param_basic.split('^')[0]
                if meta.y_param[-2:] == '^2':
                    meta.y_param_basic += '^2'

                # Figure out which channel we're working with
                channelNumber = meta.y_param.split('_ch')[-1]
                channelNumber = channelNumber.split('_pl')[0]
                channelNumber = channelNumber.split('^')[0]
                if channelNumber.isnumeric():
                    channelNumber = int(channelNumber)
                else:
                    channelNumber = 0
                meta.channelNumber = channelNumber

                # Figure out which planet we're working with
                planetNumber = meta.y_param.split('_pl')[-1]
                planetNumber = planetNumber.split('_ch')[0]
                planetNumber = planetNumber.split('^')[0]
                if planetNumber.isnumeric():
                    planetNumber = int(planetNumber)
                else:
                    # This is a parameter that changes with planet
                    planetNumber = 0
                meta.planetNumber = planetNumber

                meta.spectrum_median = None
                meta.spectrum_err = None

                # Read in S5 fitted values
                if meta.y_param_basic == 'fn':
                    # Compute nightside flux
                    meta = compute_fn(meta, log, fit_methods)
                elif 'offset_order' in meta.y_param:
                    # Compute phase curve offset of given order
                    meta = compute_offset(meta, log, fit_methods)
                elif 'amp_order' in meta.y_param:
                    # Compute phase curve amplitude of given order
                    meta = compute_amp(meta, log, fit_methods)
                elif 'pc_offset' in meta.y_param:
                    # Compute phase curve offset from all orders
                    meta = compute_pc_offset(meta, log, fit_methods)
                elif 'pc_amp' in meta.y_param:
                    # Compute phase curve amplitude from all orders
                    meta = compute_pc_amp(meta, log, fit_methods)
                elif 'morning_limb' in meta.y_param:
                    # Compute Harmonica's planet transmission strings
                    meta = compute_strings(meta, log, fit_methods,
                                           limb='morning')
                elif 'evening_limb' in meta.y_param:
                    # Compute Harmonica's planet transmission strings
                    meta = compute_strings(meta, log, fit_methods,
                                           limb='evening')
                else:
                    # Just load the parameter
                    if meta.sharedp:
                        meta = parse_s5_saves(meta, log, fit_methods, 'shared')
                    else:
                        meta = parse_unshared_saves(meta, log, fit_methods)

                if ((meta.spectrum_median is None)
                        or all(x is None for x in meta.spectrum_median)):
                    # The parameter could not be found - skip it
                    continue

                # Manipulate fitted values if needed
                if (meta.y_param_basic[:2] == 'rp' and
                        meta.y_param_basic[-2:] == '^2'):
                    meta = compute_transit_depth(meta)
                elif meta.y_param_basic in ['1/r1', '1/r3']:
                    meta = compute_timescale(meta)

                planetSuffix = getPlanetSuffix(meta)
                channelSuffix = getChannelSuffix(meta)

                if meta.y_label is None:
                    # Provide some default formatting
                    if (meta.y_param_basic[:2] == 'rp' and
                            meta.y_param_basic[-2:] == '^2'):
                        # Transit depth
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = '$(R_{\\rm p'+suffix+'}/R_{\\rm *})^2$'
                    elif meta.y_param_basic[:2] == 'rp':
                        # Radius ratio
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = '$R_{\\rm p'+suffix+'}/R_{\\rm *}$'
                    elif meta.y_param_basic in ['fp', 'fpfs']:
                        # Eclipse depth
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = '$F_{\\rm p'+suffix+',day}/F_{\\rm *}$'
                    elif meta.y_param_basic == 'fn':
                        # Nightside emission
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = ('$F_{\\rm p'+suffix + ',night'
                                        '}/F_{\\rm *}$')
                    elif meta.y_param_basic == 't0':
                        # Time of transit
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = '$t_{\\rm 0'+suffix+'}$'
                    elif meta.y_param_basic == 'AmpSin1':
                        # Sine amplitude
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = 'Amplitude of $\\sin(\\phi)$'+suffix
                    elif meta.y_param_basic == 'AmpSin2':
                        # Sine2 amplitude
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = 'Amplitude of $\\sin(2\\phi)$'+suffix
                    elif meta.y_param_basic == 'AmpCos1':
                        # Cosine amplitude
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = 'Amplitude of $\\cos(\\phi)$'+suffix
                    elif meta.y_param_basic == 'AmpCos2':
                        # Cosine2 amplitude
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = 'Amplitude of $\\cos(2\\phi)$'+suffix
                    elif meta.y_param_basic == 'pc_offset':
                        # Phase Curve Offset, first order
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = 'Phase Curve Offset'+suffix
                        if meta.y_label_unit is None:
                            meta.y_label_unit = '($^{\\circ}$E)'
                    elif meta.y_param_basic == 'pc_amp':
                        # Phase Curve Amplitude, first order
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = 'Phase Curve Amplitude'+suffix
                    elif meta.y_param_basic[:12] == 'offset_order':
                        # Phase Curve Offset, nth order
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = (f'Order {meta.y_param_basic[12:]}\n' +
                                        'Phase Curve Offset' + suffix)
                        if meta.y_label_unit is None:
                            meta.y_label_unit = '($^{\\circ}$E)'
                    elif meta.y_param_basic[:9] == 'amp_order':
                        # Phase Curve Amplitude, nth order
                        suffix = planetSuffix+channelSuffix
                        meta.y_label = (f'Order {meta.y_param_basic[9:]}\n' +
                                        'Phase Curve Amplitude' + suffix)
                    elif meta.y_param_basic in [f'u{i}' for i in range(1, 5)]:
                        # Limb darkening parameter
                        suffix = channelSuffix
                        meta.y_label = '$u_{\\rm '+meta.y_param_basic[-1]+'}$'
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
                        meta.y_label += ' for '+limb_law+suffix
                    elif meta.y_param_basic in [f'c{i}' for i in range(0, 10)]:
                        # Polynomial in time coefficient
                        suffix = channelSuffix
                        meta.y_label = ('$c_{\\rm '+meta.y_param_basic[1:] +
                                        '}$'+suffix)
                    elif meta.y_param_basic in [f'r{i}' for i in range(6)]:
                        # Exponential ramp parameters
                        suffix = channelSuffix
                        meta.y_label = ('$r_{\\rm '+meta.y_param_basic[1:] +
                                        '}$'+suffix)
                    elif meta.y_param_basic in ['1/r1', '1/r3']:
                        # Exponential ramp timescales
                        suffix = channelSuffix
                        meta.y_label = ('$1/r_{\\rm '+meta.y_param_basic[-1] +
                                        '}$'+suffix)
                    else:
                        meta.y_label = meta.y_param

                if meta.y_label_unit is None:
                    if meta.y_scalar == 1e6:
                        meta.y_label_unit = ' (ppm)'
                    elif meta.y_scalar == 100:
                        meta.y_label_unit = r' (%)'
                    elif meta.y_scalar != 1:
                        meta.y_label_unit = f' * {meta.y_scalar}'
                    else:
                        meta.y_label_unit = ''
                elif meta.y_label_unit[0] != ' ':
                    # Make sure there's a leading space for proper formatting
                    meta.y_label_unit = ' '+meta.y_label_unit

                if (rcParams['text.usetex'] and
                        (meta.y_label_unit.count(r'\%') !=
                         meta.y_label_unit.count('%'))):
                    # Need to escape % with \ for LaTeX
                    meta.y_label_unit = meta.y_label_unit.replace('%', r'\%')

                # Add any units to the y label
                meta.y_label += meta.y_label_unit

                if meta.model_spectrum is not None:
                    model_x, model_y = load_model(meta, log, meta.x_unit)
                else:
                    model_x = None
                    model_y = None

                # Make the spectrum plot
                if meta.isplots_S6 >= 1:
                    plots.plot_spectrum(meta, model_x, model_y, meta.y_scalar,
                                        meta.y_label, meta.xlabel)

                # Should we also make the scale_height version of the figure?
                make_fig6301 = (meta.isplots_S6 >= 3 and meta.has_fig6301reqs
                                and meta.y_param[:2] == 'rp')
                if make_fig6301:
                    # Make spectrum plot with scale height on the 2nd y-axis
                    scale_height = compute_scale_height(meta, log)
                    plots.plot_spectrum(meta, model_x, model_y, meta.y_scalar,
                                        meta.y_label, meta.xlabel,
                                        scale_height, meta.planet_R0)

                save_table(meta, log)

            # Copy S5 text files to a single h5 file
            meta, lc = convert_s5_LC(meta, log)

            # make citations for current stage
            util.make_citations(meta, 6)

            # Save results
            log.writelog('Saving results')
            fname = meta.outputdir+'S6_'+meta.eventlabel+"_Meta_Save"
            me.saveevent(meta, fname, save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return meta, lc


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
    if (meta.y_param_basic[:2] == 'rp' and
            meta.y_param_basic[-2:] == '^2'):
        y_param = meta.y_param[:-2]
    elif meta.y_param_basic in ['1/r1', '1/r3']:
        y_param = meta.y_param[2:]
    else:
        y_param = meta.y_param

    if 'dynesty' in fit_methods:
        fitter = 'dynesty'
    elif 'emcee' in fit_methods:
        fitter = 'emcee'
    elif 'lsq' in fit_methods:
        fitter = 'lsq'
    elif 'nuts' in fit_methods:
        fitter = 'nuts'
    elif 'exoplanet' in fit_methods:
        fitter = 'exoplanet'
    else:
        raise ValueError('No recognized fitters in fit_methods = '
                         f'{fit_methods}')

    lowers = []
    uppers = []
    medians = []
    errs = []

    if fitter in ['dynesty', 'emcee', 'nuts']:
        fname = f'S5_{fitter}_fitparams_{channel_key}.csv'
        fitted_values = pd.read_csv(meta.inputdir+fname, escapechar='#',
                                    skipinitialspace=True)
        full_keys = list(fitted_values["Parameter"])

        fname = f'S5_{fitter}_samples_{channel_key}'

        temp_keys = [y_param+f'_ch{c}' if c > 0 else y_param
                     for c in range(meta.nspecchan)]
        keys = [key for key in temp_keys if key in full_keys]
        if len(keys) == 0:
            log.writelog(f'  Parameter {y_param} was not in the list of '
                         'fitted parameters which includes:\n  ['
                         + ', '.join(full_keys)+']')
            log.writelog(f'  Skipping {y_param}')
            return meta

        for key in keys:
            ind = np.where(fitted_values["Parameter"] == key)[0][0]
            lowers.append(fitted_values["50th"][ind]
                          - fitted_values["16th"][ind])
            uppers.append(fitted_values["84th"][ind]
                          - fitted_values["50th"][ind])
            medians.append(fitted_values["50th"][ind])

        errs = np.array([lowers, uppers])
        medians = np.array(medians)
    else:
        fname = f'S5_{fitter}_fitparams_{channel_key}.csv'
        fitted_values = pd.read_csv(meta.inputdir+fname, escapechar='#',
                                    skipinitialspace=True)
        full_keys = list(fitted_values["Parameter"])
        temp_keys = [y_param+f'_ch{c}' if c > 0 else y_param
                     for c in range(meta.nspecchan)]
        keys = [key for key in temp_keys if key in full_keys]
        if len(keys) == 0:
            log.writelog(f'Parameter {y_param} was not in the list of '
                         'fitted parameters which includes:\n['
                         + ', '.join(full_keys)+']')
            log.writelog(f'Skipping {y_param}')
            return meta

        medians = []
        for key in keys:
            ind = np.where(fitted_values["Parameter"] == key)[0][0]
            if "50th" in fitted_values.keys():
                medians.append(fitted_values["50th"][ind])
            else:
                medians.append(fitted_values["Mean"][ind])
        medians = np.array(medians)

        # if lsq or exoplanet, no uncertainties
        errs = np.ones((2, len(medians)))*np.nan

    meta.spectrum_median = medians
    meta.spectrum_err = errs

    return meta


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
    spectrum_median = []
    spectrum_err = []
    for channel in range(meta.nspecchan):
        ch_number = str(channel).zfill(len(str(meta.nspecchan)))
        channel_key = f'ch{ch_number}'
        try:
            meta = parse_s5_saves(meta, log, fit_methods, channel_key)
        except FileNotFoundError:
            # This channel was skipped or was all masked.
            # Insert NaNs in its place.
            spectrum_median.extend([np.nan,])
            spectrum_err.extend([[np.nan, np.nan]])
            continue
        if meta.spectrum_median is None:
            # Parameter wasn't found, so don't keep looking for it
            meta.spectrum_median = np.array([None for _ in
                                             range(meta.nspecchan)])
            meta.spectrum_err = np.array([None for _ in range(meta.nspecchan)])
            return meta
        spectrum_median.extend(meta.spectrum_median)
        spectrum_err.extend(meta.spectrum_err.T)

    meta.spectrum_median = np.array(spectrum_median)
    meta.spectrum_err = np.array(spectrum_err).T

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
        lower = meta.spectrum_median - meta.spectrum_err[0, :]
        upper = meta.spectrum_median + meta.spectrum_err[1, :]

    meta.spectrum_median *= np.abs(meta.spectrum_median)

    if not np.all(np.isnan(meta.spectrum_err)):
        lower = meta.spectrum_median - lower*np.abs(lower)
        upper = upper*np.abs(upper) - meta.spectrum_median
        meta.spectrum_err = np.append(lower.reshape(1, -1),
                                      upper.reshape(1, -1), axis=0)

    return meta


def compute_timescale(meta):
    """Convert the fitted r1 or r3 value to a timescale.

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
        lower = 1/(median-lower) - 1/median
        upper = 1/median - 1/(median+upper)
        meta.spectrum_err = np.append(lower.reshape(1, -1),
                                      upper.reshape(1, -1), axis=0)
    meta.spectrum_median = 1/median

    return meta


def convert_s5_LC(meta, log):
    '''
    Loads spectroscopic light curves save files from S5 and write as
    single Xarray save file.

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
    lc : Astreaus object
        Data object of time-like arrays (light curve).
    '''
    event_ap_bg = (meta.eventlabel+"_ap"+str(meta.spec_hw_val)+'_bg' +
                   str(meta.bg_hw_val))

    if meta.sharedp:
        niter = 1
    else:
        niter = meta.nspecchan
    wavelengths = np.unique(meta.wavelengths)
    lc_array_setup = False
    for ch in range(niter):
        # Get the channel key
        if meta.sharedp:
            channel_key = 'shared'
        else:
            nzfill = int(np.floor(np.log10(meta.nspecchan))+1)
            channel_key = 'ch'+str(ch).zfill(nzfill)

        # Load text file
        fname = f'S5_{event_ap_bg}_Table_Save_{channel_key}.txt'
        full_fname = meta.inputdir+fname
        try:
            lc_table = astropytable.readtable(full_fname)
        except FileNotFoundError:
            # This channel was skipped or was all masked.
            # We'll insert NaNs in its place lower down.
            continue

        # Assign known values to array
        lc_table.remove_column('wavelength')
        lc_table.remove_column('bin_width')
        if not lc_array_setup:
            lc_array_setup = True
            # Record time array
            time = lc_table['time']
            lc_table.remove_column('time')
            # Get remaining column names and number
            colnames = lc_table.colnames
            n_col = len(colnames)
            n_int = len(time)
            # Create numpy array to hold data
            lc_array = np.ones((n_col, niter, n_int))*np.nan
        else:
            lc_table.remove_column('time')
        # Assign remaining values to array
        for i, col in enumerate(lc_table.itercols()):
            lc_array[i, ch] = col.value

    # Create Xarray DataArrays and dictionary
    flux_units = 'Normalized'
    wave_units = meta.wave_units
    time_units = meta.time_units
    lc_da = []
    dict = {}
    for i in range(n_col):
        lc_da.append(xrio.makeLCDA(lc_array[i], wavelengths, time, flux_units,
                                   wave_units, time_units, name=colnames[i]))
        dict[colnames[i]] = lc_da[-1]

    # Create Xarray Dataset
    lc = xrio.makeDataset(dict)
    # Write to file
    meta.lc_filename_s6 = (meta.outputdir+'S6_'+event_ap_bg + "_LC")
    xrio.writeXR(meta.lc_filename_s6, lc)
    return meta, lc


def load_s5_saves(meta, log, fit_methods, n_samples=1):
    """
    Return samples from Stage 5 uncertainty fits.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    fit_methods : string
        The type of uncertainty fitting method used in Stage 5.
    n_samples : int; optional
        The number of samples used when parameter is not found.
        Default is 1.

    Returns
    -------
    samples : list
        A list of sample arrays, one for each channel.
    """
    if 'dynesty' in fit_methods:
        fitter = 'dynesty'
    elif 'emcee' in fit_methods:
        fitter = 'emcee'
    elif 'lsq' in fit_methods:
        fitter = 'lsq'
    # Gradient based models: nuts > exoplanet
    elif 'nuts' in fit_methods:
        fitter = 'nuts'
    elif 'exoplanet' in fit_methods:
        fitter = 'exoplanet'
    else:
        raise ValueError('No recognized fitters in fit_methods = '
                         f'{fit_methods}')
    meta.fitter = fitter

    if fitter in ['nuts', 'dynesty', 'emcee']:
        if meta.sharedp:
            niter = 1
        else:
            niter = meta.nspecchan
        samples = []
        for ch in range(niter):
            # Get the channel key
            if meta.sharedp:
                channel_key = 'shared'
            else:
                nzfill = int(np.floor(np.log10(meta.nspecchan))+1)
                channel_key = 'ch'+str(ch).zfill(nzfill)

            fname = f'S5_{fitter}_samples_{channel_key}'

            # Load HDF5 files
            full_fname = meta.inputdir+fname+'.h5'
            ds = xrio.readXR(full_fname, verbose=False)
            if ds is None:
                sample = np.zeros(n_samples)
            else:
                if meta.y_param in list(ds._variables):
                    sample = ds[meta.y_param].values
                    if ch == 0:
                        n_samples = len(sample)
                else:
                    sample = np.zeros(n_samples)
            samples.append(sample)
    else:
        # No samples for lsq, so just shape it as a single value
        if meta.sharedp:
            meta = parse_s5_saves(meta, log, fit_methods, 'shared')
        else:
            meta = parse_unshared_saves(meta, log, fit_methods)
        samples = np.array(meta.spectrum_median)
        if all(x is None for x in samples):
            samples = np.zeros((meta.nspecchan, 1))

    return samples


def compute_strings(meta, log, fit_methods, limb):
    """
    Compute transit depth at morning/evening limb.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    fit_methods : string
        The type of uncertainty fitting method used in Stage 5.
    limb : string
        Can be either 'morning' or 'evening'

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated meta data object.
    """
    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load a0 string coefficients
    meta.y_param = 'a0'+suffix
    a0 = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in a0):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters')
        log.writelog(f'  Skipping {y_param}')
        return meta
    n_samples = len(a0[0])

    # Load string coefficients
    coeffs = ['a1', 'b1', 'a2', 'b2', 'a3', 'b3']
    ab_list = []
    for coeff in coeffs:
        meta.y_param = coeff+suffix
        vals = load_s5_saves(meta, log, fit_methods, n_samples=n_samples)
        if all(np.all(v == 0) for v in vals):
            # The parameter could not be found - assume fixed to 0
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters, assumed to be 0.')
        ab_list.append(vals)
    a1, b1, a2, b2, a3, b3 = ab_list

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []

    # Convert full angle in degrees to half angle in radians
    # e.g., an angle of 60 degrees spans -30 to +30 degrees
    rad = meta.strings_angle*np.pi/360
    if limb == 'morning':
        theta = np.linspace(-rad, rad, 100)
    elif limb == 'evening':
        theta = np.linspace(np.pi-rad, np.pi+rad, 100)

    # Choose a subset of samples
    ss = meta.strings_stepsize

    ht = HarmonicaTransit()
    for i in tqdm(range(meta.nspecchan)):
        if np.all(a0[i] == 0):
            # Channel wasn't found
            meta.spectrum_median.append(np.nan)
            meta.spectrum_err.append([np.nan, np.nan])
        else:
            # Compute transmission string
            ab = np.array([a0[i][::ss],
                           a1[i][::ss], b1[i][::ss],
                           a2[i][::ss], b2[i][::ss],
                           a3[i][::ss], b3[i][::ss]]).T
            ht.set_planet_transmission_string(ab)
            samples = ht.get_planet_transmission_string(theta)
            sm_16, sm_50, sm_84 = np.percentile(samples**2,
                                                [16., 50., 84.], axis=0)
            # Mean transit depth over range of angles
            sm_mean = np.mean(sm_50)
            meta.spectrum_median.append(sm_mean)
            # Two-sided uncertainty over range of angles
            meta.spectrum_err.append([np.mean(sm_84)-sm_mean,
                                     sm_mean-np.mean(sm_16)])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_offset(meta, log, fit_methods):
    # Save meta.y_param
    y_param = meta.y_param

    # Figure out the desired order
    orderSuffix = meta.y_param[-1]
    if not orderSuffix.isnumeric():
        # First order doesn't have a numeric suffix
        orderSuffix = '1'

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load sine amplitude
    meta.y_param = 'AmpSin'+orderSuffix+suffix
    ampsin = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampsin):
        meta.y_param = f'Y{orderSuffix}1{suffix}'
        ampsin = [-1*i for i in load_s5_saves(meta, log, fit_methods)]
        if all(np.all(v == 0) for v in ampsin):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine amplitude
    meta.y_param = 'AmpCos'+orderSuffix+suffix
    ampcos = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampcos):
        meta.y_param = f'Y{orderSuffix}0{suffix}'
        ampcos = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in ampcos):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []

    for i in range(meta.nspecchan):
        offsets = -np.arctan2(ampsin[i], ampcos[i])*180/np.pi
        if orderSuffix == '2':
            offsets /= 2
        offset = np.percentile(np.array(offsets), [16, 50, 84])[[1, 2, 0]]
        offset[1] -= offset[0]
        offset[2] = offset[0]-offset[2]
        meta.spectrum_median.append(offset[0])
        meta.spectrum_err.append(offset[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_amp(meta, log, fit_methods):
    if (('nuts' in fit_methods or 'exoplanet' in fit_methods) and
            'sinusoid_pc' not in meta.run_myfuncs):
        return compute_amp_starry(meta, log, fit_methods)

    # Save meta.y_param
    y_param = meta.y_param

    # Figure out the desired order
    orderSuffix = meta.y_param[-1]
    if not orderSuffix.isnumeric():
        # First order doesn't have a numeric suffix
        orderSuffix = '1'

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            # The parameter could not be found - skip it
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load sine amplitude
    meta.y_param = 'AmpSin'+orderSuffix+suffix
    ampsin = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampsin):
        meta.y_param = f'Y{orderSuffix}1{suffix}'
        ampsin = [-1*i for i in load_s5_saves(meta, log, fit_methods)]
        if all(np.all(v == 0) for v in ampsin):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine amplitude
    meta.y_param = 'AmpCos'+orderSuffix+suffix
    ampcos = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampcos):
        meta.y_param = f'Y{orderSuffix}0{suffix}'
        ampcos = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in ampcos):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []

    for i in range(meta.nspecchan):
        amps = fp[i]*np.sqrt(ampcos[i]**2+ampsin[i]**2)*2
        amp = np.percentile(np.array(amps), [16, 50, 84])[[1, 2, 0]]
        amp[1] -= amp[0]
        amp[2] = amp[0]-amp[2]
        meta.spectrum_median.append(amp[0])
        meta.spectrum_err.append(amp[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_pc_offset_poet(meta, log, fit_methods):
    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load cosine offset terms
    meta.y_param = 'cos2_off'+suffix
    cos2_off = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'cos1_off'+suffix
    cos1_off = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in cos1_off):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.')
        log.writelog(f'  Skipping {y_param}')
        return meta

    # Load cosine amplitude terms
    meta.y_param = 'cos2_amp'+suffix
    cos2_amp = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'cos1_amp'+suffix
    cos1_amp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in cos1_amp):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.')
        log.writelog(f'  Skipping {y_param}')
        return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []
    deg = np.linspace(180, -180, meta.pc_nstep)
    ss = meta.pc_stepsize
    for i in tqdm(range(meta.nspecchan)):
        # Compute phase curve
        phaseVars = (cos1_amp[i][::ss, np.newaxis]/2 *
                     np.cos(2*np.pi/360*(deg-cos1_off[i][::ss, np.newaxis])) +
                     cos2_amp[i][::ss, np.newaxis]/2 *
                     np.cos(4*np.pi/360*(deg-cos2_off[i][::ss, np.newaxis])))
        # Compute offsets in degrees
        offset_deg = deg[np.argmax(phaseVars, axis=1)]
        offset = np.percentile(np.array(offset_deg), [16, 50, 84])[[1, 2, 0]]
        offset[1] -= offset[0]
        offset[2] = offset[0]-offset[2]
        meta.spectrum_median.append(offset[0])
        meta.spectrum_err.append(offset[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_pc_offset(meta, log, fit_methods):
    if ('poet_pc' in meta.run_myfuncs and
            'sinusoid_pc' not in meta.run_myfuncs):
        return compute_pc_offset_poet(meta, log, fit_methods)

    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load sine amplitude
    meta.y_param = 'AmpSin2'+suffix
    ampsin2 = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'AmpSin1'+suffix
    ampsin1 = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampsin1):
        # The parameter could not be found - try a different one
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.  Attempting to load Y11.')
        meta.y_param = f'Y21{suffix}'
        ampsin2 = [-1*i for i in load_s5_saves(meta, log, fit_methods)]
        meta.y_param = f'Y11{suffix}'
        ampsin1 = [-1*i for i in load_s5_saves(meta, log, fit_methods)]
        if all(np.all(v == 0) for v in ampsin1):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters.')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine amplitude
    meta.y_param = 'AmpCos2'+suffix
    ampcos2 = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'AmpCos1'+suffix
    ampcos1 = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampcos1):
        # The parameter could not be found - try a different one
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.  Attempting to load Y10.')
        meta.y_param = f'Y20{suffix}'
        ampcos2 = load_s5_saves(meta, log, fit_methods)
        meta.y_param = f'Y10{suffix}'
        ampcos1 = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in ampcos1):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters.')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []
    phi = np.linspace(-np.pi, np.pi, meta.pc_nstep)[np.newaxis]
    deg = np.linspace(180, -180, meta.pc_nstep)
    ss = meta.pc_stepsize
    for i in tqdm(range(meta.nspecchan)):
        # Compute phase curve
        phaseVars = (1. + ampcos1[i][::ss, np.newaxis]*(np.cos(phi)-1.) +
                     ampsin1[i][::ss, np.newaxis]*np.sin(phi) +
                     ampcos2[i][::ss, np.newaxis]*(np.ma.cos(2.*phi)-1.) +
                     ampsin2[i][::ss, np.newaxis]*np.ma.sin(2.*phi))
        # Compute offsets in degrees
        offset_deg = deg[np.argmax(phaseVars, axis=1)]
        offset = np.percentile(np.array(offset_deg), [16, 50, 84])[[1, 2, 0]]
        offset[1] -= offset[0]
        offset[2] = offset[0]-offset[2]
        meta.spectrum_median.append(offset[0])
        meta.spectrum_err.append(offset[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_pc_amp_poet(meta, log, fit_methods):
    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            # The parameter could not be found - skip it
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine offset terms
    meta.y_param = 'cos2_off'+suffix
    cos2_off = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'cos1_off'+suffix
    cos1_off = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in cos1_off):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.')
        log.writelog(f'  Skipping {y_param}')
        return meta

    # Load cosine amplitude terms
    meta.y_param = 'cos2_amp'+suffix
    cos2_amp = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'cos1_amp'+suffix
    cos1_amp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in cos1_amp):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.')
        log.writelog(f'  Skipping {y_param}')
        return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []
    deg = np.linspace(180, -180, meta.pc_nstep)
    ss = meta.pc_stepsize
    for i in tqdm(range(meta.nspecchan)):
        # Compute phase curve
        phaseVars = (cos1_amp[i][::ss, np.newaxis]/2 *
                     np.cos(2*np.pi/360*(deg-cos1_off[i][::ss, np.newaxis])) +
                     cos2_amp[i][::ss, np.newaxis]/2 *
                     np.cos(4*np.pi/360*(deg-cos2_off[i][::ss, np.newaxis])))
        # Compute PC amplitude
        amps = fp[i][::ss]*(np.max(phaseVars, axis=1) -
                            np.min(phaseVars, axis=1))
        amp = np.percentile(np.array(amps), [16, 50, 84])[[1, 2, 0]]
        amp[1] -= amp[0]
        amp[2] = amp[0]-amp[2]
        meta.spectrum_median.append(amp[0])
        meta.spectrum_err.append(amp[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_pc_amp(meta, log, fit_methods):
    if ('poet_pc' in meta.run_myfuncs and
            'sinusoid_pc' not in meta.run_myfuncs):
        return compute_pc_amp_poet(meta, log, fit_methods)

    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            # The parameter could not be found - skip it
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load sine amplitude
    meta.y_param = 'AmpSin2'+suffix
    ampsin2 = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'AmpSin1'+suffix
    ampsin1 = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampsin1):
        # The parameter could not be found - try a different one
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.  Attempting to load Y11.')
        meta.y_param = f'Y21{suffix}'
        ampsin2 = [-1*i for i in load_s5_saves(meta, log, fit_methods)]
        meta.y_param = f'Y11{suffix}'
        ampsin1 = [-1*i for i in load_s5_saves(meta, log, fit_methods)]
        if all(np.all(v == 0) for v in ampsin1):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters.')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine amplitude
    meta.y_param = 'AmpCos2'+suffix
    ampcos2 = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'AmpCos1'+suffix
    ampcos1 = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampcos1):
        # The parameter could not be found - try a different one
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.  Attempting to load Y10.')
        meta.y_param = f'Y20{suffix}'
        ampcos2 = load_s5_saves(meta, log, fit_methods)
        meta.y_param = f'Y10{suffix}'
        ampcos1 = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in ampcos1):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters.')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []
    phi = np.linspace(-np.pi, np.pi, meta.pc_nstep)[np.newaxis]
    ss = meta.pc_stepsize
    for i in tqdm(range(meta.nspecchan)):
        # Compute phase curve
        phaseVars = (1. + ampcos1[i][::ss, np.newaxis]*(np.cos(phi)-1.) +
                     ampsin1[i][::ss, np.newaxis]*np.sin(phi) +
                     ampcos2[i][::ss, np.newaxis]*(np.ma.cos(2.*phi)-1.) +
                     ampsin2[i][::ss, np.newaxis]*np.ma.sin(2.*phi))
        # Compute PC amplitude
        amps = fp[i][::ss]*(np.max(phaseVars, axis=1) -
                            np.min(phaseVars, axis=1))
        amp = np.percentile(np.array(amps), [16, 50, 84])[[1, 2, 0]]
        amp[1] -= amp[0]
        amp[2] = amp[0]-amp[2]
        meta.spectrum_median.append(amp[0])
        meta.spectrum_err.append(amp[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_amp_starry(meta, log, fit_methods, nsamp=1e3):
    nsamp = int(nsamp)

    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            # The parameter could not be found - skip it
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    nsamp = min([nsamp, len(fp[0])])
    inds = np.random.randint(0, len(fp[0]), nsamp)

    class temp_class:
        def __init__(self):
            pass

    # Load map parameters
    if y_param[-1].isnumeric():
        ydeg = int(y_param[-1])
    else:
        ydeg = 1
    temp = temp_class()
    ell = ydeg
    for m in range(-ell, ell+1):
        meta.y_param = f'Y{ell}{m}{suffix}'
        val = load_s5_saves(meta, log, fit_methods)
        if val.shape[-1] != 0:
            setattr(temp, f'Y{ell}{m}{suffix}', val[:, inds])

    # Reset meta.y_param
    meta.y_param = y_param

    # If no parameters could not be found - skip it
    if len(temp.__dict__.keys()) == 0:
        log.writelog('  No Ylm parameters were found...')
        log.writelog(f'  Skipping {y_param}')
        return meta

    meta.spectrum_median = []
    meta.spectrum_err = []

    planet_map = starry.Map(ydeg=ydeg, nw=nsamp)
    planet_map2 = starry.Map(ydeg=ydeg, nw=nsamp)
    for i in range(meta.nspecchan):
        inds = np.random.randint(0, len(fp[i]), nsamp)
        ell = ydeg
        for m in range(-ell, ell+1):
            if hasattr(temp, f'Y{ell}{m}{suffix}'):
                planet_map[ell, m, :] = getattr(temp, f'Y{ell}{m}{suffix}')[i]
                planet_map2[ell, m, :] = getattr(temp, f'Y{ell}{m}{suffix}')[i]
        planet_map.amp = fp[i][inds]/planet_map2.flux(theta=0)[0]

        theta = np.linspace(0, 359, 360)
        fluxes = np.array(planet_map.flux(theta=theta).eval())
        min_fluxes = np.min(fluxes, axis=0)
        max_fluxes = np.max(fluxes, axis=0)
        amps = (max_fluxes-min_fluxes)
        amp = np.percentile(amps, [16, 50, 84])[[1, 2, 0]]
        amp[1] -= amp[0]
        amp[2] = amp[0]-amp[2]
        meta.spectrum_median.append(amp[0])
        meta.spectrum_err.append(amp[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_fn(meta, log, fit_methods):
    if (('nuts' in fit_methods or 'exoplanet' in fit_methods) and
            'sinusoid_pc' not in meta.run_myfuncs):
        return compute_fn_starry(meta, log, fit_methods)
    elif ('poet_pc' in meta.run_myfuncs and
            'sinusoid_pc' not in meta.run_myfuncs):
        return compute_fn_poet(meta, log, fit_methods)
    elif ('quasilambert_pc' in meta.run_myfuncs):
        meta.spectrum_median = np.zeros(meta.nspecchan)
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
        return meta

    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        # The parameter could not be found - try fpfs
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine amplitude
    meta.y_param = 'AmpCos1'+suffix
    ampcos = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in ampcos):
        # FINDME: The following only works if the model does not include any
        # terms other than Y10, Y11, Y20, Y22 (or other higher order terms
        # which evaluate to zero at the anti-stellar point). In general, should
        # use the compute_fp function.
        # FINDME: This is also not the nightside flux for starry models - just
        # the anti-stellar point flux. Really do need to use compute_fp instead
        meta.y_param = 'Y10'+suffix
        ampcos = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in ampcos):
            # The parameter could not be found - skip it
            log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []

    for i in range(meta.nspecchan):
        fluxes = fp[i]*(1-2*ampcos[i])
        flux = np.percentile(np.array(fluxes), [16, 50, 84])[[1, 2, 0]]
        flux[1] -= flux[0]
        flux[2] = flux[0]-flux[2]
        meta.spectrum_median.append(flux[0])
        meta.spectrum_err.append(flux[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_fn_poet(meta, log, fit_methods):

    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        # The parameter could not be found - try fpfs
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    # Load cosine offset terms
    meta.y_param = 'cos2_off'+suffix
    cos2_off = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'cos1_off'+suffix
    cos1_off = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in cos1_off):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.')
        log.writelog(f'  Skipping {y_param}')
        return meta

    # Load cosine amplitude terms
    meta.y_param = 'cos2_amp'+suffix
    cos2_amp = load_s5_saves(meta, log, fit_methods)
    meta.y_param = 'cos1_amp'+suffix
    cos1_amp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in cos1_amp):
        # The parameter could not be found - skip it
        log.writelog(f'  Parameter {meta.y_param} was not in the list of '
                     'fitted parameters.')
        log.writelog(f'  Skipping {y_param}')
        return meta

    # Reset meta.y_param
    meta.y_param = y_param

    meta.spectrum_median = []
    meta.spectrum_err = []
    # Only need to calculate the flux at two points,
    # anti-stellar (180 deg) and sub-stellar (0 deg)
    deg = np.array([180, 0])
    ieclipse = 1
    for i in range(meta.nspecchan):
        # Compute phase curve
        phaseVars = (cos1_amp[i][:, np.newaxis]/2 *
                     np.cos(2*np.pi/360*(deg-cos1_off[i][:, np.newaxis])) +
                     cos2_amp[i][:, np.newaxis]/2 *
                     np.cos(4*np.pi/360*(deg-cos2_off[i][:, np.newaxis])))
        # Apply normalizing offset
        phaseVars += 1 - (phaseVars[:, ieclipse])[:, np.newaxis]
        # Compute nightside flux (at deg = 180)
        fluxes = fp[i]*phaseVars[:, 0]
        flux = np.percentile(fluxes, [16, 50, 84])[[1, 2, 0]]
        flux[1] -= flux[0]
        flux[2] = flux[0]-flux[2]
        meta.spectrum_median.append(flux[0])
        meta.spectrum_err.append(flux[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

    return meta


def compute_fn_starry(meta, log, fit_methods, nsamp=1e3):
    nsamp = int(nsamp)

    # Save meta.y_param
    y_param = meta.y_param

    suffix = ''
    if meta.planetNumber > 0:
        suffix += f'_pl{meta.planetNumber}'
    if meta.channelNumber > 0:
        suffix += f'_ch{meta.channelNumber}'

    # Load eclipse depth
    meta.y_param = 'fp'+suffix
    fp = load_s5_saves(meta, log, fit_methods)
    if all(np.all(v == 0) for v in fp):
        # The parameter could not be found - try fpfs
        meta.y_param = 'fpfs'+suffix
        fp = load_s5_saves(meta, log, fit_methods)
        if all(np.all(v == 0) for v in fp):
            log.writelog('  Planet flux (fp or fpfs) was not in the list of '
                         'fitted parameters')
            log.writelog(f'  Skipping {y_param}')
            return meta

    nsamp = min([nsamp, len(fp[0])])
    inds = np.random.randint(0, len(fp[0]), nsamp)

    class temp_class:
        def __init__(self):
            pass

    # Load map parameters
    temp = temp_class()
    for ell in range(1, meta.ydeg+1):
        for m in range(-ell, ell+1):
            meta.y_param = f'Y{ell}{m}{suffix}'
            val = load_s5_saves(meta, log, fit_methods)
            if val.shape[-1] != 0:
                setattr(temp, f'Y{ell}{m}{suffix}', val[:, inds])

    # Reset meta.y_param
    meta.y_param = y_param

    # If no parameters could not be found - skip it
    if len(temp.__dict__.keys()) == 0:
        log.writelog('  No Ylm parameters were found...')
        log.writelog(f'  Skipping {y_param}')
        return meta

    meta.spectrum_median = []
    meta.spectrum_err = []

    planet_map = starry.Map(ydeg=meta.ydeg, nw=nsamp)
    planet_map2 = starry.Map(ydeg=meta.ydeg, nw=nsamp)
    for i in range(meta.nspecchan):
        inds = np.random.randint(0, len(fp[i]), nsamp)
        for ell in range(1, meta.ydeg+1):
            for m in range(-ell, ell+1):
                if hasattr(temp, f'Y{ell}{m}{suffix}'):
                    planet_map[ell, m, :] = getattr(temp,
                                                    f'Y{ell}{m}{suffix}')[i]
                    planet_map2[ell, m, :] = getattr(temp,
                                                     f'Y{ell}{m}{suffix}')[i]
        planet_map.amp = fp[i][inds]/planet_map2.flux(theta=0)[0]

        fluxes = planet_map.flux(theta=180)[0].eval()
        flux = np.percentile(np.array(fluxes), [16, 50, 84])[[1, 2, 0]]
        flux[1] -= flux[0]
        flux[2] = flux[0]-flux[2]
        meta.spectrum_median.append(flux[0])
        meta.spectrum_err.append(flux[1:])

    # Convert the lists to an array
    meta.spectrum_median = np.array(meta.spectrum_median)
    if meta.fitter == 'lsq':
        meta.spectrum_err = np.ones((2, meta.nspecchan))*np.nan
    else:
        meta.spectrum_err = np.array(meta.spectrum_err).T

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
        if (meta.y_param_basic[:2] == 'rp' and
                meta.y_param_basic[-2:] == '^2'):
            meta.planet_Rad = np.sqrt(meta.planet_Rad)
        meta.planet_Rad = np.nanmean(meta.planet_Rad)
        meta.planet_Rad *= (meta.star_Rad*constants.R_sun /
                            constants.R_jup).si.value
    if meta.planet_R0 is not None:
        meta.planet_R0 *= (constants.R_jup/(meta.star_Rad *
                                            constants.R_sun)).si.value
    meta.planet_g = ((constants.G*meta.planet_Mass*constants.M_jup) /
                     (meta.planet_Rad*constants.R_jup)**2).si.value
    log.writelog(f'  Calculated g={np.round(meta.planet_g, 2)} m/s^2 '
                 f'with Rp={np.round(meta.planet_Rad, 2)} R_jup '
                 f'and Mp={meta.planet_Mass} M_jup')
    scale_height = (constants.k_B*(meta.planet_Teq*units.K) /
                    ((meta.planet_mu*units.u) *
                     (meta.planet_g*units.m/units.s**2)))
    scale_height = scale_height.si.to('km')
    log.writelog(f'  Calculated H={np.round(scale_height, 2)} with '
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
    if not isinstance(meta.bg_hw, str):
        # Only divide if value is not a string (spectroscopic modes)
        bg_hw = meta.bg_hw//meta.expand
    else:
        bg_hw = meta.bg_hw
    inputdir += f'ap{meta.spec_hw//meta.expand}_bg{bg_hw}'+os.sep
    # Locate the old MetaClass savefile, and load new ECF into
    # that old MetaClass
    meta.inputdir = inputdir
    s5_meta, meta.inputdir, meta.inputdir_raw = \
        me.findevent(meta, 'S5', allowFail=False)
    # Merge S6 meta into old S5 meta
    meta = S6MetaClass(**me.mergeevents(meta, s5_meta).__dict__)

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
                         f'match the current y_param ({meta.y_param}), so not '
                         'using the model for this plot')
            return None, None

    model_path = os.path.join(meta.topdir, *meta.model_spectrum.split(os.sep))
    model_x, model_y = np.loadtxt(model_path, delimiter=meta.model_delimiter).T
    # Convert model_x_unit to x_unit if needed
    model_x_unit = getattr(units, meta.model_x_unit)
    model_x_unit = model_x_unit.to(x_unit, equivalencies=units.spectral())
    model_x *= model_x_unit
    # Figure out if model needs to be converted to Rp/Rs
    sqrt_model = ((meta.model_y_param[:2] == 'rp'
                   and meta.model_y_param[-2:] == '^2')
                  and meta.model_y_param != meta.y_param)
    # Figure out if model needs to be converted to (Rp/Rs)^2
    sq_model = ((meta.model_y_param[:2] == 'rp'
                 and meta.model_y_param[-2:] != '^2')
                and meta.model_y_param != meta.y_param)
    if sqrt_model:
        model_y = np.sqrt(model_y)
    elif sq_model:
        model_y *= model_y
    elif meta.model_y_param != meta.y_param:
        raise AssertionError('Unknown conversion between y_param '
                             f'{meta.y_param} and model_y_param '
                             f'{meta.model_y_param}')

    # Convert the model y-units if needed to match the data
    # y-units requested
    if meta.model_y_scalar != 1:
        model_y *= meta.model_y_scalar

    return model_x, model_y*meta.y_scalar


def save_table(meta, log):
    """Clean y_param for filenames and save the table of values.

    Also calls transit_latex_table().

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    """
    log.writelog('  Saving results as an astropy table')

    event_ap_bg = (meta.eventlabel+"_ap"+str(meta.spec_hw_val)+'_bg' +
                   str(meta.bg_hw_val))
    clean_y_param = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", meta.y_param)
    meta.tab_filename_s6 = (meta.outputdir+'S6_'+event_ap_bg+'_' +
                            clean_y_param+"_Table_Save.txt")
    wavelengths = np.mean(np.append(meta.wave_low.reshape(1, -1),
                                    meta.wave_hi.reshape(1, -1),
                                    axis=0), axis=0)
    wave_errs = (meta.wave_hi-meta.wave_low)/2
    # Trim repeated wavelengths for multwhite fits
    if len(set(wavelengths)) == 1:
        wavelengths = wavelengths[0]
        wave_errs = wave_errs[0]
    astropytable.savetable_S6(meta.tab_filename_s6, meta.y_param, wavelengths,
                              wave_errs, meta.spectrum_median,
                              meta.spectrum_err)

    transit_latex_table(meta, log)

    return


def roundToSigFigs(x, sigFigs=2):
    """Round a value to a requested number of significant figures.

    Parameters
    ----------
    x : numerical type
        A float or int to be rounded.
    sigFigs : int; optional
        The number of significant figures desired, by default 2.

    Returns
    -------
    nDec : int
        The number of decimals corresponding to sigFigs where nDec = -1 for
        a value rounded to the ten's place (e.g. 101 -> 100 if nDec = -1).
    output : str
        x formatted as a string with the requested number of significant
        figures.

    Notes
    -----
    History:

    - 2022-08-22, Taylor J Bell
        Imported code written for SPCA, and optimized for Python3.
    """
    if not np.isfinite(x) or not np.isfinite(np.log10(np.abs(x))):
        return np.nan, ""
    elif not np.isfinite(sigFigs):
        return 10, str(np.round(x, 10))

    nDec = -int(np.floor(np.log10(np.abs(x))))+sigFigs-1
    rounded = np.round(x, nDec)
    if nDec <= 0:
        # format this as an integer
        return nDec, f"{rounded:g}"
    else:
        # format this as a float
        return nDec, f"{rounded:.{nDec}f}"


def roundToDec(x, nDec=2):
    """Round a value to a requested number of decimals.

    Parameters
    ----------
    x : numerical type
        A float or int to be rounded.
    nDec : int
        The number of decimals desired, by default 2.

    Returns
    -------
    output : str
        x formatted as a string with the requested number of decimals.

    Notes
    -----
    History:

    - 2022-08-22, Taylor J Bell
        Imported code written for SPCA, and optimized for Python3.
    """
    if not np.isfinite(nDec):
        return str(x)

    if isinstance(nDec, float):
        nDec = int(np.round(nDec))

    rounded = np.round(x, nDec)
    if nDec <= 0:
        # format this as an integer
        return f"{rounded:g}"
    else:
        # format this as a float
        return f"{rounded:.{nDec}f}"


def transit_latex_table(meta, log):
    """Write a nicely formatted LaTeX table for each plotted value.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The current meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    """
    log.writelog('  Saving results as a LaTeX table')

    data = pd.read_csv(meta.tab_filename_s6, comment='#',
                       delim_whitespace=True)

    # Figure out the number of rows and columns in the table
    nvals = data.shape[0]
    rows = int(np.ceil(nvals/meta.ncols))

    # Figure out the labels for the columns
    if (meta.y_param_basic[:2] == 'rp' and
            meta.y_param_basic[-2:] == '^2'):
        suffix = getPlanetSuffix(meta)+getChannelSuffix(meta)
        colhead = '\\colhead{Transit Depth'+suffix+'}'
    elif meta.y_param_basic in ['rp', 'rprs']:
        suffix = getPlanetSuffix(meta)+getChannelSuffix(meta)
        colhead = '\\colhead{$R_{\\rm p}/R_{\\rm *}$'+suffix+'}'
    elif meta.y_param_basic in ['fp', 'fpfs']:
        suffix = getPlanetSuffix(meta)+getChannelSuffix(meta)
        colhead = '\\colhead{Eclipse Depth'+suffix+'}'
    elif meta.y_param_basic in ['fn']:
        suffix = getPlanetSuffix(meta)+getChannelSuffix(meta)
        colhead = '\\colhead{Nightside Flux'+suffix+'}'
    else:
        colhead = f"\\colhead{{{meta.y_label}}}"

    # Begin the table
    out = "\\begin{deluxetable}{"
    # Center each column
    for i in range(meta.ncols):
        out += "CC|"
    out = out[:-1]+"}\n"
    # Give the table a caption based on the tabulated data
    if meta.y_param_basic[:2] == 'rp':
        out += "\\tablecaption{\\texttt{Eureka!}'s Transit Spectroscopy "
        out += "Results \\label{tab:eureka_transit_spectra}}\n"
    elif meta.y_param_basic in ['fp', 'fpfs']:
        out += "\\tablecaption{\\texttt{Eureka!}'s Eclipse Spectroscopy "
        out += "Results \\label{tab:eureka_eclipse_spectra}}\n"
    # Label each column
    out += "\\tablehead{\n"
    for i in range(meta.ncols):
        out += "\\colhead{Wavelength} & "+colhead+" &"
    out = out[:-1]+" \\\\\n"
    # Provide each column's unit
    for i in range(meta.ncols):
        if meta.x_unit == 'um':
            xunit = '$\\mu$m'
        else:
            xunit = meta.xunit
        if meta.y_label_unit == '':
            y_unit = ''
        else:
            # Trim off the leading space
            y_unit = meta.y_label_unit[1:]
            # Need to make sure to escape % with \ for LaTeX
            if (meta.y_label_unit.count(r'\%') !=
                    meta.y_label_unit.count('%')):
                y_unit = y_unit.replace('%', r'\%')
        out += "\\colhead{("+xunit+")} & \\colhead{"+y_unit+"} &"
    out = out[:-1]+"\n}\n"
    # Begin tabulating the data
    out += "\\startdata\n"
    for i in range(rows):
        for j in range(meta.ncols):
            if j == meta.ncols-1:
                # Last column, add a newline
                end = '\\\\\n'
            else:
                # Not the last column, add an ampersand
                end = ' & '

            if i+rows*j >= nvals:
                # Ran out of values - put blanks
                out += "&"+end
                continue
            line = data.iloc[i+rows*j]

            # Round values to the correct number of significant figures
            val = line[meta.y_param+'_value']*meta.y_scalar
            upper = line[meta.y_param+'_errorpos']
            lower = line[meta.y_param+'_errorneg']
            if not (np.isfinite(upper) or np.isfinite(lower)):
                nDec = 10
            else:
                nDec1, _ = roundToSigFigs(upper*meta.y_scalar)
                nDec2, _ = roundToSigFigs(lower*meta.y_scalar)
                nDec = np.nanmax([nDec1, nDec2])
                if not np.isfinite(nDec):
                    nDec = 10
                else:
                    nDec = int(nDec)
            val = roundToDec(val, nDec)
            upper = roundToDec(upper, nDec)
            lower = roundToDec(lower, nDec)

            # Wavelength
            out += f"{np.round(line['wavelength'], 2):.2f} & "
            # val^{+upper}_{-lower}
            out += f"{val}^{{+{upper}}}_{{-{lower}}}$"
            out += end

    # End the table
    out += "\\enddata\n"
    out += "\\end{deluxetable}"

    # Save the table as a txt file
    meta.tab_filename_s6_latex = meta.tab_filename_s6[:-4]+'_LaTeX.txt'
    with open(meta.tab_filename_s6_latex, 'w') as file:
        file.write(out)

    return


def getPlanetSuffix(meta):
    suffix = ''
    if meta.planetNumber > 0:
        suffix += f', pl{meta.planetNumber}'
    return suffix


def getChannelSuffix(meta):
    suffix = ''
    if meta.channelNumber > 0:
        suffix += f', ch{meta.channelNumber}'
    return suffix
