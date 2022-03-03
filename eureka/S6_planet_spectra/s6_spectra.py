import numpy as np
import pandas as pd
from astropy import units, constants
import os, glob
import time as time_pkg
from ..lib import manageevent as me
from ..lib import readECF as rd
from ..lib import util, logedit
from ..lib import sort_nicely as sn

from . import plots_s6 as plots

#FINDME: Keep reload statements for easy testing
from importlib import reload

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return

def plot_spectra(eventlabel, ecf_path='./', s5_meta=None):
    '''Gathers together different wavelength fits and makes transmission/emission spectra.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
    ecf_path:   str
        The absolute or relative path to where ecfs are stored
    s5_meta:    MetaClass
        The metadata object from Eureka!'s S5 step (if running S5 and S6 sequentially).

    Returns
    -------
    meta:   MetaClass
        The metadata object with attributes added by S6.

    Notes
    -------
    History:

    - Feb 14, 2022 Taylor Bell
        Original version
    '''
    print("\nStarting Stage 6: Light Curve Fitting\n")

    # Initialize a new metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Load Eureka! control file and store values in Event object
    ecffile = 'S6_' + eventlabel + '.ecf'
    ecf = rd.read_ecf(ecf_path, ecffile)
    rd.store_ecf(meta, ecf)

    # load savefile
    if s5_meta == None:
        s5_meta = read_s5_meta(meta)

    meta = load_general_s5_meta_info(meta, ecf_path, s5_meta)

    if (not meta.s5_allapers) or (not meta.allapers):
        # The user indicated in the ecf that they only want to consider one aperture
        # in which case the code will consider only the one which made s4_meta.
        # Alternatively, S4 was run without allapers, so S6's allapers will only conside that one
        meta.spec_hw_range = [meta.spec_hw,]
        meta.bg_hw_range = [meta.bg_hw,]

    # Create directories for Stage 6 outputs
    meta.runs_s6 = []
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            run = util.makedirectory(meta, 'S6', ap=spec_hw_val, bg=bg_hw_val)
            meta.runs_s6.append(run)

    run_i = 0
    old_meta = meta
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta = load_specific_s5_meta_info(old_meta, ecf_path, run_i, spec_hw_val, bg_hw_val)

            # Get the directory for Stage 6 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S6', meta.runs_s6[run_i], ap=spec_hw_val, bg=bg_hw_val)
            run_i += 1

            # Copy existing S5 log file and resume log
            meta.s6_logname  = meta.outputdir + 'S6_' + meta.eventlabel + ".log"
            log         = logedit.Logedit(meta.s6_logname, read=meta.s5_logname)
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ecf
            log.writelog('Copying S6 control file')
            rd.copy_ecf(meta, ecf_path, ecffile)

            # Get the wavelength values
            wavelengths = np.mean(np.append(meta.wave_low.reshape(1,-1), meta.wave_hi.reshape(1,-1), axis=0), axis=0)
            wave_errs = (meta.wave_hi-meta.wave_low)/2

            # Convert to the user-provided x-axis unit if needed
            if hasattr(meta, 'x_unit'):
                x_unit = getattr(units, meta.x_unit)
            else:
                log.writelog('Assuming a wavelength unit of microns')
                x_unit = units.um
            # FINDME: For now this is assuming that the data is in units of microns
            # We should add something to S3 that notes what the units of the wavelength were in the FITS file
            wavelengths *= units.um.to(x_unit, equivalencies=units.spectral())
            wave_errs   *= units.um.to(x_unit, equivalencies=units.spectral())
            physical_type = str(x_unit.physical_type).title()
            if physical_type=='Length':
                physical_type = 'Wavelength'
            label_unit = x_unit.name
            if label_unit=='um':
                label_unit = r'$\mu$m'
            xlabel = physical_type+' ('+label_unit+')'

            fit_methods = meta.fit_method.strip('[').strip(']').strip().split(',')

            accepted_y_units = ['Rp/Rs', 'Rp/R*', '(Rp/Rs)^2', '(Rp/R*)^2', 'Fp/Fs', 'Fp/F*']
            if 'rp' in meta.y_unit.lower():
                y_param = 'rp'
            elif 'fp' in meta.y_unit.lower():
                y_param = 'fp'
            else:
                raise AssertionError(f'Unknown y_unit {meta.y_unit} is none of ['+', '.join(accepted_y_units)+']')

            # Read in S5 fitted values
            if meta.sharedp:
                medians, errs = parse_s5_saves(meta, fit_methods, y_param, 'shared')
            else:
                medians = []
                errs = []
                for channel in range(meta.nspecchan):
                    median, err = parse_s5_saves(meta, fit_methods, y_param, f'ch{channel}')
                    medians.append(median[0])
                    errs.append(np.array(err).reshape(-1))
                medians = np.array(medians).reshape(-1)
                errs = np.array(errs).swapaxes(0,1)
                if np.all(errs==None):
                    errs = None

            # Convert the y-axis unit to the user-provided value if needed
            if meta.y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2']:
                if errs is not None:
                    lower = np.abs((medians-errs[0,:])**2-medians**2)
                    upper = np.abs((medians+errs[1,:])**2-medians**2)
                    errs = np.append(lower.reshape(1,-1), upper.reshape(1,-1), axis=0)
                medians *= medians
                ylabel = r'$(R_{\rm p}/R_{\rm *})^2$'
            elif meta.y_unit in ['Rp/Rs', 'Rp/R*']:
                ylabel = r'$R_{\rm p}/R_{\rm *}$'
            elif meta.y_unit in ['Fp/Fs', 'Fp/F*']:
                ylabel = r'$F_{\rm p}/F_{\rm *}$'
            else:
                raise AssertionError(f'Unknown y_unit {meta.y_unit} is none of ['+', '.join(accepted_y_units)+']')

            # Convert to percent, ppm, etc. if requested
            if not hasattr(meta, 'y_scalar'):
                meta.y_scalar = 1

            if meta.y_scalar==1e6:
                ylabel += ' (ppm)'
            elif meta.y_scalar==100:
                ylabel += ' (%)'
            elif meta.y_scalar!=1:
                ylabel += f' * {meta.y_scalar}'

            if meta.model_spectrum is not None:
                model_x, model_y = np.loadtxt(os.path.join(meta.topdir, *meta.model_spectrum.split(os.sep)), delimiter=meta.model_delimiter).T
                # Convert model_x_unit to x_unit if needed
                model_x *= getattr(units, meta.model_x_unit).to(x_unit, equivalencies=units.spectral())
                if meta.model_y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2'] and meta.model_y_unit!=meta.y_unit:
                    model_y = np.sqrt(model_y)
                elif meta.model_y_unit in ['Rp/Rs', 'Rp/R*'] and meta.model_y_unit!=meta.y_unit:
                    model_y *= model_y
                elif meta.model_y_unit not in accepted_y_units:
                    raise AssertionError(f'Unknown model_y_unit {meta.model_y_unit} is none of ['+', '.join(accepted_y_units)+']')
                elif meta.model_y_unit != meta.y_unit:
                    raise AssertionError(f'Unknown conversion between y_unit {meta.y_unit} and model_y_unit {meta.model_y_unit}')

                if not hasattr(meta, 'model_y_scalar'):
                    meta.model_y_scalar = 1

                # Convert the model y-units if needed to match the data y-units requested
                if meta.model_y_scalar!=1:
                    model_y *= meta.model_y_scalar
            else:
                model_x = None
                model_y = None

            # Make the spectrum plot
            if meta.isplots_S6>=1:
                plots.plot_spectrum(meta, wavelengths, medians, errs, wave_errs, model_x, model_y, meta.y_scalar, ylabel, xlabel)

            if meta.isplots_S6>=3 and y_param=='rp' and np.all([hasattr(meta, val) for val in ['planet_Teq', 'planet_mu', 'planet_Rad', 'planet_Mass', 'star_Rad', 'planet_R0']]):
                # Make the spectrum plot
                if meta.planet_Rad is None:
                    meta.planet_Rad = medians
                    if meta.y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2']:
                        meta.planet_Rad = np.sqrt(meta.planet_Rad)
                    meta.planet_Rad = np.mean(meta.planet_Rad)
                    meta.planet_Rad *= (meta.star_Rad*constants.R_sun/constants.R_jup).si.value
                if meta.planet_R0 is not None:
                    meta.planet_R0 *= (constants.R_jup/(meta.star_Rad*constants.R_sun)).si.value
                meta.planet_g = ((constants.G*meta.planet_Mass*constants.M_jup)/(meta.planet_Rad*constants.R_jup)**2).si.value
                log.writelog(f'Calculated g={np.round(meta.planet_g,2)} m/s^2 with Rp={np.round(meta.planet_Rad, 2)} R_jup and Mp={meta.planet_Mass} M_jup')
                scaleHeight = (constants.k_B*(meta.planet_Teq*units.K)/((meta.planet_mu*units.u)*(meta.planet_g*units.m/units.s**2))).si.to('km')
                log.writelog(f'Calculated H={np.round(scaleHeight,2)} with g={np.round(meta.planet_g, 2)} m/s^2, Teq={meta.planet_Teq} K, and mu={meta.planet_mu} u')
                scaleHeight = (scaleHeight/(meta.star_Rad*constants.R_sun)).si.value
                plots.plot_spectrum(meta, wavelengths, medians, errs, wave_errs, model_x, model_y, meta.y_scalar, ylabel, xlabel, scaleHeight, meta.planet_R0)

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            # Save results
            log.writelog('Saving results')
            me.saveevent(meta, meta.outputdir + 'S6_' + meta.eventlabel + "_Meta_Save", save=[])

            log.closelog()

    return meta

def parse_s5_saves(meta, fit_methods, y_param, channel_key='shared'):
    for fitter in fit_methods:
        if fitter in ['dynesty', 'emcee']:
            fname = f'S5_{fitter}_samples_{channel_key}.csv'
            samples = pd.read_csv(meta.inputdir+fname, escapechar='#', skipinitialspace=True)
            if y_param=='fp':
                keys = [key for key in samples.keys() if 'fp' in key]
            else:
                keys = [key for key in samples.keys() if 'rp' in key]
            if len(keys)==0:
                raise AssertionError(f'Parameter {y_param} was not in the list of fitted parameters which includes:'
                                    +', '.join(samples.keys()))
            spectra_samples = np.array([samples[key] for key in keys])
            lowers, medians, uppers = np.percentile(spectra_samples, [16,50,84], axis=1)
            lowers = np.abs(medians-lowers)
            uppers = np.abs(uppers-medians)
            errs = np.array([lowers, uppers])
        else:
            fname = f'S5_{fitter}_fitparams_{channel_key}.csv'
            fitted_values = pd.read_csv(meta.inputdir+fname, escapechar='#', skipinitialspace=True)
            if y_param=='fp':
                keys = [key for key in fitted_values.keys() if 'fp' in key]
            else:
                keys = [key for key in fitted_values.keys() if 'rp' in key]
            if len(keys)==0:
                raise AssertionError(f'Parameter {y_param} was not in the list of fitted parameters which includes:'
                                    +', '.join(samples.keys()))
            medians = np.array([fitted_values[key] for key in keys])
            errs = None

    return medians, errs

def read_s5_meta(meta):

    # Search for the S5 output metadata in the inputdir provided in
    # First just check the specific inputdir folder
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    files = glob.glob(rootdir+'S5_'+meta.eventlabel+'*_Meta_Save.dat')
    if len(files)==0:
        # There were no metadata files in that folder, so let's see if there are in children folders
        files = glob.glob(rootdir+'**/S5_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)
        files = sn.sort_nicely(files)

    if len(files)==0:
        # There may be no metafiles in the inputdir - raise an error and give a helpful message
        raise AssertionError('Unable to find an output metadata file from Eureka!\'s S5 step '
                            +'in the inputdir: \n"{}"!'.format(rootdir))

    elif len(files)>1:
        # There may be multiple runs - use the most recent but warn the user
        print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                +'Using the metadata file: \n{}\n'.format(files[-1])
                +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                +'of the run you intended, please provide a more precise folder for the metadata file.')

    fname = files[-1] # Pick the last file name (should be the most recent or only file)
    fname = fname[:-4] # Strip off the .dat ending

    s5_meta = me.loadevent(fname)

    return s5_meta

def load_general_s5_meta_info(meta, ecf_path, s5_meta):

    # Need to remove the topdir from the outputdir
    s5_outputdir = s5_meta.outputdir[len(s5_meta.topdir):]
    if s5_outputdir[0]=='/':
        s5_outputdir = s5_outputdir[1:]
    if s5_outputdir[-1]!='/':
        s5_outputdir += '/'
    s5_allapers = s5_meta.allapers

    # Overwrite the temporary meta object made above to be able to find s5_meta
    meta = s5_meta

    # Load Eureka! control file and store values in the S4 metadata object
    ecffile = 'S6_' + meta.eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecf_path, ecffile)
    rd.store_ecf(meta, ecf)

    # Overwrite the inputdir with the exact output directory from S5
    meta.inputdir = s5_outputdir
    meta.old_datetime = s5_meta.datetime # Capture the date that the
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir

    meta.s5_allapers = s5_allapers

    return meta

def load_specific_s5_meta_info(meta, ecf_path, run_i, spec_hw_val, bg_hw_val):
    # Do some folder swapping to be able to reuse this function to find the correct S5 outputs
    tempfolder = meta.outputdir_raw
    meta.outputdir_raw = '/'.join(meta.inputdir_raw.split('/')[:-2])
    meta.inputdir = util.pathdirectory(meta, 'S5', meta.runs_s5[run_i], old_datetime=meta.old_datetime, ap=spec_hw_val, bg=bg_hw_val)
    meta.outputdir_raw = tempfolder

    # Read in the correct S5 metadata for this aperture pair
    tempfolder = meta.inputdir
    meta.inputdir = meta.inputdir[len(meta.topdir):]
    new_meta = read_s5_meta(meta)
    meta.inputdir = tempfolder

    # Load S6 Eureka! control file and store values in the S5 metadata object
    ecffile = 'S6_' + meta.eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecf_path, ecffile)
    rd.store_ecf(new_meta, ecf)

    # Save correctly identified folders from earlier
    new_meta.inputdir = meta.inputdir
    new_meta.outputdir = meta.outputdir
    new_meta.inputdir_raw = meta.inputdir_raw
    new_meta.outputdir_raw = meta.outputdir_raw

    new_meta.runs_s6 = meta.runs_s6
    new_meta.datetime = meta.datetime

    return new_meta
