import numpy as np
import glob, os, shutil
import time as time_pkg
from ..lib import manageevent as me
from ..lib import readECF
from ..lib import util, logedit
from ..lib.readEPF import Parameters
from . import lightcurve as lc
from . import models as m

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return

def fitlc(eventlabel, ecf_path='./', s4_meta=None):
    '''Fits 1D spectra with various models and fitters.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
    ecf_path:   str
        The absolute or relative path to where ecfs are stored
    s4_meta:    MetaClass
        The metadata object from Eureka!'s S4 step (if running S4 and S5 sequentially).

    Returns
    -------
    meta:   MetaClass
        The metadata object with attributes added by S5.

    Notes
    -------
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
    '''
    print("\nStarting Stage 5: Light Curve Fitting\n")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S5_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel

    # load savefile
    if s4_meta == None:
        s4_meta = read_s4_meta(meta)

    meta = load_general_s4_meta_info(meta, ecf_path, s4_meta)

    if (not meta.s4_allapers) or (not meta.allapers):
        # The user indicated in the ecf that they only want to consider one aperture
        # in which case the code will consider only the one which made s4_meta.
        # Alternatively, S4 was run without allapers, so S5's allapers will only conside that one
        meta.spec_hw_range = [meta.spec_hw,]
        meta.bg_hw_range = [meta.bg_hw,]

    if meta.testing_S5:
        # Only fit a single channel while testing unless doing a shared fit then do two
        chanrng = 1
    else:
        chanrng = meta.nspecchan

    # Create directories for Stage 5 outputs
    meta.runs_s5 = []
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            run = util.makedirectory(meta, 'S5', ap=spec_hw_val, bg=bg_hw_val)
            meta.runs_s5.append(run)

    run_i = 0
    old_meta = meta
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:

            t0 = time_pkg.time()

            meta = load_specific_s4_meta_info(old_meta, ecf_path, run_i, spec_hw_val, bg_hw_val)

            # Get the directory for Stage 5 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S5', meta.runs_s5[run_i], ap=spec_hw_val, bg=bg_hw_val)
            run_i += 1

            # Copy existing S4 log file and resume log
            meta.s5_logname  = meta.outputdir + 'S5_' + meta.eventlabel + ".log"
            log         = logedit.Logedit(meta.s5_logname, read=meta.s4_logname)
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ecf
            log.writelog('Copying S5 control file', mute=(not meta.verbose))
            meta.copy_ecf()
            # Copy parameter ecf
            log.writelog('Copying S5 parameter control file', mute=(not meta.verbose))
            shutil.copy(os.path.join(ecf_path, meta.fit_par), meta.outputdir)

            # Set the intial fitting parameters
            params = Parameters(ecf_path, meta.fit_par)
            sharedp = False
            for arg, val in params.dict.items():
                if 'shared' in val:
                    sharedp = True
            meta.sharedp = sharedp

            if meta.sharedp and meta.testing_S5:
                chanrng = min([2, meta.nspecchan])

            # Subtract off the user provided time value to avoid floating point precision problems when fitting for values like t0
            offset = params.time_offset.value
            time = meta.time - offset
            if offset!=0:
                time_units = meta.time_units+f' - {offset}'
            else:
                time_units = meta.time_units

            if sharedp:
                #Make a long list of parameters for each channel
                longparamlist, paramtitles = make_longparamlist(meta, params, chanrng)

                log.writelog("\nStarting Shared Fit of {} Channels\n".format(chanrng))

                flux = np.ma.masked_array([])
                flux_err = np.ma.masked_array([])
                for channel in range(chanrng):
                    flux = np.ma.append(flux,meta.lcdata[channel,:] / np.ma.mean(meta.lcdata[channel,:]))
                    flux_err = np.ma.append(flux_err,meta.lcerr[channel,:] / np.ma.mean(meta.lcdata[channel,:]))

                meta = fit_channel(meta,time,flux,0,flux_err,eventlabel,sharedp,params,log,longparamlist,time_units,paramtitles,chanrng)

                # Save results
                log.writelog('Saving results')
                me.saveevent(meta, meta.outputdir + 'S5_' + meta.eventlabel + "_Meta_Save", save=[])
            else:
                for channel in range(chanrng):
                    #Make a long list of parameters for each channel
                    longparamlist, paramtitles = make_longparamlist(meta, params, chanrng)

                    log.writelog("\nStarting Channel {} of {}\n".format(channel+1, chanrng))

                    # Get the flux and error measurements for the current channel
                    flux = meta.lcdata[channel,:]
                    flux_err = meta.lcerr[channel,:]

                    # Normalize flux and uncertainties to avoid large flux values (FINDME: replace when constant offset is implemented)
                    flux_err = flux_err/ flux.mean()
                    flux = flux / flux.mean()

                    meta = fit_channel(meta,time,flux,channel,flux_err,eventlabel,sharedp,params,log,longparamlist,time_units,paramtitles,chanrng)

                    # Save results
                    log.writelog('Saving results', mute=(not meta.verbose))
                    me.saveevent(meta, meta.outputdir + 'S5_' + meta.eventlabel + "_Meta_Save", save=[])

            # Calculate total time
            total = (time_pkg.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            log.closelog()

    return meta

def fit_channel(meta,time,flux,chan,flux_err,eventlabel,sharedp,params,log,longparamlist,time_units,paramtitles,chanrng):
    # Load the relevant values into the LightCurve model object
    lc_model = lc.LightCurve(time, flux, chan, chanrng, log, longparamlist, unc=flux_err, time_units=time_units, name=eventlabel, share=sharedp)

    if hasattr(meta, 'testing_model') and meta.testing_model:
        # FINDME: Use this area to add systematics into the data
        # when testing new systematics models. In this case, I'm
        # introducing an exponential ramp to test m.ExpRampModel().
        log.writelog('****Adding exponential ramp systematic to light curve****')
        fakeramp = m.ExpRampModel(parameters=params, name='ramp', fmt='r--', log=log,
                                  longparamlist=lc_model.longparamlist, nchan=lc_model.nchannel_fitted, paramtitles=paramtitles)
        fakeramp.coeffs = np.array([-1,40,-3, 0, 0, 0]).reshape(1,-1)*np.ones(lc_model.nchannel_fitted)
        flux *= fakeramp.eval(time=time)
        lc_model.flux = flux

    # Make the astrophysical and detector models
    modellist=[]
    if 'batman_tr' in meta.run_myfuncs:
        t_transit = m.BatmanTransitModel(parameters=params, name='transit', fmt='r--', log=log,
                                         longparamlist=lc_model.longparamlist, nchan=lc_model.nchannel_fitted, paramtitles=paramtitles)
        modellist.append(t_transit)
    if 'batman_ecl' in meta.run_myfuncs:
        t_eclipse = m.BatmanEclipseModel(parameters=params, name='eclipse', fmt='r--', log=log,
                                         longparamlist=lc_model.longparamlist, nchan=lc_model.nchannel_fitted, paramtitles=paramtitles)
        modellist.append(t_eclipse)
    if 'sinusoid_pc' in meta.run_myfuncs:
        model_names = np.array([model.name for model in modellist])
        transit_model = None
        eclipse_model = None
        # Nest any transit and/or eclipse models inside of the phase curve model
        if 'transit' in model_names:
            transit_model = modellist.pop(np.where(model_names=='transit')[0][0])
            model_names = np.array([model.name for model in modellist])
        if'eclipse' in model_names:
            eclipse_model = modellist.pop(np.where(model_names=='eclipse')[0][0])
            model_names = np.array([model.name for model in modellist])
        t_phase = m.SinusoidPhaseCurveModel(parameters=params, name='phasecurve', fmt='r--', log=log,
                                            longparamlist=lc_model.longparamlist, nchan=lc_model.nchannel_fitted, paramtitles=paramtitles,
                                            transit_model=transit_model, eclipse_model=eclipse_model)
        modellist.append(t_phase)
    if 'polynomial' in meta.run_myfuncs:
        t_polynom = m.PolynomialModel(parameters=params, name='polynom', fmt='r--', log=log,
                                      longparamlist=lc_model.longparamlist, nchan=lc_model.nchannel_fitted, paramtitles=paramtitles)
        modellist.append(t_polynom)
    if 'expramp' in meta.run_myfuncs:
        t_ramp = m.ExpRampModel(parameters=params, name='ramp', fmt='r--', log=log,
                                longparamlist=lc_model.longparamlist, nchan=lc_model.nchannel_fitted, paramtitles=paramtitles)
        modellist.append(t_ramp)
    if 'GP' in meta.run_myfuncs:
        t_GP = m.GPModel(meta.kernel_class, meta.kernel_inputs, lc_model, parameters=params, name='GP', fmt='r--', log=log)
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
    if meta.sharedp:
        nspecchan = chanrng
    else:
        nspecchan = 1

    longparamlist=[ [] for i in range(nspecchan)]
    tlist=list(params.dict.keys())
    for param in tlist:
        if 'free' in params.dict[param]:
            longparamlist[0].append(param)
            for c in np.arange(nspecchan-1):
                title=param+'_'+str(c+1)
                params.__setattr__(title,params.dict[param])
                longparamlist[c+1].append(title)
        elif 'shared' in params.dict[param]:
            for c in np.arange(nspecchan):
                longparamlist[c].append(param)
        else:
            for c in np.arange(nspecchan):
                longparamlist[c].append(param)
    paramtitles=longparamlist[0]

    return longparamlist, paramtitles

def read_s4_meta(meta):

    # Search for the S2 output metadata in the inputdir provided in
    # First just check the specific inputdir folder
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    fnames = glob.glob(rootdir+'S4_'+meta.eventlabel+'*_Meta_Save.dat')
    if len(fnames)==0:
        # There were no metadata files in that folder, so let's see if there are in children folders
        fnames = glob.glob(rootdir+'**/S4_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)

    if len(fnames)>=1:
        # get the folder with the latest modified time
        fname = max(fnames, key=os.path.getmtime)

    if len(fnames)==0:
        # There may be no metafiles in the inputdir - raise an error and give a helpful message
        raise AssertionError('Unable to find an output metadata file from Eureka!\'s S4 step '
                            +'in the inputdir: \n"{}"!'.format(rootdir))
    elif len(fnames)>1:
        # There may be multiple runs - use the most recent but warn the user
        print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                +'Using the metadata file: \n{}\n'.format(fname)
                +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                +'of the run you intended, please provide a more precise folder for the metadata file.')

    fname = fname[:-4] # Strip off the .dat ending

    s4_meta = me.loadevent(fname)

    # Code to not break backwards compatibility with old MetaClass save files but also use the new MetaClass going forwards
    s4_meta = readECF.MetaClass(**s4_meta.__dict__)

    return s4_meta

def load_general_s4_meta_info(meta, ecf_path, s4_meta):
    # Need to remove the topdir from the outputdir
    s4_outputdir = s4_meta.outputdir[len(meta.topdir):]
    if s4_outputdir[0]=='/':
        s4_outputdir = s4_outputdir[1:]
    if s4_outputdir[-1]!='/':
        s4_outputdir += '/'
    s4_allapers = s4_meta.allapers

    # Overwrite the temporary meta object made above to be able to find s4_meta
    meta = s4_meta

    # Load Eureka! control file and store values in the S4 metadata object
    ecffile = 'S5_' + meta.eventlabel + '.ecf'
    meta.read(ecf_path, ecffile)

    # Overwrite the inputdir with the exact output directory from S4
    meta.inputdir = s4_outputdir
    meta.old_datetime = s4_meta.datetime # Capture the date that the S4 data was made (to figure out it's foldername)
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir
    meta.s4_allapers = s4_allapers

    return meta

def load_specific_s4_meta_info(meta, ecf_path, run_i, spec_hw_val, bg_hw_val):
    # Do some folder swapping to be able to reuse this function to find the correct S4 outputs
    tempfolder = meta.outputdir_raw
    meta.outputdir_raw = '/'.join(meta.inputdir_raw.split('/')[:-2])
    meta.inputdir = util.pathdirectory(meta, 'S4', meta.runs_s4[run_i], old_datetime=meta.old_datetime, ap=spec_hw_val, bg=bg_hw_val)
    meta.outputdir_raw = tempfolder

    # Read in the correct S4 metadata for this aperture pair
    tempfolder = meta.inputdir
    meta.inputdir = meta.inputdir[len(meta.topdir):]
    new_meta = read_s4_meta(meta)
    meta.inputdir = tempfolder

    # Load S5 Eureka! control file and store values in the S4 metadata object
    ecffile = 'S5_' + meta.eventlabel + '.ecf'
    new_meta.read(ecf_path, ecffile)

    # Save correctly identified folders from earlier
    new_meta.inputdir = meta.inputdir
    new_meta.outputdir = meta.outputdir
    new_meta.inputdir_raw = meta.inputdir_raw
    new_meta.outputdir_raw = meta.outputdir_raw

    new_meta.runs_s5 = meta.runs_s5
    new_meta.datetime = meta.datetime

    new_meta.spec_hw = spec_hw_val
    new_meta.bg_hw = bg_hw_val

    return new_meta
