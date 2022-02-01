#! /usr/bin/env python

# Generic Stage 4 light curve generation pipeline


# Proposed Steps
# -------- -----
# 1.  Read in Stage 3 data products
# 2.  Replace NaNs with zero
# 3.  Determine wavelength bins
# 4.  Increase resolution of spectra (optional)
# 5.  Smooth spectra (optional)
# 6.  Applying 1D drift correction
# 7.  Generate light curves
# 8.  Save Stage 4 data products
# 9.  Produce plots


import sys, os, time, shutil, glob
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from . import plots_s4
from . import drift
from ..lib import sort_nicely as sn
from ..lib import logedit
from ..lib import readECF as rd
from ..lib import manageevent as me
from ..lib import astropytable
from ..lib import util


class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return


def lcJWST(eventlabel, s3_meta=None):
    '''Compute photometric flux over specified range of wavelengths.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
    s3_meta:    MetaClass
        The metadata object from Eureka!'s S3 step (if running S3 and S4 sequentially).

    Returns
    -------
    meta:   MetaClass
        The metadata object with attributes added by S4.

    Notes
    -----
    History:

    - June 2021 Kevin Stevenson
        Initial version
    - October 2021 Taylor Bell
        Updated to allow for inputs from new S3
    '''
    # Initialize a new metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Load Eureka! control file and store values in Event object
    ecffile = 'S4_' + eventlabel + '.ecf'
    ecf = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    #load savefile
    if s3_meta == None:
        # Search for the S2 output metadata in the inputdir provided in
        # First just check the specific inputdir folder
        rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
        if rootdir[-1]!='/':
            rootdir += '/'
        fnames = glob.glob(rootdir+'S3_'+meta.eventlabel+'*_Meta_Save.dat')
        if len(fnames)==0:
            # There were no metadata files in that folder, so let's see if there are in children folders
            fnames = glob.glob(rootdir+'**/S3_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)
            fnames = sn.sort_nicely(fnames)

        if len(fnames)==0:
            # There may be no metafiles in the inputdir - raise an error and give a helpful message
            raise AssertionError('Unable to find an output metadata file from Eureka!\'s S3 step '
                                +'in the inputdir: \n"{}"!'.format(rootdir))
        elif len(fnames)>1:
            # There may be multiple runs - use the most recent but warn the user
            print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                 +'Using the metadata file: \n{}\n'.format(fnames[-1])
                 +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                 +'of the run you intended, please provide a more precise folder for the metadata file.')

        fname = fnames[-1] # Pick the last file name (should be the most recent or only file)
        fname = fname[:-4] # Strip off the .dat ending

        s3_meta = me.loadevent(fname)

    # Need to remove the topdir from the outputdir
    s3_outputdir = s3_meta.outputdir[len(s3_meta.topdir):]
    if s3_outputdir[0]=='/':
        s3_outputdir = s3_outputdir[1:]

    meta = s3_meta

    # Load Eureka! control file and store values in the S3 metadata object
    ecffile = 'S4_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Overwrite the inputdir with the exact output directory from S3
    meta.inputdir = s3_outputdir
    meta.old_datetime = meta.datetime # Capture the date that the
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir

    if not meta.allapers:
        # The user indicated in the ecf that they only want to consider one aperture
        meta.spec_hw_range = [meta.spec_hw,]
        meta.bg_hw_range = [meta.bg_hw,]

    run_i = 0
    for spec_hw_val in meta.spec_hw_range:

        for bg_hw_val in meta.bg_hw_range:

            t0 = time.time()

            meta.spec_hw = spec_hw_val

            meta.bg_hw = bg_hw_val

            # Do some folder swapping to be able to reuse this function to find S3 outputs
            tempfolder = meta.outputdir_raw
            meta.outputdir_raw = meta.inputdir_raw
            meta.inputdir = util.pathdirectory(meta, 'S3', meta.runs[run_i], old_datetime=meta.old_datetime, ap=spec_hw_val, bg=bg_hw_val)
            meta.outputdir_raw = tempfolder
            run_i += 1

            # Create directories for Stage 4 processing outputs
            run = util.makedirectory(meta, 'S4', ap=spec_hw_val, bg=bg_hw_val)
            meta.outputdir = util.pathdirectory(meta, 'S4', run, ap=spec_hw_val, bg=bg_hw_val)

            # Copy existing S3 log file and resume log
            meta.s4_logname  = meta.outputdir + 'S4_' + meta.eventlabel + ".log"
            log         = logedit.Logedit(meta.s4_logname, read=meta.s3_logname)
            log.writelog("\nStarting Stage 4: Generate Light Curves\n")
            log.writelog(f"Input directory: {s3_outputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")

            # Copy ecf (and update outputdir in case S4 is being called sequentially with S3)
            log.writelog('Copying S4 control file')
            # shutil.copy(ecffile, meta.outputdir)
            new_ecfname = meta.outputdir + ecffile.split('/')[-1]
            with open(new_ecfname, 'w') as new_file:
                with open(ecffile, 'r') as file:
                    for line in file.readlines():
                        if len(line.strip())==0 or line.strip()[0]=='#':
                            new_file.write(line)
                        else:
                            line_segs = line.strip().split()
                            if line_segs[0]=='inputdir':
                                new_file.write(line_segs[0]+'\t\t/'+meta.inputdir+'\t'+' '.join(line_segs[2:])+'\n')
                            else:
                                new_file.write(line)

            log.writelog("Loading S3 save file")
            table = astropytable.readtable(meta.tab_filename)

            # Reverse the reshaping which has been done when saving the astropy table
            optspec = np.reshape(table['optspec'].data, (-1, meta.subnx))
            opterr = np.reshape(table['opterr'].data, (-1, meta.subnx))
            wave_1d = table['wave_1d'].data[0:meta.subnx]
            meta.bjdtdb = table['bjdtdb'].data[::meta.subnx]

            #Replace NaNs with zero
            optspec[np.where(np.isnan(optspec))] = 0
            opterr[np.where(np.isnan(opterr))] = 0
            meta.n_int, meta.subnx   = optspec.shape

            # Determine wavelength bins
            binsize     = (meta.wave_max - meta.wave_min)/meta.nspecchan
            meta.wave_low = np.round([i for i in np.linspace(meta.wave_min, meta.wave_max-binsize, meta.nspecchan)],3)
            meta.wave_hi  = np.round([i for i in np.linspace(meta.wave_min+binsize, meta.wave_max, meta.nspecchan)],3)

            # Apply 1D drift/jitter correction
            if meta.correctDrift == True:
                #Calculate drift over all frames and non-destructive reads
                log.writelog('Applying drift/jitter correction')
                # Compute drift/jitter
                meta = drift.spec1D(optspec, meta, log)
                # Correct for drift/jitter
                for n in range(meta.n_int):
                    spline     = spi.UnivariateSpline(np.arange(meta.subnx), optspec[n], k=3, s=0)
                    spline2    = spi.UnivariateSpline(np.arange(meta.subnx), opterr[n],  k=3, s=0)
                    optspec[n] = spline(np.arange(meta.subnx)+meta.drift1d[n])
                    opterr[n]  = spline2(np.arange(meta.subnx)+meta.drift1d[n])
                # Plot Drift
                if meta.isplots_S4 >= 1:
                    plots_s4.drift1d(meta)


            log.writelog("Generating light curves")
            meta.lcdata   = np.zeros((meta.nspecchan, meta.n_int))
            meta.lcerr    = np.zeros((meta.nspecchan, meta.n_int))
            # ev.eventname2 = ev.eventname
            for i in range(meta.nspecchan):
                log.writelog(f"  Bandpass {i} = %.3f - %.3f" % (meta.wave_low[i], meta.wave_hi[i]))
                # Compute valid indeces within wavelength range
                index   = np.where((wave_1d >= meta.wave_low[i])*(wave_1d < meta.wave_hi[i]))[0]
                # Sum flux for each spectroscopic channel
                meta.lcdata[i]    = np.sum(optspec[:,index],axis=1)
                # Add uncertainties in quadrature
                meta.lcerr[i]     = np.sqrt(np.sum(opterr[:,index]**2,axis=1))

                # Plot each spectroscopic light curve
                if meta.isplots_S4 >= 3:
                    plots_s4.binned_lightcurve(meta, meta.bjdtdb, i)

            # Calculate total time
            total = (time.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

            # Save results
            log.writelog('Saving results')
            me.saveevent(meta, meta.outputdir + 'S4_' + meta.eventlabel + "_Meta_Save", save=[])

            # if (isplots >= 1) and (correctDrift == True):
            #     # Plot Drift
            #     # Plots corrected 2D light curves

            log.closelog()

    return meta
