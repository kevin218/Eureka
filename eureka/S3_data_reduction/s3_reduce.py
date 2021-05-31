#! /usr/bin/env python

# Generic Stage 3 reduction pipeline

"""
# Proposed Steps
# -------- -----
# 1.  Read in all data frames and header info from Stage 2 data products
# 2.  Record JD and other relevant header information
# 3.  Apply light-time correction (if necessary)
# 4.  Calculate trace and 1D+2D wavelength solutions (if necessary)
# 5.  Make flats, apply flat field correction (Stage 2)
# 6.  Manually mask regions
# 7.  Compute difference frames OR slopes (Stage 1)
# 8.  Perform outlier rejection of BG region
# 9.  Background subtraction
# 10. Compute 2D drift, apply rough (integer-pixel) correction
# 11. Full-frame outlier rejection for time-series stack of NDRs
# 12. Apply sub-pixel 2D drift correction
# 13. Extract spectrum through summation
# 14. Compute median frame
# 15. Optimal spectral extraction
# 16. Save Stage 3 data products
# 17. Produce plots
"""

import sys, os, time
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka/eureka/S3_data_reduction')
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka/eureka/lib')
sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka/eureka/S3_data_reduction')
sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka/eureka/lib')
from importlib import reload
import numpy as np
import logedit
import readECF as rd
import manageevent as me
#import sort_nicely as sn
import matplotlib.pyplot as plt

import savetable
import barycorr
import optspex

class Event():
  def __init__(self):

    # initialize Univ
    #Univ.__init__(self)
    #self.initpars(ecf)
    #self.foo = 2
    return

def reduceJWST(eventlabel, isplots=False):
    '''
    Reduces data images and calculated optimal spectra.

    Parameters
    ----------
    eventlabel  : str, Unique label for this dataset
    isplots     : boolean, Set True to produce plots

    Returns
    -------
    ev          : Event object

    Remarks
    -------


    History
    -------
    Written by Kevin Stevenson      May 2021

    '''

    t0      = time.time()

    # Initialize event object
    ev              = Event()
    ev.eventlabel   = eventlabel

    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    ev.dirname = 'S3_' + datetime + '_' + ev.eventlabel
    if not os.path.exists(ev.dirname):
        os.makedirs(ev.dirname)
    if not os.path.exists(ev.dirname+"/figs"):
        os.makedirs(ev.dirname+"/figs")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(ev, ecf)

    # Load instrument module
    exec('import ' + ev.inst + ' as inst', globals())
    reload(inst)

    # Open new log file
    ev.logname  = './'+ev.dirname + '/S3_' + ev.eventlabel + ".log"
    log         = logedit.Logedit(ev.logname)
    log.writelog("\nStarting Stage 3 Reduction")

    # Create list of file segments
    ev.segment_list = []
    for fname in os.listdir(ev.topdir + ev.datadir):
        if fname.endswith(ev.suffix + '.fits'):
            ev.segment_list.append(ev.topdir + ev.datadir +'/'+ fname)
    ev.segment_list = ev.segment_list#sn.sort_nicely(ev.segment_list)
    num_data_files = len(ev.segment_list)
    log.writelog(f'\nFound {num_data_files} data file(s) ending in {ev.suffix}.fits')

    ev.stdspec = np.array([])
    ev.stdvar  = np.array([])
    # Loop over each segment
    for m in range(num_data_files):
        # Report progress

        # Read in data frame and header
        log.writelog(f'Reading file {m}')
        data, err, dq, wave, v0, int_times, mhdr, shdr = inst.read(ev.segment_list[m], returnHdr=True)
        # Get number of integrations and frame dimensions
        n_int, ny, nx = data.shape
        intstart = mhdr['INTSTART']
        # Locate science image
        xref_sci = shdr['XREF_SCI']
        yref_sci = shdr['YREF_SCI']
        # Target Coordinates
        ev.ra  = mhdr['TARG_RA'] * np.pi / 180.0  # stores right ascension
        ev.dec = mhdr['TARG_DEC'] * np.pi / 180.0  # stores right ascension
        # Record integration mid-times in BJD_TDB
        mjdutc = int_times['int_mid_MJD_UTC']
        #bjdtdb = int_times['int_mid_BJD_TDB']
        bjdtdb = barycorr.to_bjdtdb(ev, mjdutc, m)
        # Trim data to subarray region of interest
        subdata  = data[:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        suberr   = err [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subdq    = dq  [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subwave  = wave[  ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subv0    = v0  [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subny    = ev.ywindow[1]-ev.ywindow[0]
        subnx    = ev.xwindow[1]-ev.xwindow[0]
        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future to flag certain pixels
        submask = np.ones(subdata.shape)
        # FINDME: Need to convert flux from MJy/sr to electrons
        # FINDME: err and v0 are all NaNs
        if np.sum(np.isnan(suberr)) > 30000:
            log.writelog("ERR array is all NaNs... Assuming ERR = sqrt(DATA), which is very wrong if still in MJy/sr!!!")
            suberr = np.sqrt(np.abs(subdata))
        # Perform outlier rejection of sky background along time axis
        log.writelog('Performing background outlier rejection')
        bg_y1    = int(yref_sci - ev.bg_hw)
        bg_y2    = int(yref_sci + ev.bg_hw)
        submask = inst.flag_bg(subdata, suberr, submask, bg_y1, bg_y2, ev.bg_thresh)
        log.writelog('Performing background subtraction')
        # Write background
        def writeBG(arg):
            bg_data, bg_mask, n = arg
            subbg[n] = bg_data
            submask[n] = bg_mask
            return
        # Compute background for each integration
        subbg  = np.zeros((subdata.shape))
        if ev.ncpu == 1:
            # Only 1 CPU
            for n in range(n_int):
                print('Does background for integration {0}/{1}'.format(n, n_int-1))
                # Fit sky background with out-of-spectra data
                writeBG(inst.fit_bg(subdata[n], submask[n], bg_y1, bg_y2, ev.bg_deg, ev.p3thresh, n, isplots))
        else:
            # Multiple CPUs
            pool = mp.Pool(ev.ncpu)
            for n in range(n_int):
                res = pool.apply_async(inst.fit_bg, args=(subdata[n], submask[n], bg_y1, bg_y2, bg_deg, ev.p3thresh, n, isplots), callback=writeBG)
            pool.close()
            pool.join()
            res.wait()
        # Calculate variance
        # bgerr       = np.std(bg, axis=1)/np.sqrt(np.sum(mask, axis=1))
        # bgerr[np.where(np.isnan(bgerr))] = 0.
        # v0[np.where(np.isnan(v0))] = 0.   # FINDME: v0 is all NaNs
        # v0         += np.mean(bgerr**2)
        # variance    = abs(data) / gain + ev.v0    # FINDME: Gain reference file: 'crds://jwst_nircam_gain_0056.fits'
        #variance    = abs(subdata*submask) / gain + v0

        # Perform background subtraction
        subdata    -= subbg
        # Calulate drift2D
        # print("Calculating 2D drift...")
        # print("Performing rough, pixel-scale drift correction...")
        # Outlier rejection of full frame along time axis
        # print("Performing full-frame outlier rejection...")

        if isplots >= 3:
            for n in range(n_int):
                print('Does plots {0}/{1}'.format(n, n_int - 1))
                plt.figure(3001)
                plt.clf()
                plt.suptitle(str(intstart+n))
                plt.subplot(211)
                plt.imshow(subdata[n]*submask[n], origin='lower', aspect='auto', vmin=0, vmax=10000)
                #plt.imshow(subdata[n], origin='lower', aspect='auto', vmin=0, vmax=10000)
                plt.subplot(212)
                #plt.imshow(submask[i], origin='lower', aspect='auto', vmax=1)
                mean = np.median(subbg[n])
                std  = np.std(subbg[n])
                plt.imshow(subbg[n], origin='lower', aspect='auto',vmin=mean-3*std,vmax=mean+3*std)
                #plt.imshow(submask[n], origin='lower', aspect='auto', vmin=0, vmax=1)
                plt.savefig(ev.dirname+'/figs/fig3001-'+str(intstart+n)+'-Image+Background.png')
                #plt.pause(0.1)
        # print("Performing sub-pixel drift correction...")
        # Extract standard spectrum and its variance
        stdspec     = np.sum(subdata, axis=1)
        stdvar      = np.sum(suberr**2, axis=1)
        # Compute fraction of masked pixels within regular spectral extraction window
        #numpixels   = 2.*ev.spec_width*subnx
        #fracMaskReg = (numpixels - np.sum(apmask,axis=(2,3)))/numpixels

        # Compute median frame
        ev.meddata  = np.median(subdata, axis=0)

        # Extract optimal spectrum with uncertainties
        print("Performing optimal spectral extraction...")

        #see line 817 in /Eureka/eureka/S3_data_reduction/wfc3_1reduce.py

        print(" Done.")

        # Plotting results
        if isplots >= 3:
            for n in range(n_int):
                print('Does plots {0}/{1}'.format(n, n_int - 1))
                plt.figure(3002)
                plt.clf()
                plt.suptitle(str(intstart+n))
                plt.errorbar(range(subnx), stdspec[n], yerr=np.sqrt(stdvar[n]), fmt='-', color='C1', ecolor='C0', label='Std Spec')
                #plt.errorbar(range(subnx), spectra[n], specerr[n], fmt='-', color='C2', ecolor='C4', label='Optimal Spec')
                plt.legend(loc='best')
                plt.savefig(ev.dirname+'/figs/fig3002-'+str(intstart+n)+'-Spectrum.png')
                #plt.pause(0.1)
        # Append results
        if len(ev.stdspec) == 0:
            ev.stdspec = stdspec
            ev.stdvar  = stdvar
            ev.shapes = np.array([stdspec.shape])
            ev.wave = np.array([subwave[0]])
            ev.mjdutc = np.array([mjdutc])
            ev.bjdtdb = np.array([bjdtdb])
        else:
            ev.stdspec = np.append(ev.stdspec, stdspec, axis=0)
            ev.stdvar  = np.append(ev.stdvar, stdvar, axis=0)
            ev.shapes  = np.append(ev.shapes, np.array([stdspec.shape]), axis=0)
            ev.wave = np.append(ev.wave, np.array([subwave[0]]), axis=0)
            ev.mjdutc = np.append(ev.mjdutc, np.array([mjdutc]), axis=0)
            ev.bjdtdb = np.append(ev.bjdtdb, np.array([bjdtdb]), axis=0)


    # Calculate total time
    total = (time.time() - t0)/60.
    log.writelog('\nTotal time (min): ' + str(np.round(total,2)))


    # Save results
    log.writelog('Pickle-Save the event...')
    me.saveevent(ev, ev.dirname + '/S3_' + ev.eventlabel + "_Save", save=[])

    log.writelog('Saving results as astropy table...')
    savetable.savetable(ev)

    log.closelog()
    return ev
