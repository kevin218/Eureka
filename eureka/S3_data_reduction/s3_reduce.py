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
import numpy as np
import matplotlib.pyplot as plt
from ..lib import logedit
from ..lib import readECF as rd
from ..lib import manageevent as me
from ..lib import sort_nicely as sn
from . import bright2flux as b2f
from . import optspex
from importlib import reload
from ..lib import savetable
from ..lib import barycorr
reload(optspex)
reload(b2f)

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
    ev.workdir = 'S3_' + datetime + '_' + ev.eventlabel
    if not os.path.exists(ev.workdir):
        os.makedirs(ev.workdir)
    if not os.path.exists(ev.workdir+"/figs"):
        os.makedirs(ev.workdir+"/figs")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(ev, ecf)

    # Load instrument module
    exec('from . import ' + ev.inst + ' as inst', globals())
    reload(inst)

    # Open new log file
    ev.logname  = './'+ev.workdir + '/S3_' + ev.eventlabel + ".log"
    log         = logedit.Logedit(ev.logname)
    log.writelog("\nStarting Stage 3 Reduction")

    # Create list of file segments
    ev.segment_list = []
    for fname in os.listdir(ev.topdir + ev.datadir):
        if fname.endswith(ev.suffix + '.fits'):
            ev.segment_list.append(ev.topdir + ev.datadir +'/'+ fname)
    ev.segment_list = sn.sort_nicely(ev.segment_list)
    num_data_files = len(ev.segment_list)
    log.writelog(f'\nFound {num_data_files} data file(s) ending in {ev.suffix}.fits')

    ev.stdspec = np.array([])
    # Loop over each segment
    for m in range(num_data_files):
        # Report progress

        # Read in data frame and header
        log.writelog(f'Reading file {m+1} of {num_data_files}')
        data, err, dq, wave, v0, int_times, mhdr, shdr = inst.read(ev.segment_list[m], returnHdr=True)
        # Get number of integrations and frame dimensions
        n_int, ny, nx = data.shape
        intstart = mhdr['INTSTART']
        # Locate source postion
        src_xpos = shdr['SRCXPOS']-ev.xwindow[0]
        src_ypos = shdr['SRCYPOS']-ev.ywindow[0]
        # Target Coordinates
        ev.ra  = mhdr['TARG_RA'] * np.pi / 180.0  # stores right ascension
        ev.dec = mhdr['TARG_DEC'] * np.pi / 180.0  # stores right ascension
        # Record integration mid-times in BJD_TDB
        mjdutc = int_times['int_mid_MJD_UTC']
        #bjdtdb = int_times['int_mid_BJD_TDB']
        bjdtdb = barycorr.to_bjdtdb(ev, mjdutc, m)
        # Trim data to subarray region of interest
        subdata     = data    [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        suberr      = err     [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subdq       = dq      [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subwave     = wave    [  ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subv0       = v0      [:,ev.ywindow[0]:ev.ywindow[1],ev.xwindow[0]:ev.xwindow[1]]
        subny       = ev.ywindow[1]-ev.ywindow[0]
        subnx       = ev.xwindow[1]-ev.xwindow[0]
        # Create bad pixel mask (1 = good, 0 = bad)
        # FINDME: Will want to use DQ array in the future to flag certain pixels
        submask = np.ones(subdata.shape)
        if shdr['BUNIT'] == 'MJy/sr':
            # Convert from brightness units (MJy/sr) to flux units (uJy/pix)
            #log.writelog('Converting from brightness to flux units')
            #subdata, suberr, subv0 = b2f.bright2flux(subdata, suberr, subv0, shdr['PIXAR_A2'])
            # Convert from brightness units (MJy/sr) to DNs
            log.writelog('  Converting from brightness units (MJy/sr) to DNs')
            photfile = ev.topdir + ev.ancildir +'/'+ mhdr['R_PHOTOM'][7:]
            subdata, suberr, subv0 = b2f.bright2dn(subdata, suberr, subv0, subwave, photfile, mhdr, shdr)

        # Check if arrays have NaNs
        if np.sum(np.isnan(subdata)) > 0:
            log.writelog("  WARNING: DATA array has NaNs.  Your subregion is probably off the edge of the detector subarray. Masking NaN region and continuing, but you should probably stop and reconsider your choices.")
            inan = np.where(np.isnan(subdata))
            #subdata[inan]  = 0
            submask[inan]  = 0
        if np.sum(np.isnan(suberr)) > 0:
            log.writelog("  WARNING: ERR array has NaNs. Your subregion is probably off the edge of the detector subarray. Masking NaN region and continuing, but you should probably stop and reconsider your choices.")
            inan = np.where(np.isnan(suberr))
            #suberr[inan]  = np.sqrt(np.abs(subdata[inan]))
            submask[inan]  = 0
        if np.sum(np.isnan(subv0)) > 0:
            log.writelog("  WARNING: v0 array has NaNs. Your subregion is probably off the edge of the detector subarray. Masking NaN region and continuing, but you should probably stop and reconsider your choices.")
            inan = np.where(np.isnan(subv0))
            #subv0[inan]   = 0
            submask[inan]  = 0

        # Manually mask regions [colstart, colend, rowstart, rowend]
        if hasattr(ev, 'manmask'):
            log.writelog("  Masking manually identified bad pixels")
            for i in range(len(ev.manmask)):
                ind, colstart, colend, rowstart, rowend = ev.manmask[i]
                submask[rowstart:rowend,colstart:colend] = 0

        # Perform outlier rejection of sky background along time axis
        log.writelog('  Performing background outlier rejection')
        bg_y1    = int(src_ypos - ev.bg_hw)
        bg_y2    = int(src_ypos + ev.bg_hw)
        submask = inst.flag_bg(subdata, suberr, submask, bg_y1, bg_y2, ev.bg_thresh)

        # Write background
        def writeBG(arg):
            bg_data, bg_mask, n = arg
            subbg[n] = bg_data
            submask[n] = bg_mask
            return
        # Compute background for each integration
        log.writelog('  Performing background subtraction')
        subbg  = np.zeros((subdata.shape))
        if ev.ncpu == 1:
            # Only 1 CPU
            for n in range(n_int):
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
                plt.figure(3301)
                plt.clf()
                plt.suptitle(str(intstart+n))
                plt.subplot(211)
                max = np.max(subdata[n]*submask[n])
                plt.imshow(subdata[n]*submask[n], origin='lower', aspect='auto', vmin=0, vmax=max/10)
                #plt.imshow(subdata[n], origin='lower', aspect='auto', vmin=0, vmax=10000)
                plt.subplot(212)
                #plt.imshow(submask[i], origin='lower', aspect='auto', vmax=1)
                median = np.median(subbg[n])
                std  = np.std(subbg[n])
                plt.imshow(subbg[n], origin='lower', aspect='auto',vmin=median-3*std,vmax=median+3*std)
                #plt.imshow(submask[n], origin='lower', aspect='auto', vmin=0, vmax=1)
                plt.savefig(ev.workdir+'/figs/fig3301-'+str(intstart+n)+'-Image+Background.png')
                #plt.pause(0.1)
        # print("Performing sub-pixel drift correction...")

        # Select only aperture region
        ap_y1       = int(src_ypos - ev.spec_hw)
        ap_y2       = int(src_ypos + ev.spec_hw)
        apdata      = subdata[:,ap_y1:ap_y2]
        aperr       = suberr [:,ap_y1:ap_y2]
        apmask      = submask[:,ap_y1:ap_y2]
        apbg        = subbg  [:,ap_y1:ap_y2]
        apv0        = subv0  [:,ap_y1:ap_y2]
        # Extract standard spectrum and its variance
        stdspec     = np.sum(apdata, axis=1)
        stdvar      = np.sum(aperr**2, axis=1)  #FINDME: stdvar >> stdspec, which is a problem
        # Compute fraction of masked pixels within regular spectral extraction window
        #numpixels   = 2.*ev.spec_width*subnx
        #fracMaskReg = (numpixels - np.sum(apmask,axis=(2,3)))/numpixels

        # Compute median frame
        ev.medsubdata   = np.median(subdata, axis=0)
        ev.medapdata    = np.median(apdata, axis=0)

        # Extract optimal spectrum with uncertainties
        log.writelog("  Performing optimal spectral extraction")
        optspec     = np.zeros((stdspec.shape))
        opterr      = np.zeros((stdspec.shape))
        gain        = 1         #FINDME: need to determine correct gain
        for n in range(n_int):
            optspec[n], opterr[n], mask = optspex.optimize(apdata[n], apmask[n], apbg[n], stdspec[n], gain, apv0[n], p5thresh=ev.p5thresh, p7thresh=ev.p7thresh, fittype=ev.fittype, window_len=ev.window_len, deg=ev.prof_deg, n=intstart+n, isplots=isplots, eventdir=ev.workdir, meddata=ev.medapdata)

        # Plotting results
        if isplots >= 3:
            for n in range(n_int):
                plt.figure(3302)
                plt.clf()
                plt.suptitle(str(intstart+n))
                plt.plot(range(subnx), stdspec[n], '-', color='C1', label='Std Spec')
                #plt.errorbar(range(subnx), stdspec[n], yerr=np.sqrt(stdvar[n]), fmt='-', color='C1', ecolor='C0', label='Std Spec')
                plt.errorbar(range(subnx), optspec[n], opterr[n], fmt='-', color='C2', ecolor='C2', label='Optimal Spec')
                plt.legend(loc='best')
                plt.savefig(ev.workdir+'/figs/fig3302-'+str(intstart+n)+'-Spectrum.png')
                #plt.pause(0.1)
        # Append results
        if len(ev.stdspec) == 0:
            ev.wave     = subwave
            ev.stdspec  = stdspec
            ev.stdvar   = stdvar
            ev.optspec  = optspec
            ev.opterr   = opterr
            ev.mjdutc   = mjdutc
            ev.bjdtdb   = bjdtdb
        else:
            ev.stdspec  = np.append(ev.stdspec, stdspec, axis=0)
            ev.stdvar   = np.append(ev.stdvar, stdvar, axis=0)
            ev.optspec  = np.append(ev.optspec, optspec, axis=0)
            ev.opterr   = np.append(ev.opterr, opterr, axis=0)
            ev.mjdutc   = np.append(ev.mjdutc, mjdutc, axis=0)
            ev.bjdtdb   = np.append(ev.bjdtdb, bjdtdb, axis=0)

    # Calculate total time
    total = (time.time() - t0)/60.
    log.writelog('\nTotal time (min): ' + str(np.round(total,2)))


    # Save results
    log.writelog('Saving results')
    me.saveevent(ev, ev.workdir + '/S3_' + ev.eventlabel + "_Save", save=[])

    log.writelog('Saving results as astropy table...')
    savetable.savetable(ev)

    log.writelog('Generating figures')
    if isplots >= 1:
        # 2D light curve without drift correction
        plt.figure(3101, figsize=(8,8))  #ev.n_files/20.+0.8))
        plt.clf()
        wmin        = ev.wave.min()
        wmax        = ev.wave.max()
        n_int, nx   = ev.optspec.shape
        #iwmin       = np.where(ev.wave[src_ypos]>wmin)[0][0]
        #iwmax       = np.where(ev.wave[src_ypos]>wmax)[0][0]
        vmin        = 0.97
        vmax        = 1.03
        #normspec    = np.mean(ev.optspec,axis=1)/np.mean(ev.optspec[ev.inormspec[0]:ev.inormspec[1]],axis=(0,1))
        normspec    = ev.optspec/np.mean(ev.optspec,axis=0)
        plt.imshow(normspec,origin='lower', aspect='auto',extent=[wmin,wmax,0,n_int],vmin=vmin,vmax=vmax,cmap=plt.cm.RdYlBu_r)
        ediff       = np.zeros(n_int)
        for m in range(n_int):
            ediff[m]    = 1e6*np.median(np.abs(np.ediff1d(normspec[m])))
            # plt.scatter(ev.wave[src_ypos], np.zeros(nx)+m, c=normspec[m],
            #             s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
        plt.title("MAD = "+str(np.round(np.mean(ediff),0).astype(int)) + " ppm")
        #plt.xlim(wmin,wmax)
        #plt.ylim(0,n_int)
        plt.ylabel('Integration Number')
        plt.xlabel('Wavelength ($\mu m$)')
        plt.colorbar(label='Normalized Flux')
        plt.tight_layout()
        plt.savefig(ev.workdir+'/figs/fig3101-2D_LC.png')

    log.closelog()
    return ev
