#! /usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.interpolate as spi
from astropy.io import fits as pf
import matplotlib.pyplot as plt
import multiprocessing as mp
from ..lib import manageevent as me
from ..lib import sort_nicely as sn
from ..lib import centroid, suntimecorr, utc_tt
import time, os, sys, shutil
#import hst_scan as hst
from importlib import reload
#reload(hst)

def reduceWFC3(eventname, eventdir, madVariable=False, madVarSet=False, isplots=False):
    '''
    Reduces data images and calculated optimal spectra.

    Parameters
    ----------
    isplots     : Set True to produce plots

    Returns
    -------
    None

    Remarks
    -------
    Requires eventname_params file to intialize event object

    Steps
    -----
    1.  Read in all data frames and header info
    2.  Record JD, scan direction, etc
    3.  Group images by frame, batch, and orbit number
    4.  Calculate centroid of direct image(s)
    5.  Calculate trace and 1D+2D wavelength solutions
    6.  Make flats, apply flat field correction
    7.  Manually mask regions
    8.  Apply light-time correction
    9.  Compute difference frames
    10. Compute scan length
    11. Perform outlier rejection of BG region
    12. Background subtraction
    13. Compute 2D drift, apply rough (integer-pixel) correction
    14. Full-frame outlier rejection for time-series stack of NDRs
    15. Apply sub-pixel 2D drift correction
    16. Extract spectrum through summation
    17. Compute median frame
    18. Optimal spectral extraction
    19. Save results, plot figures

    History
    -------
    Written by Kevin Stevenson      January 2017

    '''
    evpname = eventname + '_params'
    #exec 'import ' + evpname + ' as evp' in locals()
    #exec('import ' + evpname + ' as evp', locals())
    exec('import ' + evpname + ' as evp', globals())
    reload(evp)

    t0 = time.time()
    # Initialize event object
    # All parameters are specified in this file
    ev = evp.event_init()
    try:
        aux = evp.aux_init()
    except:
        print("Need to update event file to include auxiliary object.")
        return

    ev.eventdir = eventdir

    # Create directories
    if not os.path.exists(ev.eventdir):
        os.makedirs(ev.eventdir)
    if not os.path.exists(ev.eventdir+"/figs"):
        os.makedirs(ev.eventdir+"/figs")

    # Copy ev_params file
    shutil.copyfile(evpname + '.py', ev.eventdir+'/'+evpname+'.py')

    # Reset attribute for MAD variable (added by K. Showalter)
    if madVariable:
        setattr(ev,madVariable,madVarSet)
        ev.madVarStr = madVariable
        ev.madVariable = madVarSet

    # Object
    ev.obj_list = []   #Do not rename ev.obj_list!
    if ev.objfile == None:
        #Retrieve files within specified range
        for i in range(ev.objstart,ev.objend):
            ev.obj_list.append(ev.loc_sci + ev.filebase + str(i).zfill(4) + ".fits")
    elif ev.objfile == 'all':
        #Retrieve all files from science directory
        for fname in os.listdir(ev.loc_sci):
            ev.obj_list.append(ev.loc_sci +'/'+ fname)
        ev.obj_list = sn.sort_nicely(ev.obj_list)
    else:
        #Retrieve filenames from list
        files = np.genfromtxt(ev.objfile, dtype=str, comments='#')
        for fname in files:
            ev.obj_list.append(ev.loc_sci +'/'+ fname)
        # handle = open(ev.objfile)
        # for line in handle:
        #     print(line)
        #     ev.obj_list.append(ev.loc_sci + line)
        # handle.close()
    ev.n_files = len(ev.obj_list)

    #Determine image size and filter/grism
    hdulist         = pf.open(ev.obj_list[0].rstrip())
    nx              = hdulist['SCI',1].header['NAXIS1']
    ny              = hdulist['SCI',1].header['NAXIS2']
    ev.grism        = hdulist[0].header['FILTER']
    ev.detector     = hdulist[0].header['DETECTOR']
    ev.flatoffset   = [[-1*hdulist['SCI',1].header['LTV2'], -1*hdulist['SCI',1].header['LTV1']]]
    n_reads         = hdulist['SCI',1].header['SAMPNUM']
    hdulist.close()

    # Record JD and exposure times
    print('Reading data & headers, recording JD and exposure times...')
    ywindow     = ev.ywindow[0]
    xwindow     = ev.xwindow[0]
    subny       = ywindow[1] - ywindow[0]
    subnx       = xwindow[1] - xwindow[0]
    subdata     = np.zeros((ev.n_files,n_reads,subny,subnx))
    suberr      = np.zeros((ev.n_files,n_reads,subny,subnx))
    data_mhdr   = []
    data_hdr    = []
    ev.jd       = np.zeros(ev.n_files)
    ev.exptime  = np.zeros(ev.n_files)
    for m in range(ev.n_files):
        data, err, hdr, mhdr = hst.read(ev.obj_list[m].rstrip())
        subdata[m]  = data[0,:,ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
        suberr [m]  = err [0,:,ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
        data_mhdr.append(mhdr[0])
        data_hdr.append(hdr[0])
        ev.jd[m]       = 2400000.5 + 0.5*(data_mhdr[m]['EXPSTART'] + data_mhdr[m]['EXPEND'])
        ev.exptime[m]  = data_mhdr[m]['EXPTIME']

    # Assign scan direction
    ev.scandir = np.zeros(ev.n_files, dtype=int)
    ev.n_scan0 = 0
    ev.n_scan1 = 0
    try:
        scan0 = data_mhdr[0]['POSTARG2']
        scan1 = data_mhdr[1]['POSTARG2']
        for m in range(ev.n_files):
            if data_mhdr[m]['POSTARG2'] == scan0:
                ev.n_scan0 += 1
            elif data_mhdr[m]['POSTARG2'] == scan1:
                ev.scandir[m] = 1
                ev.n_scan1 += 1
            else:
                print('WARNING: Unknown scan direction for file ' + str(m) + '.')
        print("# of files in scan direction 0: " + str(ev.n_scan0))
        print("# of files in scan direction 1: " + str(ev.n_scan1))
    except:
        ev.n_scan0 = ev.n_files
        print("Unable to determine scan direction, assuming unidirectional.")

    # Group frames into frame, batch, and orbit number
    ev.framenum, ev.batchnum, ev.orbitnum = hst.groupFrames(ev.jd)

    # Determine read noise and gain
    ev.readNoise = np.mean((data_mhdr[0]['READNSEA'],
                            data_mhdr[0]['READNSEB'],
                            data_mhdr[0]['READNSEC'],
                            data_mhdr[0]['READNSED']))
    print('Read noise: ' + str(ev.readNoise))
    print('Gain: ' + str(ev.gain))
    #ev.v0 = (ev.readNoise/ev.gain)**2     #Units of ADU
    ev.v0 = ev.readNoise**2                #Units of electrons

    # Calculate centroid of direct image(s)
    ev.img_list = []
    if isinstance(ev.directfile, str) and ev.directfile.endswith('.fits'):
        ev.img_list.append(ev.loc_cal + ev.directfile)
    else:
        #Retrieve filenames from list
        handle = open(ev.directfile)
        for line in handle:
            ev.img_list.append(ev.loc_cal + line)
        handle.close()
    ev.n_img = len(ev.img_list)
    ev.centroid, ev.directim = hst.imageCentroid(ev.img_list, ev.centroidguess, ev.centroidtrim, ny, ev.obj_list[0])

    """
    # Calculate theoretical centroids along spatial scan direction
    ev.centroids = []
    for j in range(ev.n_img):
        ev.centroids.append([])
        for i in range(ev.n_spec):
            # Can assume that scan direction is only in y direction (no x component)
            # because we will apply drift correction to make it so
            ev.centroids[j].append([np.zeros(subny)+ev.centroid[j][0],ev.centroid[j][1]])

    # Calculate trace
    print("Calculating 2D trace and wavelength assuming " + ev.grism + " filter/grism...")
    ev.xrange   = []
    for i in range(ev.n_spec):
        ev.xrange.append(np.arange(ev.xwindow[i][0],ev.xwindow[i][1]))
    ev.trace2d  = []
    ev.wave2d   = []
    for j in range(ev.n_img):
        ev.trace2d.append([])
        ev.wave2d.append([])
        for i in range(ev.n_spec):
            ev.trace2d[j].append(hst.calcTrace(ev.xrange[i], ev.centroids[j][i], ev.grism))
            ev.wave2d[j].append(hst.calibrateLambda(ev.xrange[i], ev.centroids[j][i], ev.grism)/1e4)     #wavelength in microns

    if ev.detector == 'IR':
        print("Calculating slit shift values using last frame...")
        i = 0   #Use first spectrum
        j = -1  #Use last image
        #spectrum    = subdata[j]
        spectrum    = pf.getdata(ev.obj_list[j])
        ev.slitshift, ev.shift_values, ev.yfit = hst.calc_slitshift2(spectrum, ev.xrange[i], ev.ywindow[i], ev.xwindow[i])
        ev.wavegrid  = ev.wave2d
        ev.wave = []
        for j in range(ev.n_img):
            ev.wave.append([])
            for i in range(ev.n_spec):
                ev.wave[j].append(np.mean(ev.wavegrid[j][i],axis=0))
    else:
        # Assume no slitshift for UVIS
        ev.yfit         = range(ev.ywindow[0][1] - ev.ywindow[0][0])
        ev.slitshift    = np.zeros(ev.ywindow[0][1] - ev.ywindow[0][0])
        ev.shift_values = np.zeros(len(ev.yfit))

    # Make list of master flat field frames
    subflat     = np.ones((ev.n_img,ev.n_spec,subny,subnx))
    flatmask    = np.ones((ev.n_img,ev.n_spec,subny,subnx))
    if ev.flatfile == None:
        print('No flat frames found.')
        flat_hdr    = None
        flat_mhdr   = None
    else:
        print('Loading flat frames...')
        for j in range(ev.n_img):
            tempflat, tempmask = hst.makeflats(ev.flatfile, ev.wavegrid[j], ev.xwindow, ev.ywindow, ev.flatoffset, ev.n_spec, ny, nx, sigma=ev.flatsigma)
            for i in range(ev.n_spec):
                subflat[j][i]   = tempflat[i][ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
                flatmask[j][i]  = tempmask[i][ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]

    # Manually mask regions [specnum, colstart, colend]
    if hasattr(ev, 'manmask'):
        print("\rMasking manually identified bad pixels.")
        for j in range(ev.n_img):
            for i in range(len(ev.manmask)):
                ind, colstart, colend, rowstart, rowend = ev.manmask[i]
                n = ind % ev.n_spec
                flatmask[j][n][rowstart:rowend,colstart:colend] = 0 #ev.window[:,ind][0]:ev.window[:,ind][1]

    # Calculate reduced image
    for m in range(ev.n_files):
        #Select appropriate flat, mask, and slitshift
        if ev.n_img == (np.max(ev.orbitnum)+1):
            j = int(ev.orbitnum[m])
        else:
            j = 0
        for n in range(n_reads):
            subdata[m][n] /= subflat[j][0]

    """
    # Read in drift2D from previous iteration
    # np.save("drift2D.npy",ev.drift2D)
    #try:
    #    drift2D = np.load("drift2D.npy")
    #except:
    #    print("drift2D.npy not found.")
    drift2D = np.zeros((ev.n_files,n_reads-1,2))
    # Calculate centroids for each grism frame
    ev.centroids = np.zeros((ev.n_files,n_reads-1,2))
    for m in range(ev.n_files):
        for n in range(n_reads-1):
            ev.centroids[m,n] = np.array([ev.centroid[0][0]+drift2D[m,n,0],
                                          ev.centroid[0][1]+drift2D[m,n,1]])
            #ev.centroids[m,n] = np.array([np.zeros(subny)+ev.centroid[0][0]+drift2D[m,n,0],
            #                              np.zeros(subnx)+ev.centroid[0][1]+drift2D[m,n,1]])

    # Calculate trace
    print("Calculating 2D trace and wavelength assuming " + ev.grism + " filter/grism...")
    ev.xrange   = np.arange(ev.xwindow[0][0],ev.xwindow[0][1])
    trace2d  = np.zeros((ev.n_files,n_reads-1,subny,subnx))
    wave2d   = np.zeros((ev.n_files,n_reads-1,subny,subnx))
    for m in range(ev.n_files):
        for n in range(n_reads-1):
            trace2d[m,n] = hst.calcTrace(ev.xrange, ev.centroids[m,n], ev.grism)
            wave2d[m,n]  = hst.calibrateLambda(ev.xrange, ev.centroids[m,n], ev.grism)/1e4   #wavelength in microns

    # Assume no slitshift
    ev.yfit         = range(ev.ywindow[0][1] - ev.ywindow[0][0])
    ev.slitshift    = np.zeros(ev.ywindow[0][1] - ev.ywindow[0][0])
    ev.shift_values = np.zeros(len(ev.yfit))
    ev.wave         = np.mean(wave2d, axis=2)
    print("Wavelength Range: %.3f - %.3f" % (np.min(ev.wave), np.max(ev.wave)))
    #iwmax       = np.where(ev.wave[0][0]>1.65)[0][0]
    #print(ev.wave[0,0])
    #print(ev.wave[0,1])
    #print(ev.centroids)

    # Make list of master flat field frames
    subflat     = np.ones((ev.n_files,subny,subnx))
    flatmask    = np.ones((ev.n_files,subny,subnx))
    if ev.flatfile == None:
        print('No flat frames found.')
        flat_hdr    = None
        flat_mhdr   = None
    else:
        print('Loading flat frames...')
        print(ev.flatfile)
        for m in range(ev.n_files):
            tempflat, tempmask = hst.makeflats(ev.flatfile, [np.mean(wave2d[m],axis=0)], ev.xwindow, ev.ywindow, ev.flatoffset, ev.n_spec, ny, nx, sigma=ev.flatsigma)
            #tempflat    = [pf.getdata(ev.flatfile)]
            #tempmask    = [np.ones(tempflat[0].shape)]
            subflat[m]  = tempflat[0][ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
            flatmask[m] = tempmask[0][ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]

    # Manually mask regions [specnum, colstart, colend]
    if hasattr(ev, 'manmask'):
        print("\rMasking manually identified bad pixels.")
        for m in range(ev.n_files):
            for i in range(len(ev.manmask)):
                ind, colstart, colend, rowstart, rowend = ev.manmask[i]
                flatmask[m][rowstart:rowend,colstart:colend] = 0

    #FINDME: Change flat field
    #subflat[:,:,28] /= 1.015
    #subflat[:,:,50] /= 1.015
    #subflat[:,:,70] *= 1.01
    """
    plt.figure(2)
    plt.clf()
    plt.imshow(np.copy(subdata[10,-1]),origin='lower',aspect='auto',
                vmin=0,vmax=25000,cmap=plt.cm.RdYlBu_r)
    plt.ylim(65,95)
    plt.show()
    """
    # Calculate reduced image
    subdata /= subflat[:,np.newaxis]
    #subdata /= np.mean(subflat,axis=0)[np.newaxis,np.newaxis]
    """
    # FINDME
    # Perform self flat field calibration
    # drift2D_int  = np.round(edrift2D,0)
    # Identify frames outside SAA
    iNoSAA      = np.where(np.round(drift2D[:,0,0],0)==0)[0]
    # Select subregion with lots of photons
    normdata    = np.copy(subdata[iNoSAA,-1,69:91,15:147])
    normmask    = flatmask[iNoSAA,69:91,15:147]
    normdata[np.where(normmask==0)] = 0
    # Normalize flux in each row to remove ramp/transit/variable scan rate
    normdata   /= np.sum(normdata,axis=2)[:,:,np.newaxis]
    # Divide by mean spectrum to remove wavelength dependence
    normdata   /= np.mean(normdata,axis=(0,1))[np.newaxis,np.newaxis,:]
    # Average frames to get flat-field correction
    flat_norm   = np.mean(normdata,axis=0)
    flat_norm[np.where(np.mean(normmask,axis=0)<1)] = 1
    '''
    normdata   /= np.mean(normdata,axis=(1,2))[:,np.newaxis,np.newaxis]
    flat_window = np.median(normdata,axis=0)
    medflat     = np.median(flat_window, axis=0)
    flat_window /= medflat
    flat_window /= np.median(flat_window,axis=1)[:,np.newaxis]
    flat_norm   = flat_window/np.mean(flat_window)
    '''
    plt.figure(3)
    plt.clf()
    plt.imshow(np.copy(subdata[10,-1]),origin='lower',aspect='auto',
                vmin=0,vmax=25000,cmap=plt.cm.RdYlBu_r)
    plt.ylim(65,95)

    ff      = np.load('ff.npy')
    subff   = ff[ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]

    #subdata[:,:,69:91,15:147] /= flat_norm
    subdata /= subff

    plt.figure(4)
    plt.clf()
    plt.imshow(subdata[10,-1],origin='lower',aspect='auto',vmin=0,vmax=25000,cmap=plt.cm.RdYlBu_r)
    plt.ylim(65,95)

    plt.figure(1)
    plt.clf()
    plt.imshow(flat_norm,origin='lower',aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.pause(0.1)

    ev.flat_norm = flat_norm
    return ev
    """
    """
    if isplots:
        # Plot normalized flat fields
        plt.figure(1000, figsize=(12,8))
        plt.clf()
        plt.suptitle('Master Flat Frames')
        for i in range(ev.n_spec):
            for j in range(ev.n_img):
                #plt.subplot(ev.n_spec,ev.n_img,i*ev.n_img+j+1)
                plt.subplot(2,np.ceil(ev.n_img/2.),i*ev.n_img+j+1)
                plt.title(str(j) +','+ str(i))
                plt.imshow(subflat[j][i], origin='lower')
        plt.tight_layout()
        plt.savefig(ev.eventdir + '/figs/fig1000-Flats.png')
        # Plot masks
        plt.figure(1001, figsize=(12,8))
        plt.clf()
        plt.suptitle('Mask Frames')
        for i in range(ev.n_spec):
            for j in range(ev.n_img):
                #plt.subplot(ev.n_spec,ev.n_img,i*ev.n_img+j+1)
                plt.subplot(2,np.ceil(ev.n_img/2.),i*ev.n_img+j+1)
                plt.title(str(j) +','+ str(i))
                plt.imshow(flatmask[j][i], origin='lower')
        plt.tight_layout()
        plt.savefig(ev.eventdir + '/figs/fig1001-Masks.png')
        if ev.detector == 'IR':
            # Plot Slit shift
            plt.figure(1004, figsize=(12,8))
            plt.clf()
            plt.suptitle('Model Slit Tilts/Shifts')
            plt.plot(ev.shift_values, ev.yfit, '.')
            plt.plot(ev.slitshift, range(ev.ywindow[0][0],ev.ywindow[0][1]), 'r-', lw=2)
            plt.xlim(-1,1)
            plt.savefig(ev.eventdir + '/figs/fig1004-SlitTilt.png')
        plt.pause(0.1)
    """

    ev.ra       = data_mhdr[0]['RA_TARG']*np.pi/180
    ev.dec      = data_mhdr[0]['DEC_TARG']*np.pi/180
    if ev.horizonsfile != None:
        # Apply light-time correction, convert to BJD_TDB
        # Horizons file created for HST around time of observations
        print("Converting times to BJD_TDB...")
        ev.bjd_corr = suntimecorr.suntimecorr(ev.ra, ev.dec, ev.jd, ev.horizonsfile)
        bjdutc      = ev.jd + ev.bjd_corr/86400.
        ev.bjdtdb   = utc_tt.utc_tt(bjdutc,ev.leapdir)
        print('BJD_corr range: ' + str(ev.bjd_corr[0]) + ', ' + str(ev.bjd_corr[-1]))
    else:
        print("No Horizons file found.")
        ev.bjdtdb   = ev.jd

    if n_reads > 1:
        ev.n_reads  = n_reads
        # Subtract pairs of subframes
        diffdata = np.zeros((ev.n_files,ev.n_reads-1,subny,subnx))
        differr  = np.zeros((ev.n_files,ev.n_reads-1,subny,subnx))
        for m in range(ev.n_files):
            for n in range(n_reads-1):
                #diffmask[m,n] = np.copy(flatmask[j][0])
                #diffmask[m,n][np.where(suberr[m,n  ] > diffthresh*np.std(suberr[m,n  ]))] = 0
                #diffmask[m,n][np.where(suberr[m,n+1] > diffthresh*np.std(suberr[m,n+1]))] = 0
                diffdata[m,n] = subdata[m,n+1]-subdata[m,n]
                differr [m,n] = np.sqrt(suberr[m,n+1]**2+suberr[m,n]**2)
    else:
        # FLT data has already been differenced
        # FLT files subtract first from last, 2 reads
        ev.n_reads  = 2
        diffdata    = subdata
        differr     = suberr

    diffmask = np.zeros((ev.n_files,ev.n_reads-1,subny,subnx))
    guess    = np.zeros((ev.n_files,ev.n_reads-1),dtype=int)
    for m in range(ev.n_files):
        #Select appropriate mask
        #if ev.n_img == (np.max(ev.orbitnum)+1):
        #    j = int(ev.orbitnum[m])
        #else:
        #    j = 0
        for n in range(n_reads-1):
            diffmask[m,n] = np.copy(flatmask[m][0])
            try:
                diffmask[m,n][ np.where(differr[m,n] > ev.diffthresh*
                              np.median(differr[m,n],axis=1)[:,np.newaxis])] = 0
                #diffdata[m,n] *= diffmask[m,n]
            except:
                # May fail for FLT files
                print("Diffthresh failed.")

            foo         = diffdata[m,n]*diffmask[m,n]
            guess[m,n]  = np.median(np.where(foo > np.mean(foo))[0]).astype(int)
        # Guess may be skewed if first file is zeros
        if guess[m,0] < 0 or guess[m,0] > subny:
            guess[m,0] = guess[m,1]

    # Compute full scan length
    ev.scanHeight   = np.zeros(ev.n_files)
    for m in range(ev.n_files):
        scannedData = np.sum(subdata[m,-1], axis=1)
        xmin        = np.min(guess[m])
        xmax        = np.max(guess[m])
        scannedData/= np.median(scannedData[xmin:xmax+1])
        scannedData-= 0.5
        #leftEdge    = np.where(scannedData > 0)/2)[0][0]
        #rightEdge   = np.where(scannedData > 0)/2)[0][-1]
        #yrng        = range(leftEdge-5, leftEdge+5, 1)
        yrng        = range(subny)
        spline      = spi.UnivariateSpline(yrng, scannedData[yrng], k=3, s=0)
        roots       = spline.roots()
        try:
            ev.scanHeight[m]    = roots[1]-roots[0]
        except:
            pass

    #Outlier rejection of sky background along time axis
    print("Performing background outlier rejection...")
    import sigrej, optspex
    for p in range(2):
        iscan   = np.where(ev.scandir == p)[0]
        if len(iscan) > 0:
            for n in range(ev.n_reads-1):
                # Set limits on the sky background
                x1      = (guess[iscan,n].min()-ev.fitbghw).astype(int)
                x2      = (guess[iscan,n].max()+ev.fitbghw).astype(int)
                bgdata1 = diffdata[iscan,n,:x1 ]
                bgmask1 = diffmask[iscan,n,:x1 ]
                bgdata2 = diffdata[iscan,n, x2:]
                bgmask2 = diffmask[iscan,n, x2:]
                bgerr1  = np.median(suberr[iscan,n,:x1 ])
                bgerr2  = np.median(suberr[iscan,n, x2:])
                estsig1 = [bgerr1 for j in range(len(ev.sigthresh))]
                estsig2 = [bgerr2 for j in range(len(ev.sigthresh))]
                diffmask[iscan,n,:x1 ] = sigrej.sigrej(bgdata1, ev.sigthresh, bgmask1, estsig1)
                diffmask[iscan,n, x2:] = sigrej.sigrej(bgdata2, ev.sigthresh, bgmask2, estsig2)

    # Write background
    #global bg, diffmask
    def writeBG(arg):
        background, mask, m, n = arg
        bg[m,n] = background
        diffmask[m,n] = mask
        return

    # STEP 3: Fit sky background with out-of-spectra data
    # FINDME: parallelrize bg subtraction
    print("Performing background subtraction...")
    x1  = np.zeros((ev.n_files,ev.n_reads-1), dtype=int)
    x2  = np.zeros((ev.n_files,ev.n_reads-1), dtype=int)
    bg  = np.zeros((diffdata.shape))
    if ev.ncpu == 1:
        # Only 1 CPU
        for m in range(ev.n_files):
            for n in range(ev.n_reads-1):
                x1[m,n] = (guess[m,n]-ev.fitbghw).astype(int)
                x2[m,n] = (guess[m,n]+ev.fitbghw).astype(int)
                writeBG(hst.fitbg(diffdata[m,n], diffmask[m,n], x1[m,n], x2[m,n],
                                  ev.bgdeg, ev.p3thresh, isplots, m, n, ev.n_files))
    else:
        # Multiple CPUs
        pool = mp.Pool(ev.ncpu)
        for m in range(ev.n_files):
            for n in range(ev.n_reads-1):
                x1[m,n] = (guess[m,n]-ev.fitbghw).astype(int)
                x2[m,n] = (guess[m,n]+ev.fitbghw).astype(int)
                res = pool.apply_async(hst.fitbg, args=(diffdata[m,n], diffmask[m,n], x1[m,n], x2[m,n],
                                       ev.bgdeg, ev.p3thresh, isplots, m, n, ev.n_files), callback=writeBG)
        pool.close()
        pool.join()
        res.wait()
    print(" Done.")

    # STEP 2: Calculate variance
    bgerr       = np.std(bg, axis=2)/np.sqrt(np.sum(diffmask, axis=2))
    bgerr[np.where(np.isnan(bgerr))] = 0.
    ev.v0      += np.mean(bgerr**2)
    variance    = abs(diffdata) / ev.gain + ev.v0
    #variance    = abs(subdata*submask) / gain + v0
    # Perform background subtraction
    diffdata   -= bg

    #
    '''
    foo = np.sum(diffdata*diffmask, axis=2)
    guess = []
    for i in range(nreads-1):
        guess.append(np.median(np.where(foo[i] > np.mean(foo[i]))[0]).astype(int))
    guess   = np.array(guess)
    # Guess may be skewed if first file is zeros
    if guess[0] < 0 or guess[0] > subnx:
        guess[0] = guess[1]
    '''

    # Write drift2D
    def writeDrift2D(arg):
        drift2D, m, n = arg
        # Assign to array of spectra and uncertainties
        ev.drift2D[m,n] = drift2D
        return
    '''
    # Calulate drift2D
    def calcDrift2D():#im1, im2, m, n):
        print("test")
        drift2D = imr.chi2_shift(im1, im2, boundary='constant', nthreads=4,
                                 zeromean=False, return_error=False)
        return (drift2D, m, n)
    '''
    print("Calculating 2D drift...")
    #FINDME: instead of calculating scanHeight, consider fitting stretch factor
    ev.drift2D  = np.zeros((ev.n_files, ev.n_reads-1, 2))
    if ev.ncpu == 1:
        # Only 1 CPU
        for m in range(ev.n_files):
            p   = int(ev.scandir[m])
            for n in range(ev.n_reads-1):
                writeDrift2D(hst.calcDrift2D(diffdata[ev.iref[p],n]*diffmask[ev.iref[p],n],
                                             diffdata[m,n]*diffmask[m,n], m, n, ev.n_files))
    else:
        # Multiple CPUs
        pool = mp.Pool(ev.ncpu)
        for m in range(ev.n_files):
            p   = int(ev.scandir[m])
            for n in range(ev.n_reads-1):
                #res = pool.apply_async(hst.calcDrift2D)
                res = pool.apply_async(hst.calcDrift2D, args=(diffdata[ev.iref[p],n]*diffmask[ev.iref[p],n],
                                       diffdata[m,n]*diffmask[m,n], m, n, ev.n_files), callback=writeDrift2D)
        pool.close()
        pool.join()
        res.wait()
    print(" Done.")
    #np.save("drift2D.npy",ev.drift2D)

    #global shiftdata, shiftmask
    print("Performing rough, pixel-scale drift correction...")
    import scipy.ndimage.interpolation as spni
    ev.drift2D_int  = np.round(ev.drift2D,0)
    shiftdata   = np.zeros(diffdata.shape)
    shiftmask   = np.zeros(diffmask.shape)
    shiftvar    = np.zeros(diffdata.shape)
    shiftbg     = np.zeros(diffdata.shape)
    # Correct for drift by integer pixel numbers, no interpolation
    for m in range(ev.n_files):
        for n in range(ev.n_reads-1):
            shiftdata[m,n] = spni.shift(diffdata[m,n], -1*ev.drift2D_int[m,n,::-1], order=0,
                             mode='constant', cval=0)
            shiftmask[m,n] = spni.shift(diffmask[m,n], -1*ev.drift2D_int[m,n,::-1], order=0,
                             mode='constant', cval=0)
            shiftvar [m,n] = spni.shift(variance[m,n], -1*ev.drift2D_int[m,n,::-1], order=0,
                             mode='constant', cval=0)
            shiftbg  [m,n] = spni.shift(bg      [m,n], -1*ev.drift2D_int[m,n,::-1], order=0,
                             mode='constant', cval=0)
            """
            # spni.shift does not handle constant boundaries correctly
            if ev.drift2D_int[m,n,0] > 0:
                shiftdata[m,n,:,-1*ev.drift2D_int[m,n,0]:] = 0
                shiftmask[m,n,:,-1*ev.drift2D_int[m,n,0]:] = 0
                shiftvar [m,n,:,-1*ev.drift2D_int[m,n,0]:] = 0
                shiftbg  [m,n,:,-1*ev.drift2D_int[m,n,0]:] = 0
            elif ev.drift2D_int[m,n,0] < 0:
                #print(m,n,-1*ev.drift2D_int[m,n,0])
                shiftdata[m,n,:,:-1*ev.drift2D_int[m,n,0]] = 0
                shiftmask[m,n,:,:-1*ev.drift2D_int[m,n,0]] = 0
                shiftvar [m,n,:,:-1*ev.drift2D_int[m,n,0]] = 0
                shiftbg  [m,n,:,:-1*ev.drift2D_int[m,n,0]] = 0
            """

    # Outlier rejection of full frame along time axis
    print("Performing full-frame outlier rejection...")
    for p in range(2):
        iscan   = np.where(ev.scandir == p)[0]
        if len(iscan) > 0:
            for n in range(ev.n_reads-1):
                #y1  = guess[ev.iref,n] - ev.spec_width
                #y2  = guess[ev.iref,n] + ev.spec_width
                #estsig      = [differr[ev.iref,n,y1:y2] for j in range(len(ev.sigthresh))]
                shiftmask[iscan,n] = sigrej.sigrej(shiftdata[iscan,n], ev.sigthresh, shiftmask[iscan,n])#, estsig)
    """
    # Replace bad pixels using 2D Gaussian kernal along x and time axes
    def writeReplacePixels(arg):
        shift, m, n, i, j   = arg
        shiftdata[m,n,i,j]  = shift
        return

    #import smoothing
    #reload(smoothing)
    ny, nx, sy, sx = (2,2,1,1)
    wherebad    = np.array(np.where(shiftmask==0)).T
    #smdata      = np.copy(shiftdata)
    print("Replacing " + str(len(wherebad)) + " bad pixels...")
    k       = 0
    ktot    = len(wherebad)
    #FINDME: multiple CPUs is inefficient
    if ev.ncpu >= 1:
        # Only 1 CPU
        for m,n,i,j in wherebad:
            #sys.stdout.write('\r'+str(k+1)+'/'+str(len(wherebad)))
            #sys.stdout.flush()
            writeReplacePixels(hst.replacePixels(shiftdata[:,n,:,j], shiftmask[:,n,:,j], m, n, i, j, k, ktot, ny, nx, sy, sx))
            #Pad image initially with zeros
            #newim = np.zeros(np.array(shiftdata[:,n,:,j].shape) + 2*np.array((ny, nx)))
            #newim[ny:-ny, nx:-nx] = shiftdata[:,n,:,j]
            #Calculate kernel
            #gk = smoothing.gauss_kernel_mask2((ny,nx), (sy,sx), (m,i), shiftmask[:,n,:,j])
            #shiftdata[m,n,i,j] = np.sum(gk * newim[m:m+2*ny+1, i:i+2*nx+1])
            k += 1
    else:
        # Multiple CPUs
        pool = mp.Pool(ev.ncpu)
        for m,n,i,j in wherebad:
            res = pool.apply_async(hst.replacePixels, args=(shiftdata[:,n,:,j], shiftmask[:,n,:,j], m, n, i, j, k, ktot, ny, nx, sy, sx), callback=writeReplacePixels)
            k += 1
        pool.close()
        pool.join()
        res.wait()
    print(" Done.")
    """
    if isplots >= 3:
        for m in range(ev.n_files):
            for n in range(ev.n_reads-1):
                plt.figure(1010)
                plt.clf()
                plt.suptitle(str(m) + "," + str(n))
                plt.subplot(211)
                plt.imshow(shiftdata[m,n]*shiftmask[m,n], origin='lower', aspect='auto', vmin=0, vmax=500)
                plt.subplot(212)
                #plt.imshow(submask[i], origin='lower', aspect='auto', vmax=1)
                mean = np.median(shiftbg[m,n])
                std  = np.std(shiftbg[m,n])
                plt.imshow(shiftbg[m,n], origin='lower', aspect='auto',vmin=mean-3*std,vmax=mean+3*std)
                plt.savefig(ev.eventdir+'/figs/fig1010-'+str(m)+'-'+str(n)+'-Image+Background.png')
                #plt.pause(0.1)

    """
    apdata  = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    apmask  = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    apvar   = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    apbg    = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    for n in range(ev.n_reads-1):
        y1  = guess[ev.iref,n] - ev.spec_width
        y2  = guess[ev.iref,n] + ev.spec_width
        apdata[:,n] = shiftdata[:,n,y1:y2]
        apmask[:,n] = shiftmask[:,n,y1:y2]
        apvar [:,n] = shiftvar [:,n,y1:y2]
        apbg  [:,n] = shiftbg  [:,n,y1:y2]
    """
    print("Performing sub-pixel drift correction...")
    istart      = 0
    #corrdata    = np.zeros(diffdata.shape)
    #corrmask    = np.zeros(diffdata.shape)
    # Select aperture data
    apdata  = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    apmask  = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    apvar   = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    apbg    = np.zeros((ev.n_files,ev.n_reads-1,ev.spec_width*2,subnx))
    iy, ix  = np.indices((ev.spec_width*2,subnx))
    iyy, ixx= np.indices((subny,subnx))
    #FINDME: should be using (3,3)
    kx, ky  = (1,1)
    # Correct for drift
    for n in range(ev.n_reads-1):
        #FINDME: change below to handle case of single scan direction
        y1  = [guess[ev.iref[0],n] - ev.spec_width, guess[ev.iref[1],n] - ev.spec_width]
        #y2  = guess[ev.iref,n] + ev.spec_width
        for m in range(ev.n_files):
            p   = int(ev.scandir[m])
            # Data
            spline = spi.RectBivariateSpline(range(subny), range(subnx), shiftdata[m,n], kx=kx, ky=ky, s=0)
            apdata[m,n] = (spline.ev((iy+y1[p]+ev.drift2D[m,n,1]-ev.drift2D_int[m,n,1]).flatten(),
                          (ix+ev.drift2D[m,n,0]-ev.drift2D_int[m,n,0]).flatten())).reshape \
                          (ev.spec_width*2,subnx)
            #spni.shift works identically to spi.RectBivariateSpline
            #set prefilter=False and order=3 to enable spline filtering (smoothing)
            #apdata[m,n] = spni.shift(shiftdata[m,n], ev.drift2D_int[m,n,::-1]-ev.drift2D[m,n,::-1], order=1,
            #                         mode='constant', cval=0, prefilter=True)[y1:y2]
            # Mask
            spline = spi.RectBivariateSpline(range(subny), range(subnx), shiftmask[m,n], kx=kx, ky=ky, s=0)
            apmask[m,n] = np.round((spline.ev((iy+y1[p]+ev.drift2D[m,n,1]-ev.drift2D_int[m,n,1]).flatten(),
                          (ix+ev.drift2D[m,n,0]-ev.drift2D_int[m,n,0]).flatten())).reshape \
                          (ev.spec_width*2,subnx),0).astype(int)
            # Variance
            spline = spi.RectBivariateSpline(range(subny), range(subnx), shiftvar[m,n], kx=kx, ky=ky, s=0)
            apvar[m,n] = (spline.ev((iy+y1[p]+ev.drift2D[m,n,1]-ev.drift2D_int[m,n,1]).flatten(),
                         (ix+ev.drift2D[m,n,0]-ev.drift2D_int[m,n,0]).flatten())).reshape \
                         (ev.spec_width*2,subnx)
            # Background
            spline = spi.RectBivariateSpline(range(subny), range(subnx), shiftbg[m,n], kx=kx, ky=ky, s=0)
            apbg[m,n] = (spline.ev((iy+y1[p]+ev.drift2D[m,n,1]-ev.drift2D_int[m,n,1]).flatten(),
                        (ix+ev.drift2D[m,n,0]-ev.drift2D_int[m,n,0]).flatten())).reshape \
                        (ev.spec_width*2,subnx)

    """
    #Outlier rejection of aperture along time axis
    print("Performing aperture outlier rejection...")
    for n in range(ev.n_reads-1):
        #y1  = guess[ev.iref,n] - ev.spec_width
        #y2  = guess[ev.iref,n] + ev.spec_width
        #estsig      = [differr[ev.iref,n,y1:y2] for j in range(len(ev.sigthresh))]
        apmask[:,n] = sigrej.sigrej(apdata[:,n], ev.sigthresh, apmask[:,n])#, estsig)
    """
    # STEP 4: Extract standard spectrum and its variance
    #stdspec     = np.zeros((ev.n_files,subnx))
    #stdvar      = np.zeros((ev.n_files,subnx))
    #stdbg       = np.zeros((ev.n_files,subnx))
    #fracMaskReg = np.zeros(nreads-1)
    #for n in range(nreads-1):
    #stdspec     = np.sum((apdata*apmask), axis=2)
    #stdvar      = np.sum((apvar *apmask), axis=2)
    stdspec     = np.sum(apdata, axis=2)
    stdvar      = np.sum(apvar , axis=2)
    #stdbg       = np.sum((bg      *apmask), axis=2)
    # Compute fraction of masked pixels within regular spectral extraction window
    numpixels   = 2.*ev.spec_width*subnx
    fracMaskReg = (numpixels - np.sum(apmask,axis=(2,3)))/numpixels

    # Compute median frame
    ev.meddata  = np.median(apdata, axis=0)

    # Extract optimal spectrum with uncertainties
    print("Performing optimal spectral extraction...")
    spectra     = np.zeros((stdspec.shape))
    specerr     = np.zeros((stdspec.shape))
    fracMaskOpt = np.zeros((ev.n_files, ev.n_reads-1))
    #tempmask    = np.ones((ev.spec_width*2,subnx))
    for m in range(ev.n_files):
        sys.stdout.write('\r'+str(m+1)+'/'+str(ev.n_files))
        sys.stdout.flush()
        for n in range(ev.n_reads-1):
            #smoothspec  = smooth.medfilt(stdspec[i], window_len)
            spectra[m,n], specerr[m,n], mask = optspex.optimize(apdata[m,n], apmask[m,n], apbg[m,n], stdspec[m,n], ev.gain, ev.v0, p5thresh=ev.p5thresh, p7thresh=ev.p7thresh, fittype=ev.fittype, window_len=ev.window_len, deg=ev.deg, n=m, iread=n, isplots=isplots, eventdir=ev.eventdir, meddata=ev.meddata[n])
            # Compute fraction of masked pixels within optimal spectral extraction window
            numpixels           = 1.*mask.size
            fracMaskOpt[m,n]    = (np.sum(apmask[m,n]) - np.sum(mask))/numpixels
    print(" Done.")

    if isplots >= 3:
        for m in range(ev.n_files):
            for n in range(ev.n_reads-1):
                plt.figure(1011)
                plt.clf()
                plt.suptitle(str(m) + "," + str(n))
                #plt.errorbar(ev.wave[m], stdspec, np.sqrt(stdvar), fmt='-')
                plt.errorbar(range(subnx), stdspec[m,n], np.sqrt(stdvar[m,n]), fmt='b-')
                plt.errorbar(range(subnx), spectra[m,n], specerr[m,n], fmt='g-')
                plt.savefig(ev.eventdir+'/figs/fig1011-'+str(m)+'-'+str(n)+'-Spectrum.png')
                #plt.pause(0.1)

    # Calculate total time
    total = (time.time() - t0)/60.
    print('\nTotal time (min): ' + str(np.round(total,2)))

    ev.guess = guess

    # Save results
    print('Saving results...')
    aux.spectra     = spectra
    aux.specerr     = specerr
    #aux.specbg      = specbg
    aux.fracMaskReg = fracMaskReg
    aux.fracMaskOpt = fracMaskOpt
    aux.data_hdr    = data_hdr
    aux.data_mhdr   = data_mhdr
    aux.mask        = mask
    #aux.trace2d     = trace2d
    #aux.wave2d      = wave2d
    #aux.bias_mhdr   = bias_mhdr
    aux.subflat = subflat
    me.saveevent(aux, ev.eventdir + '/d-' + ev.eventname + '-data')
    me.saveevent( ev, ev.eventdir + '/d-' + ev.eventname + '-w2')    #, delete=['flat_mhdr'])

    if isplots:
        # 2D light curve without drift correction
        plt.figure(1012, figsize=(8,ev.n_files/20.+0.8))
        plt.clf()
        if ev.grism == 'G102':
            wmin        = 0.82
            wmax        = 1.22
        else: #G141
            wmin        = 1.125
            wmax        = 1.65
        iwmin       = np.where(ev.wave[0][0]>wmin)[0][0]
        iwmax       = np.where(ev.wave[0][0]>wmax)[0][0]
        vmin        = 0.97
        vmax        = 1.03
        #FINDME
        normspec    = np.zeros((ev.n_files,subnx))
        for p in range(2):
            iscan   = np.where(ev.scandir == p)[0]
            if len(iscan) > 0:
                normspec[iscan] = np.mean(spectra[iscan],axis=1)/ \
                                  np.mean(spectra[iscan[ev.inormspec[0]:ev.inormspec[1]]],axis=(0,1))
        #normspec    = np.mean(spectra,axis=1)/np.mean(spectra[ev.inormspec[0]:ev.inormspec[1]],axis=(0,1))
        #normspec    = np.mean(spectra,axis=1)/np.mean(spectra[-6:],axis=(0,1))
        #normspec    = np.mean(ev.stdspec,axis=1)/np.mean(ev.stdspec[-6:],axis=(0,1))
        ediff       = np.zeros(ev.n_files)
        for m in range(ev.n_files):
            ediff[m]    = 1e6*np.median(np.abs(np.ediff1d(normspec[m,iwmin:iwmax])))
            plt.scatter(ev.wave[0][0], np.zeros(subnx)+m, c=normspec[m],
                        s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
        plt.title("MAD = "+str(np.round(np.mean(ediff),0).astype(int)) + " ppm")
        plt.xlim(wmin,wmax)
        plt.ylim(0,ev.n_files)
        plt.ylabel('Frame Number')
        plt.xlabel('Wavelength ($\mu m$)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(ev.eventdir+'/figs/fig1012-2D_LC.png')
        '''
        # Plot individual non-destructive reads
        vmin        = 0.97
        vmax        = 1.03
        iwmin       = np.where(ev.wave[0][0]>wmin)[0][0]
        iwmax       = np.where(ev.wave[0][0]>wmax)[0][0]
        normspec    = spectra[:,istart:]/np.mean(spectra[-6:,istart:],axis=0)
        for n in range(ev.n_reads-1):
            plt.figure(1100+n, figsize=(8,6.5))
            plt.clf()
            ediff       = np.zeros(ev.n_files)
            for m in range(ev.n_files):
                ediff[m]    = 1e6*np.median(np.abs(np.ediff1d(normspec[m,n,iwmin:iwmax])))
                plt.scatter(ev.wave[0][0], np.zeros(normspec.shape[-1])+m, c=normspec[m,n],
                            s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
            plt.title("MAD = "+str(np.round(np.mean(ediff),0)) + " ppm")
            plt.xlim(wmin,wmax)
            plt.ylim(0,ev.n_files)
            plt.ylabel('Frame Number')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(ev.eventdir+'/figs/fig'+str(1100+n)+'-2D_LC.png')
        '''

    #FINDME
    ev.spectra  = spectra
    ev.subflat  = subflat
    ev.subdata  = subdata
    ev.suberr   = suberr
    ev.diffdata = diffdata
    ev.differr  = differr
    ev.diffmask = diffmask
    ev.shiftdata=shiftdata
    ev.shiftmask=shiftmask
    ev.bg       = bg
    ev.apdata   = apdata
    ev.apmask   = apmask
    ev.stdspec  = stdspec
    '''
    #ev.mad2 = np.round(np.mean(ediff),0).astype(int)
    f = open('W2_MAD_'+ madVariable +'.txt','a+')
    f.write(str(madVarSet) + ',' + str(np.round(np.mean(ediff),0).astype(int)) + '\n')
    f.close()
    print('W2_MAD_'+ madVariable +'.txt saved\n')
    '''

    return ev
