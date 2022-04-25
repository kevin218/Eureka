
# NIRCam specific rountines go here
import numpy as np
import multiprocessing as mp
from astropy.io import fits
import scipy.interpolate as spi
import scipy.ndimage as spni
from . import background, nircam
from . import bright2flux as b2f
from . import hst_scan as hst
from ..lib import suntimecorr, utc_tt

def preparation_step(meta, log):

    meta.gain = 1

    obstimes, CRPIX1, CRPIX2, postarg1, postarg2, ny, meta, log = separate_direct(meta, log)
    meta, log = separate_scan_direction(obstimes, postarg2, meta, log)
    
    # Calculate centroid of direct image(s)
    meta.centroid = hst.imageCentroid(meta.direct_list, meta.centroidguess, meta.centroidtrim, ny, CRPIX1, CRPIX2, postarg1, postarg2)
    
    # Initialize listto hold centroid positions from later steps in this stage
    meta.centroids = []
    meta.subflat = []
    meta.flatmask = []
    meta.scanHeight = []
    meta.diffmask = []
    meta.subdiffmask = []
    meta.drift2D = []
    meta.drift2D_int = []
    meta.subdata_ref = []
    meta.diffmask_ref = []

    return meta, log

def conclusion_step(meta, log):
    # Convert these lists to arrays
    meta.centroids = np.array(meta.centroids)
    meta.subflat = np.array(meta.subflat)
    meta.flatmask = np.array(meta.flatmask)
    meta.scanHeight = np.array(meta.scanHeight)
    meta.diffmask = np.array(meta.diffmask)
    meta.subdiffmask = np.array(meta.subdiffmask)
    meta.drift2D = np.array(meta.drift2D)
    meta.drift2D_int = np.array(meta.drift2D_int)
    meta.subdata_ref = np.array(meta.subdata_ref)
    meta.diffmask_ref = np.array(meta.diffmask_ref)

    return meta, log

def separate_direct(meta, log):

    # Figure out which files are IMAGING or SPECTROSCOPIC
    obstypes = []
    obstimes = []
    postarg1 = []
    postarg2 = []
    CRPIX1 = []
    CRPIX2 = []
    for fname in meta.segment_list:
        with fits.open(fname) as file:
            obstypes.append(file[0].header['OBSTYPE'])
            obstimes.append(file[0].header['EXPSTART'])
            # Get the POSTARG2 parameter so we can later separate scan directions
            postarg1.append(file[0].header['POSTARG1'])
            postarg2.append(file[0].header['POSTARG2'])
            CRPIX1.append(file[1].header['CRPIX1'])
            CRPIX2.append(file[1].header['CRPIX2'])
            ny = file[1].header['NAXIS2']
    obstypes = np.array(obstypes)
    obstimes = np.array(obstimes)
    postarg1 = np.array(postarg1)
    postarg2 = np.array(postarg2)
    CRPIX1 = np.array(CRPIX1)
    CRPIX2 = np.array(CRPIX2)

    # Make sure all the files are in order of observation time
    order = np.argsort(obstimes)
    meta.segment_list = meta.segment_list[order]
    obstypes = obstypes[order]
    obstimes = obstimes[order]
    postarg1 = postarg1[order]
    postarg2 = postarg2[order]
    CRPIX1 = CRPIX1[order]
    CRPIX2 = CRPIX2[order]

    if np.all(obstypes=='IMAGING'):
        # All observations are in imaging mode
        raise AssertionError('All observations cannot be in imaging mode!\n'
        +'Eureka is currently not capable of handling imaging datasets from Hubble/WFC3.')
    elif np.all(obstypes=='SPECTROSCOPIC'):
        # All observations are in spectroscopy mode
        # This is an issue as an imaging mode observation is needed for wavelength calibration
        raise AssertionError('All observations cannot be spectroscopic!\n'
        +'At least one direct image is needed for wavelength calibration.')
    elif np.any(np.logical_and(obstypes!='SPECTROSCOPIC', obstypes!='IMAGING')):
        # There is one or more unexpected OBSTYPEs - throw a useful error
        unknowns = np.unique(obstypes[np.logical_and(obstypes!='SPECTROSCOPIC', obstypes!='IMAGING')])
        raise AssertionError('Unknown OBSTYPE(s) encountered: {}.\n'.format(unknowns)
        +'Expected only SPECTROSCOPIC and IMAGING OBSTYPEs.')
    else:
        # There is a mix of some direct images for wavelength calibration and science spectra as expected
        
        # Make separate lists of direct images and science images
        meta.direct_list = meta.segment_list[obstypes=='IMAGING']
        meta.n_img = len(meta.direct_list)
        meta.segment_list = meta.segment_list[obstypes=='SPECTROSCOPIC']
        meta.num_data_files = len(meta.segment_list)
        postarg1 = postarg1[obstypes=='SPECTROSCOPIC']
        postarg2 = postarg2[obstypes=='SPECTROSCOPIC']
        CRPIX1 = CRPIX1[obstypes=='SPECTROSCOPIC'][0]
        CRPIX2 = CRPIX2[obstypes=='SPECTROSCOPIC'][0]
        
        # Figure out which direct image should be used by each science image
        # If there are multiple direct images, this will use the most recent one
        direct_times = obstimes[obstypes=='IMAGING']
        science_times = obstimes[obstypes=='SPECTROSCOPIC']
        meta.direct_index = np.zeros(meta.segment_list.shape, dtype=int)
        for i in range(len(science_times)):
            meta.direct_index[i] = np.where(science_times[i]>direct_times)[0][-1]

    return obstimes, CRPIX1, CRPIX2, postarg1, postarg2, ny, meta, log

def separate_scan_direction(obstimes, postarg2, meta, log):

    if meta.num_data_files==1:
        # There is only one image
        meta.scandir = np.zeros(meta.num_data_files, dtype=int)
        meta.n_scan0 = 1
        meta.n_scan1 = 0
    else:
        # Assign scan direction
        meta.scandir = np.zeros(meta.num_data_files, dtype=int)
        meta.n_scan0 = 0
        meta.n_scan1 = 0
        scan0 = postarg2[0]
        scan1 = postarg2[1]
        for m in range(meta.num_data_files):
            if postarg2[m] == scan0:
                meta.n_scan0 += 1
            elif postarg2[m] == scan1:
                meta.scandir[m] = 1
                meta.n_scan1 += 1
            else:
                log.writelog('WARNING: Unknown scan direction for file ' + str(m) + '.')
    
    log.writelog("# of files in scan direction 0: " + str(meta.n_scan0), mute=(not meta.verbose))
    log.writelog("# of files in scan direction 1: " + str(meta.n_scan1), mute=(not meta.verbose))

    # Group frames into frame, batch, and orbit number
    meta.framenum, meta.batchnum, meta.orbitnum = hst.groupFrames(obstimes)

    return meta, log

def read(filename, data, meta):
    '''Reads single FITS file from HST's WFC3 instrument.

    Parameters
    ----------
    filename:   str
        Single filename to read
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaClass
        The metadata object

    Returns
    -------
    data: DataClass
        The updated data object with the fits data stored inside

    Notes
    -----
    History:

    - January 2017 Kevin Stevenson
            Initial code as implemented in the WFC3 pipeline
    - 18-19 Nov 2021 Taylor Bell
        Edited and decomposed WFC3 code to integrate with Eureka!          
    '''
    assert isinstance(filename, str)

    #Determine image size and filter/grism
    with fits.open(filename) as hdulist:
        data.filename   = filename
        data.mhdr       = hdulist[0].header
        data.shdr       = hdulist[1].header
        meta.nx         = data.shdr['NAXIS1']
        meta.ny         = data.shdr['NAXIS2']
        meta.grism      = data.mhdr['FILTER']
        meta.detector   = data.mhdr['DETECTOR']
        meta.flatoffset = [[-1*data.shdr['LTV2'], -1*data.shdr['LTV1']]]
        
        # Determine if we are using IMA or FLT files
        if filename.endswith('flt.fits'):
            # FLT files subtract first from last, 2 reads
            meta.nreads = 2
        else:
            meta.nreads = data.shdr['SAMPNUM']

        if data.shdr['BUNIT'] == 'ELECTRONS/S':
            #Science data and uncertainties were previously in units of e-/sec,
            #therefore multiply by sample time to get electrons.
            samptime = data.shdr['SAMPTIME']
        else:
            samptime = 1

        data.data = np.zeros((meta.nreads,meta.ny,meta.nx))    #Flux
        data.err = np.zeros((meta.nreads,meta.ny,meta.nx))    #Flux
        data.dq = np.zeros((meta.nreads,meta.ny,meta.nx))    #Flux
        jd = []
        j = 0
        for rd in range(meta.nreads,0,-1):
            data.data[j] = hdulist['SCI',rd].data*samptime
            data.err[j]  = hdulist['ERR',rd].data*samptime
            data.dq[j]   = hdulist['DQ', rd].data
            jd.append(2400000.5 + hdulist['SCI',rd].header['ROUTTIME']-0.5*hdulist['SCI',rd].header['DELTATIM']/3600/24)
            j += 1
        jd = np.array(jd)

        # data.data = hdulist[1].data*samptime
        # data.err  = hdulist[2].data*samptime
        # data.dq  = hdulist[3].data
    
    # jd  = np.array([2400000.5 + 0.5*(data.mhdr['EXPSTART'] + data.mhdr['EXPEND'])])
    ra  = data.mhdr['RA_TARG']*np.pi/180
    dec = data.mhdr['DEC_TARG']*np.pi/180
    if meta.horizonsfile != None:
        # Apply light-time correction, convert to BJD_TDB
        # Horizons file created for HST around time of observations
        bjd_corr = suntimecorr.suntimecorr(ra, dec, jd, meta.horizonsfile)
        bjdutc      = jd + bjd_corr/86400.
        data.time   = utc_tt.utc_tdb(bjdutc, meta.leapdir)   # FINDME: this was utc_tt, but I believe it should have been utc_tdb instead
        meta.time_units = 'BJD_TDB'
    elif meta.firstFile:
        print("WARNING: No Horizons file found. Using JD rather than BJD_TDB.")
        data.time   = jd
        meta.time_units = 'JD'
    else:
        data.time   = jd

    data.exptime = data.mhdr['EXPTIME']

    # Determine read noise and gain
    readNoise = np.mean((data.mhdr['READNSEA'],
                            data.mhdr['READNSEB'],
                            data.mhdr['READNSEC'],
                            data.mhdr['READNSED']))
    data.v0 = readNoise**2*np.ones_like(data.data)   #Units of electrons
    
    # Calculate centroids for each frame
    centroids = np.zeros((meta.nreads-1,2))
    # Figure out which direct image is the relevant one for this observation
    image_number = np.where(meta.segment_list==filename)[0][0]
    centroid_index = meta.direct_index[image_number]
    # Use the same centroid for each read
    centroids[:,0] = meta.centroid[centroid_index][0]
    centroids[:,1] = meta.centroid[centroid_index][1]
    meta.centroids.append(centroids)
    
    # Calculate trace
    print("Calculating wavelength assuming " + meta.grism + " filter/grism...")
    xrange   = np.arange(0,meta.nx)
    data.wave  = hst.calibrateLambda(xrange, centroids[0], meta.grism)/1e4   #wavelength in microns
    data.wave = data.wave*np.ones((meta.ny,1)) # Assume no skew over the detector
    
    # Figure out which read this file starts and ends with
    data.intstart = image_number*(meta.nreads-1)
    data.intend = (image_number+1)*(meta.nreads-1)
    
    if meta.flatfile == None:
        print('No flat frames found.')
    else:
        data, meta = flatfield(data, meta)
    
    data, meta = difference_frames(data,meta)
    
    data.variance = np.zeros_like(data.data)
    
    return data, meta

def flatfield(data, meta):
    # Make list of master flat field frames
    
    print('Loading flat frames...')
    print(meta.flatfile)
    tempflat, tempmask = hst.makeflats(meta.flatfile, [np.mean(data.wave,axis=0),], [[0,meta.nx],], [[0,meta.ny],], meta.flatoffset, 1, meta.ny, meta.nx, sigma=meta.flatsigma, isplots=meta.isplots_S3)
    subflat  = tempflat[0]
    flatmask = tempmask[0]
    
    meta.subflat.append(subflat)
    meta.flatmask.append(flatmask)
    
    # Calculate reduced image
    subflat[np.where(flatmask==0)] = 1
    subflat[np.where(subflat==0)] = 1
    data.data /= subflat
    
    return data, meta

def difference_frames(data, meta):

    import matplotlib.pyplot as plt

    if meta.nreads > 1:
        # Subtract pairs of subframes
        diffdata = np.zeros((meta.nreads-1,meta.ny,meta.nx))
        differr  = np.zeros((meta.nreads-1,meta.ny,meta.nx))
        for n in range(meta.nreads-1):
            diffdata[n] = data.data[n+1]-data.data[n]
            differr [n-1] = np.sqrt(data.err[n]**2+data.err[n-1]**2)
    else:
        # FLT data has already been differenced
        diffdata    = data.data
        differr     = data.err
    
    diffmask = np.zeros((meta.nreads-1,meta.ny,meta.nx))
    data.guess    = np.zeros((meta.nreads-1),dtype=int)
    for n in range(meta.nreads-1):
        diffmask[n] = np.copy(meta.flatmask[-1][0])
        try:
            diffmask[n][ np.where(differr[n] > meta.diffthresh*
                            np.median(differr[n],axis=1)[:,np.newaxis])] = 0
        except:
            # May fail for FLT files
            print("Diffthresh failed - this may happen for FLT files.")
        
        masked_data = diffdata[n]*diffmask[n]
        data.guess[n]  = np.median(np.where(masked_data > np.mean(masked_data))[0]).astype(int)
    # Guess may be skewed if first read is zeros
    if data.guess[0] < 0 or data.guess[0] > meta.ny:
        data.guess[0] = data.guess[1]
    
    # Compute full scan length
    scannedData = np.sum(data.data[-1], axis=1)
    xmin        = np.min(data.guess)
    xmax        = np.max(data.guess)
    scannedData/= np.median(scannedData[xmin:xmax+1])
    scannedData-= 0.5
    yrng        = range(meta.ny)
    spline      = spi.UnivariateSpline(yrng, scannedData[yrng], k=3, s=0)
    roots       = spline.roots()
    try:
        meta.scanHeight.append(roots[1]-roots[0])
    except:
        pass

    meta.diffmask.append(diffmask)
    # Save the non-differenced frame data in case it is useful
    data.raw_data = np.copy(data.data)
    data.raw_err = np.copy(data.err)
    # Overwrite the data array with the differenced data since that's what we'll use for the other steps
    data.data = diffdata
    data.err = differr
    data.time = data.time[1:]

    return data, meta

def flag_bg(data, meta):
    '''Outlier rejection of sky background along time axis.

    Uses the code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)

    Parameters
    ----------
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaData
        The metadata object

    Returns
    -------
    data:   DataClass
        The updated data object with outlier background pixels flagged.
    '''
    return nircam.flag_bg(data, meta)

def fit_bg(dataim, datamask, datav0, datavariance, n, meta, isplots=False):
    '''Fit for a non-uniform background.

    Uses the code written for NIRCam, but adds on some extra steps 
    '''
    bg, mask, n = nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)

    # Calculate variance assuming background dominated rather than read noise dominated
    bgerr       = np.std(bg[n], axis=0)/np.sqrt(np.sum(meta.subdiffmask[-1][n], axis=0))
    bgerr[np.where(np.logical_not(np.isfinite(bgerr)))] = 0.
    datav0 += np.mean(bgerr**2)
    datavariance = abs(dataim) / meta.gain + datav0

    return (dataim, datamask, datav0, datavariance, n)

def correct_drift2D(data, meta, m):
    '''
    Parameters
    ----------
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaClass
        The metadata object
    m:  int
        The current file number

    '''

    # Write drift2D
    def writeDrift2D(arg):
        drift2D, m, n = arg
        # Assign to array of spectra and uncertainties
        meta.drift2D[-1][n] = drift2D
        return
    
    # Save the reference frame for each scan direction if not yet done
    if m<2:
        # FINDME: This requires that the reference files be the first two files.
        # Using other files as the reference files will require loading in all of the frames at once
        # This will still work for observations with only one scan direction, since the second ref file will never be used
        meta.subdata_ref.append(data.subdata)
        meta.diffmask_ref.append(meta.diffmask[-1])

    print("Calculating 2D drift...")
    #FINDME: instead of calculating scanHeight, consider fitting stretch factor
    drift2D  = np.zeros((meta.nreads-1, 2))
    meta.drift2D.append(drift2D)
    if meta.ncpu == 1:
        # Only 1 CPU
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p   = meta.scandir[m]
        for n in range(meta.nreads-1):
            writeDrift2D(hst.calcDrift2D(meta.subdata_ref[p][0]*meta.subdiffmask[p][0],
                                         data.subdata[n]*meta.subdiffmask[-1][n], m, n, meta.num_data_files))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p   = meta.scandir[m]
        for n in range(meta.nreads-1):
            res = pool.apply_async(hst.calcDrift2D, args=(meta.subdata_ref[p][0]*meta.subdiffmask[p][0],
                                                          data.subdata[n]*meta.subdiffmask[-1][n],
                                                          m, n, meta.num_data_files), callback=writeDrift2D)
        pool.close()
        pool.join()
        res.wait()
    
    print("Performing rough, pixel-scale drift correction...")
    meta.drift2D_int.append(np.round(meta.drift2D[-1],0))
    # Correct for drift by integer pixel numbers, no interpolation
    for n in range(meta.nreads-1):
        data.subdata[n]     = spni.shift(data.subdata[n],  -1*meta.drift2D_int[-1][n,::-1], order=0,
                                         mode='constant', cval=0)
        data.submask[n]     = spni.shift(data.submask[n],  -1*meta.drift2D_int[-1][n,::-1], order=0,
                                         mode='constant', cval=0)
        data.subvariance[n] = spni.shift(data.subvariance[n], -1*meta.drift2D_int[-1][n,::-1], order=0,
                                         mode='constant', cval=0)
        data.subbg[n]       = spni.shift(data.subbg[n],    -1*meta.drift2D_int[-1][n,::-1], order=0,
                                         mode='constant', cval=0)

    # FINDME: The following cannot be run since we don't have the full time axis
    # Outlier rejection of full frame along time axis
    # print("Performing full-frame outlier rejection...")
    # for p in range(2):
    #     iscan   = np.where(ev.scandir == p)[0]
    #     if len(iscan) > 0:
    #         for n in range(meta.nreads-1):
    #             #y1  = data.guess[ev.iref,n] - meta.spec_hw
    #             #y2  = data.guess[ev.iref,n] + meta.spec_hw
    #             #estsig      = [differr[ev.iref,n,y1:y2] for j in range(len(ev.sigthresh))]
    #             shiftmask[iscan,n] = sigrej.sigrej(shiftdata[iscan,n], ev.sigthresh, shiftmask[iscan,n])#, estsig)

    print("Performing sub-pixel drift correction...")
    # Get indices for each pixel
    ix = range(meta.subnx)
    iy = range(meta.subny)
    # Define the degrees of the bivariate spline
    kx, ky  = (1,1) #FINDME: should be using (3,3)
    # Correct for drift
    for n in range(meta.nreads-1):
        # Get index of reference frame
        # (0 = forward scan, 1 = reverse scan)
        p   = meta.scandir[m]
        # Need to swap ix and iy because of numpy
        spline = spi.RectBivariateSpline(iy, ix, data.subdata[n], kx=kx, ky=ky, s=0)
        # Need to subtract drift2D since documentation says (where im1 is the reference image)
        # "Measures the amount im2 is offset from im1 (i.e., shift im2 by -1 * these #'s to match im1)"
        data.subdata[n]  = spline((iy-meta.drift2D[-1][n,1]+meta.drift2D_int[-1][n,1]).flatten(),
                                  (ix-meta.drift2D[-1][n,0]+meta.drift2D_int[-1][n,0]).flatten())
        spline = spi.RectBivariateSpline(iy, ix, data.submask[n], kx=kx, ky=ky, s=0)
        data.submask[n]  = spline((iy-meta.drift2D[-1][n,1]+meta.drift2D_int[-1][n,1]).flatten(),
                                  (ix-meta.drift2D[-1][n,0]+meta.drift2D_int[-1][n,0]).flatten())
        spline = spi.RectBivariateSpline(iy, ix, data.subvariance[n], kx=kx, ky=ky, s=0)
        data.subvariance[n] = spline((iy-meta.drift2D[-1][n,1]+meta.drift2D_int[-1][n,1]).flatten(),
                                     (ix-meta.drift2D[-1][n,0]+meta.drift2D_int[-1][n,0]).flatten())
        spline = spi.RectBivariateSpline(iy, ix, data.subbg[n], kx=kx, ky=ky, s=0)
        data.subbg[n]    = spline((iy-meta.drift2D[-1][n,1]+meta.drift2D_int[-1][n,1]).flatten(),
                                  (ix-meta.drift2D[-1][n,0]+meta.drift2D_int[-1][n,0]).flatten())

    return data, meta