import numpy as np
from astropy.io import fits
import scipy.interpolate as spi
import scipy.ndimage.interpolation as spni
from ..lib import gaussian as g
from ..lib import smooth, smoothing, centroid
import matplotlib.pyplot as plt
from . import optspex
from . import julday
import sys
#import hst_scan as hst
from ..lib import sort_nicely as sn
import astropy.io.fits as pf

try:
    basestring
except NameError:
    basestring = str

# Read FITS files from HST's WFC3 instrument
def read(filenames, returnHdr=True):
    '''
    Reads FITS files from HST's WFC3 instrument.

    Parameters
    ----------
    filenames         : Single or list of filenames to read
    returnHdr         : Set True to return header files

    Returns
    -------
    data            : Array of data frames
    err             : Array of uncertainty frames
    hdr             : List of header files
    master_hdr        : List of master header files

    History
    -------
    Written by Kevin Stevenson        November 2012

    '''
    if isinstance(filenames, basestring):
        filenames    = [filenames]
    hdulist = fits.open(filenames[0])
    nx        = hdulist['SCI',1].header['NAXIS1']
    ny        = hdulist['SCI',1].header['NAXIS2']
    # Determine if we are using IMA or FLT files
    # FLT files already subtract first from last, only 1 read
    if filenames[0].endswith('flt.fits'):
        nreads    = 1
    else:
        nreads    = hdulist['SCI',1].header['SAMPNUM']
    nfiles    = len(filenames)
    data = np.zeros((nfiles,nreads,ny,nx))    #Flux
    err    = np.zeros((nfiles,nreads,ny,nx))    #Uncertainty
    hdr    = []
    mhdr = []
    i    = 0
    for name in filenames:
        hdulist    = fits.open(name)
        hdr.append([])
        j = 0
        for rd in range(nreads,0,-1):
            if hdulist['SCI',rd].header['BUNIT'] == 'ELECTRONS/S':
                #Science data and uncertainties were previously in units of e-/sec,
                #therefore multiply by sample time to get electrons.
                samptime    = hdulist['SCI',rd].header['SAMPTIME']
                data[i,j]     = hdulist['SCI',rd].data*samptime
                err[i,j]    = hdulist['ERR',rd].data*samptime
            else:
                data[i,j]     = hdulist['SCI',rd].data
                err[i,j]    = hdulist['ERR',rd].data
            hdr[i].append(hdulist['SCI',rd].header)
            j += 1
        mhdr.append(hdulist[0].header)
        i += 1

    if returnHdr:
        return data, err, hdr, mhdr
    else:
        return data, err

def imageCentroid(filenames, guess, trim, ny, scifile):
    '''
    Calculate centroid for a list of direct images.

    Parameters
    ----------
    filenames     : List of direct image filenames
    guess         : Paired list, centroid guess
    trim        : Trim image when calculating centroid

    Returns
    -------
    center        : Centroids

    History
    -------
    Written by Kevin Stevenson        November 2013
    Added IRSUB256                    March 2016
    '''
    nfiles = len(filenames)
    centers     = []
    image         = []
    scihdr0 = fits.getheader(scifile,0)
    scihdr1 = fits.getheader(scifile,1)
    for i in range(nfiles):
        image.append(fits.getdata(filenames[i].rstrip()))
        #hdr0 = fits.getheader(filenames[i],0)
        #hdr1 = fits.getheader(filenames[i],1)
        calhdr0 = fits.getheader(filenames[i].rstrip(),0)
        calhdr1 = fits.getheader(filenames[i].rstrip(),1)
        #Calculate centroid, correct for difference in image size, if any
        centers.append(centroid.ctrgauss(image[i], guess=guess, trim=trim) - (image[i].shape[0]-ny)/2.)
        xoffset    = scihdr1['CRPIX1'] - calhdr1['CRPIX1'] + (scihdr0['POSTARG1'] - calhdr0['POSTARG1'])/0.135
        yoffset    = scihdr1['CRPIX2'] - calhdr1['CRPIX2'] + (scihdr0['POSTARG2'] - calhdr0['POSTARG2'])/0.121
        centers[i][0] += yoffset
        centers[i][1] += xoffset
        print("Adding "+str(xoffset)+','+str(yoffset)+" pixels to x,y centroid position.")
        """
        if calhdr0['APERTURE'] == 'IRSUB256':
            #centers[i][1] -= 111
            #xref_correct = xref + CRPIX1_spec - CRPIX1_im + (POSTARG1_spec - POSTARG1_im)/0.135
            #offset    = scihdr1['CRPIX1'] - calhdr1['CRPIX1'] + (scihdr0['POSTARG1'] - calhdr0['POSTARG1'])/0.135
            #centers[i][1] += offset
            xoffset    = scihdr1['CRPIX1'] - calhdr1['CRPIX1'] + (scihdr0['POSTARG1'] - calhdr0['POSTARG1'])/0.135
            yoffset    = scihdr1['CRPIX2'] - calhdr1['CRPIX2'] + (scihdr0['POSTARG2'] - calhdr0['POSTARG2'])/0.121
            centers[i][0] += yoffset
            centers[i][1] += xoffset
            print("****WARNING: Direct image uses IRSUB256, adding "+str(xoffset)+','+str(yoffset)+" pixels to x,y position.")
        if calhdr0['APERTURE'] == 'IRSUB512':
            #centers[i][1] -= 111
            #xref_correct = xref + CRPIX1_spec - CRPIX1_im + (POSTARG1_spec - POSTARG1_im)/0.135
            xoffset    = scihdr1['CRPIX1'] - calhdr1['CRPIX1'] + (scihdr0['POSTARG1'] - calhdr0['POSTARG1'])/0.135
            yoffset    = scihdr1['CRPIX2'] - calhdr1['CRPIX2'] + (scihdr0['POSTARG2'] - calhdr0['POSTARG2'])/0.121
            centers[i][0] += yoffset
            centers[i][1] += xoffset
            print("****WARNING: Direct image uses IRSUB512, adding "+str(xoffset)+','+str(yoffset)+" pixels to x,y position.")
        """
    return centers, image

def groupFrames(dates):
    '''
    Group frames by orbit and batch number

    Parameters
    ----------
    dates         : Time in days
    exptime     : exposure time in seconds
    '''
    n_frames    = len(dates)
    framenum    = np.zeros(n_frames)
    batchnum    = np.zeros(n_frames)
    orbitnum    = np.zeros(n_frames)
    frame     = 0
    batch     = 0
    orbit     = 0
    framegap    = np.median(np.ediff1d(dates))
    orbitgap    = np.max(np.ediff1d(dates))
    for i in range(1,n_frames):
        if dates[i]-dates[i-1] < 2*framegap:
            #New frames, same batch, same orbit
            frame += 1
        elif dates[i]-dates[i-1] > 0.5*orbitgap:
            #Reset frame, new batch, rest orbit
            frame    = 0
            batch    = 0
            orbit += 1
        else: #dates[i]-dates[i-1] > 3*exptime[i]/86400.:
            #Reset frame, new batch, same orbit
            frame    = 0
            batch += 1
        framenum[i] = frame
        batchnum[i] = batch
        orbitnum[i] = orbit

    return framenum, batchnum, orbitnum

def calcTrace(x, centroid, grism):
    '''
    Calculates the WFC3 trace given the position of the direct image in physical pixels.

    Parameters
    ----------
    x             : physical pixel values along dispersion direction over which the trace is calculated
    centroid    : [y,x] pair describing the centroid of the direct image

    Returns
    -------
    y             : computed trace

    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    '''
    yref, xref = centroid

    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x         = x[np.newaxis]

    if grism == 'G141':
        #WFC3-2009-17.pdf
        #Table 1: Field dependent trace descriptions for G141.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DYDX_A_0 = [1.96882E+00,    9.09159E-05,    -1.93260E-03]
        DYDX_A_1 = [1.04275E-02,    -7.96978E-06,     -2.49607E-06,     1.45963E-09,    1.39757E-08,    4.84940E-10]
    elif grism == 'G102':
        #WFC3-2009-18.pdf
        #Table 1: Field dependent trace descriptions for G102.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DYDX_A_0 = [-3.55018E-01,    3.28722E-05,     -1.44571E-03]
        DYDX_A_1 = [ 1.42852E-02,     -7.20713E-06,     -2.42542E-06,     1.18294E-09,    1.19634E-08,    6.17274E-10
]
    else:
        print("Unknown filter/grism: " + grism)
        return 0

    DYDX_0 = DYDX_A_0[0] + DYDX_A_0[1]*xref + DYDX_A_0[2]*yref
    DYDX_1 = DYDX_A_1[0] + DYDX_A_1[1]*xref + DYDX_A_1[2]*yref + \
             DYDX_A_1[3]*xref**2 + DYDX_A_1[4]*xref*yref + DYDX_A_1[5]*yref**2

    y        = DYDX_0 + DYDX_1*(x-xref) + yref

    return y

    return

def calibrateLambda(x, centroid, grism):
    '''
    Calculates coefficients for the dispersion solution

    Parameters
    ----------
    x             : physical pixel values along dispersion direction over which the wavelength is calculated
    centroid    : [y,x] pair describing the centroid of the direct image

    Returns
    -------
    y             : computed wavelength values

    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    '''
    yref, xref = centroid

    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x         = x[np.newaxis]

    if grism == 'G141':
        #WFC3-2009-17.pdf
        #Table 5: Field dependent wavelength solution for G141.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DLDP_A_0 = [8.95431E+03,    9.35925E-02,            0.0,             0.0,             0.0,            0.0]
        DLDP_A_1 = [4.51423E+01,    3.17239E-04,    2.17055E-03,    -7.42504E-07,     3.48639E-07,    3.09213E-07]
    elif grism == 'G102':
        #WFC3-2009-18.pdf
        #Table 5: Field dependent wavelength solution for G102.
        #FINDME: y^2 term not given in Table 5, assuming 0.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DLDP_A_0 = [6.38738E+03,    4.55507E-02,            0.0]
        DLDP_A_1 = [2.35716E+01,    3.60396E-04,    1.58739E-03,    -4.25234E-07,    -6.53726E-08,            0.0]
    else:
        print("Unknown filter/grism: " + grism)
        return 0

    DLDP_0 = DLDP_A_0[0] + DLDP_A_0[1]*xref + DLDP_A_0[2]*yref
    DLDP_1 = DLDP_A_1[0] + DLDP_A_1[1]*xref + DLDP_A_1[2]*yref + \
             DLDP_A_1[3]*xref**2 + DLDP_A_1[4]*xref*yref + DLDP_A_1[5]*yref**2

    y        = DLDP_0 + DLDP_1*(x-xref) + yref

    return y

# Make master flat fields
def makeflats(flatfile, wave, xwindow, ywindow, flatoffset, n_spec, ny, nx, sigma=5, isplots=0):
    '''
    Makes master flatfield image and new mask for WFC3 data.

    Parameters
    ----------
    flatfile        : List of files containing flatfiles images
    wave            : wavelengths
    xwindow         : Array containing image limits in wavelength direction
    ywindow         : Array containing image limits in spatial direction
    n_spec            : Number of spectra
    sigma             : Sigma rejection level

    Returns
    -------
    flat_master     : Single master flatfield image
    mask_master     : Single bad-pixel mask image

    History
    -------
    Written by Kevin Stevenson        November 2012

    '''
    # Read in flat frames
    hdulist     = fits.open(flatfile)
    flat_mhdr     = hdulist[0].header
    #print(hdulist[0].data)
    wmin        = float(flat_mhdr['wmin'])/1e4
    wmax        = float(flat_mhdr['wmax'])/1e4
    #nx        = flat_mhdr['naxis1']
    #ny        = flat_mhdr['naxis2']
    # Build flat field, only compute for subframe
    flat_master = []
    mask_master = []
    for i in range(n_spec):
        # Read and assemble flat field
        # Select windowed region containing the data
        x         = (wave[i] - wmin)/(wmax - wmin)
        #print("Extracting flat field region:")
        #print(ywindow[i][0]+flatoffset[i][0],ywindow[i][1]+flatoffset[i][0],xwindow[i][0]+flatoffset[i][1],xwindow[i][1]+flatoffset[i][1])

        ylower = int(ywindow[i][0]+flatoffset[i][0])
        yupper = int(ywindow[i][1]+flatoffset[i][0])
        xlower = int(xwindow[i][0]+flatoffset[i][1])
        xupper = int(xwindow[i][1]+flatoffset[i][1])
        #flat_window += hdulist[j].data[ylower:yupper,xlower:xupper]*x**j

        if flatfile[-19:] == 'sedFFcube-both.fits':
            #sedFFcube-both

            flat_window = hdulist[1].data[ylower:yupper,xlower:xupper]
            for j in range(2,len(hdulist)):
                flat_window += hdulist[j].data[ylower:yupper,xlower:xupper]*x**(j-1)
        else:
            #WFC3.IR.G141.flat.2
            flat_window = hdulist[0].data[ylower:yupper,xlower:xupper]
            for j in range(1,len(hdulist)):
                #print(j)
                flat_window += hdulist[j].data[ylower:yupper,xlower:xupper]*x**j

        # Initialize bad-pixel mask
        mask_window = np.ones(flat_window.shape,dtype=np.float32)
        #mask_window[ywindow[i][0]:ywindow[i][1],xwindow[i][0]:xwindow[i][1]] = 1
        '''
        # Populate bad pixel submask where flat > sigma*std
        flat_mean = np.mean(subflat)
        flat_std    = np.std(subflat)
        #mask[np.where(np.abs(subflat-flat_mean) > sigma*flat_std)] = 0
        # Mask bad pixels in subflat by setting to zero
        subflat *= mask
        '''
        """
        # Normalize flat by taking out the spectroscopic effect
        # Not fitting median spectrum trace, using straight median instead
        # flat_window /= np.median(flat_window, axis=0)
        medflat        = np.median(flat_window, axis=0)
        fitmedflat     = smooth.smooth(medflat, 15)

        if isplots >= 3:
            plt.figure(1009)
            plt.clf()
            plt.suptitle("Median Flat Frame With Best Fit")
            plt.title(str(i))
            plt.plot(medflat, 'bo')
            plt.plot(fitmedflat, 'r-')
            #plt.savefig()
            plt.pause(0.5)

        flat_window /= fitmedflat
        flat_norm = flat_window / np.median(flat_window[np.where(flat_window <> 0)])
        """
        flat_norm = flat_window

        if sigma != None and sigma > 0:
            # Reject points from flat and flag them in the mask
            #Points that are outliers, do this for the high and low sides separately
            # 1. Reject points < 0
            index = np.where(flat_norm < 0)
            flat_norm[index] = 1.
            mask_window[index] = 0
            # 2. Reject outliers from low side
            ilow    = np.where(flat_norm < 1)
            dbl     = np.concatenate((flat_norm[ilow],1+(1-flat_norm[ilow])))     #Make distribution symetric about 1
            std     = 1.4826*np.median(np.abs(dbl - np.median(dbl)))            #MAD
            ibadpix = np.where((1-flat_norm[ilow]) > sigma*std)
            flat_norm[ilow[0][ibadpix],ilow[1][ibadpix]] = 1.
            mask_window[ilow[0][ibadpix],ilow[1][ibadpix]] = 0
            # 3. Reject outliers from high side
            ihi     = np.where(flat_norm > 1)
            dbl     = np.concatenate((flat_norm[ihi],2-flat_norm[ihi]))         #Make distribution symetric about 1
            std     = 1.4826*np.median(np.abs(dbl - np.median(dbl)))            #MAD
            ibadpix = np.where((flat_norm[ihi]-1) > sigma*std)
            flat_norm[ihi[0][ibadpix],ihi[1][ibadpix]] = 1.
            mask_window[ihi[0][ibadpix],ihi[1][ibadpix]] = 0

        #Put the subframes back in the full frames
        flat_new = np.ones((ny,nx),dtype=np.float32)
        mask     = np.zeros((ny,nx),dtype=np.float32)
        flat_new[ywindow[i][0]:ywindow[i][1],xwindow[i][0]:xwindow[i][1]] = flat_norm
        mask    [ywindow[i][0]:ywindow[i][1],xwindow[i][0]:xwindow[i][1]] = mask_window
        flat_master.append(flat_new)
        mask_master.append(mask)

    return flat_master, mask_master

# Make master flat fields
def makeBasicFlats(flatfile, xwindow, ywindow, flatoffset, ny, nx, sigma=5, isplots=0):
    '''
    Makes master flatfield image (with no wavelength correction) and new mask for WFC3 data.

    Parameters
    ----------
    flatfile        : List of files containing flatfiles images
    xwindow         : Array containing image limits in wavelength direction
    ywindow         : Array containing image limits in spatial direction
    n_spec          : Number of spectra
    sigma           : Sigma rejection level

    Returns
    -------
    flat_master     : Single master flatfield image
    mask_master     : Single bad-pixel mask image

    History
    -------
    Written by Kevin Stevenson      November 2012
    Removed wavelength dependence   February 2018
    '''
    # Read in flat frames
    hdulist     = pf.open(flatfile)
    #flat_mhdr   = hdulist[0].header
    #wmin        = float(flat_mhdr['wmin'])/1e4
    #wmax        = float(flat_mhdr['wmax'])/1e4
    #nx        = flat_mhdr['naxis1']
    #ny        = flat_mhdr['naxis2']
    # Build flat field, only compute for subframe
    flat_master = []
    mask_master = []
    # Read and assemble flat field
    # Select windowed region containing the data
    #x       = (wave[i] - wmin)/(wmax - wmin)
    ylower  = int(ywindow[0]+flatoffset[0])
    yupper  = int(ywindow[1]+flatoffset[0])
    xlower  = int(xwindow[0]+flatoffset[1])
    xupper  = int(xwindow[1]+flatoffset[1])
    if flatfile[-19:] == 'sedFFcube-both.fits':
        #sedFFcube-both
        flat_window = hdulist[1].data[ylower:yupper,xlower:xupper]
    else:
        #WFC3.IR.G141.flat.2
        flat_window = hdulist[0].data[ylower:yupper,xlower:xupper]

    # Initialize bad-pixel mask
    mask_window = np.ones(flat_window.shape,dtype=np.float32)
    #mask_window[ywindow[i][0]:ywindow[i][1],xwindow[i][0]:xwindow[i][1]] = 1
    flat_norm = flat_window

    if sigma != None and sigma > 0:
        # Reject points from flat and flag them in the mask
        # Points that are outliers, do this for the high and low sides separately
        # 1. Reject points < 0
        index = np.where(flat_norm < 0)
        flat_norm[index] = 1.
        mask_window[index] = 0
        # 2. Reject outliers from low side
        ilow    = np.where(flat_norm < 1)
        dbl     = np.concatenate((flat_norm[ilow],1+(1-flat_norm[ilow])))   #Make distribution symetric about 1
        std     = 1.4826*np.median(np.abs(dbl - np.median(dbl)))            #MAD
        ibadpix = np.where((1-flat_norm[ilow]) > sigma*std)
        flat_norm[ilow[0][ibadpix],ilow[1][ibadpix]] = 1.
        mask_window[ilow[0][ibadpix],ilow[1][ibadpix]] = 0
        # 3. Reject outliers from high side
        ihi     = np.where(flat_norm > 1)
        dbl     = np.concatenate((flat_norm[ihi],2-flat_norm[ihi]))         #Make distribution symetric about 1
        std     = 1.4826*np.median(np.abs(dbl - np.median(dbl)))            #MAD
        ibadpix = np.where((flat_norm[ihi]-1) > sigma*std)
        flat_norm[ihi[0][ibadpix],ihi[1][ibadpix]] = 1.
        mask_window[ihi[0][ibadpix],ihi[1][ibadpix]] = 0

    #Put the subframes back in the full frames
    flat_new = np.ones((ny,nx),dtype=np.float32)
    mask     = np.zeros((ny,nx),dtype=np.float32)
    flat_new[ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]] = flat_norm
    mask    [ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]] = mask_window
    flat_master.append(flat_new)
    mask_master.append(mask)

    return flat_master, mask_master

# Calculate slitshifts
def calc_slitshift2(spectrum, xrng, ywindow, xwindow, width=5, deg=1):
    '''
    Calcualte horizontal shift to correct tilt in data using spectrum.

    History
    -------
    Written by Kevin Stevenson        July 2014

    '''
    ny, nx        = spectrum.shape
    # Determine spectrum boundaries on detector along y

    ind         = np.where(spectrum[:,nx//2] > np.mean(spectrum[:,nx//2]))
    # Select smaller subset for cross correlation to ensure good signal
    ystart        = np.min(ind)+5
    yend        = np.max(ind)-5
    subspec     = spectrum[ystart:yend,xwindow[0]:xwindow[1]]
    subny,subnx = subspec.shape
    drift         = np.zeros(subny)
    # Create reference spectrum that is slightly smaller for 'valid' cross correlation
    ref_spec    = subspec[subny//2-1,5:-5]
    ref_spec     -= np.mean(ref_spec[np.where(np.isnan(ref_spec) == False)])
    # Perform cross correlation for each row
    for h in range(subny):
        fit_spec    = subspec[h]
        fit_spec     -= np.mean(fit_spec[np.where(np.isnan(fit_spec) == False)])
        vals        = np.correlate(ref_spec, fit_spec, mode='valid')
        params, err = g.fitgaussian(vals, guess=[width/5., width*1., vals.max()-np.median(vals)])
        drift[h]    = len(vals)/2 - params[1]
    # Fit a polynomial to shifts, evaluate
    shift_values = drift
    yfit         = range(ystart,yend)
    shift_coeffs = np.polyfit(yfit, shift_values, deg=deg)
    shift_models = np.polyval(shift_coeffs, range(ywindow[0],ywindow[1]))

    return shift_models, shift_values, yfit

    #return ev

# Estimate slit shift
def calc_slitshift(wavegrid, xrng, refwave=None, width=3, deg=2):
    '''
    Calculates horizontal shift to correct tilt in data using wavelength.

    Parameters
    ----------


    Returns
    -------


    History
    -------
    Written by Kevin Stevenson        Nov 2013

    '''
    n_spec = len(wavegrid)

    shift_models = []
    shift_values = []
    for i in range(n_spec):
        ny, nx    = wavegrid[i].shape
        loc     = np.zeros(ny)
        if refwave == None:
            refwave = np.mean(wavegrid[i])
        # Interpolate to find location of reference wavelength
        for h in range(ny):
            tck     = spi.splrep(wavegrid[i][h],xrng[i],s=0,k=3)
            loc[h]    = spi.splev(refwave,tck)
        # Fit a polynomial to shifts, evaluate
        shift = loc - loc.mean()
        shift_coeffs = np.polyfit(range(ny), shift, deg=deg)
        shift_models.append(np.polyval(shift_coeffs, range(ny)))
        shift_values.append(shift)
    return shift_models, shift_values
"""
def correct_slitshift(reddata, mask, data_hdr, slitshift, window, isreverse=False):
    '''
    Old routine no longer used.
    '''
    # Create slit-shift-corrected indices of region containing data
    ny, nx     = np.shape(reddata)
    location = find_data(data_hdr)
    subny    = window[1] - window[0]
    subnx    = location[1] - location[0]
    xgrid, ygrid = np.meshgrid(range(location[0],location[1]), range(window[0],window[1]))
    if isreverse:
        xgrid    = (xgrid.T - slitshift).T
    else:
        xgrid    = (xgrid.T + slitshift).T
    # Interpolate reduced data to account for slit shift
    spline     = spi.RectBivariateSpline(range(ny), range(nx), reddata, kx=1, ky=1, s=0)
    # Evaluate interpolated array within region containing data
    subdata    = spline.ev(ygrid.flatten(), xgrid.flatten()).reshape(subny,subnx)
    # Do the same for the bad pixel mask
    foomask = np.zeros((ny,nx))
    foomask[window[0]:window[1],location[0]:location[1]] = mask[window[0]:window[1],location[0]:location[1]]
    spline     = spi.RectBivariateSpline(range(ny), range(nx), foomask, kx=1, ky=1, s=0)
    submask    = spline.ev(ygrid.flatten(), xgrid.flatten()).reshape(subny,subnx).astype(int)

    return subdata, submask
"""
def correct_slitshift2(data, slitshift, mask=None, isreverse=False):
    '''
    Applies horizontal shift to correct tilt in data.

    Parameters
    ----------


    Returns
    -------


    History
    -------
    Written by Kevin Stevenson        June 2012

    '''
    # Create slit-shift-corrected indices
    ny, nx     = np.shape(data)
    xgrid, ygrid = np.meshgrid(range(nx), range(ny))
    if isreverse:
        xgrid    = (xgrid.T - slitshift).T
    else:
        xgrid    = (xgrid.T + slitshift).T
    # Interpolate reduced data to account for slit shift
    spline     = spi.RectBivariateSpline(range(ny), range(nx), data, kx=3, ky=3)
    # Evaluate interpolated array within region containing data
    cordata    = spline.ev(ygrid.flatten(), xgrid.flatten()).reshape(ny,nx)
    # Do the same for the bad pixel mask
    if mask != None:
        spline     = spi.RectBivariateSpline(range(ny), range(nx), mask, kx=3, ky=3)
        #cormask    = np.round(spline.ev(ygrid.flatten(), xgrid.flatten()).reshape(ny,nx),2).astype(int)
        cormask    = spline.ev(ygrid.flatten(), xgrid.flatten()).reshape(ny,nx)
        cormask[np.where(cormask >= 0.9)] = 1
        return cordata, cormask.astype(int)
    else:
        return cordata

# Calulate drift2D
#import image_registration as imr
def calcDrift2D(im1, im2, m, n, n_files):
    try:
        sys.stdout.write('\r'+str(m+1)+'/'+str(n_files))
        sys.stdout.flush()
    except:
        pass
    drift2D = imr.chi2_shift(im1, im2, boundary='constant', nthreads=1,
                             zeromean=False, return_error=False)
    return (drift2D, m, n)

# Fit background
def fitbg(diffdata, diffmask, x1, x2, bgdeg, p3thresh, isplots, m, n, n_files):
    try:
        sys.stdout.write('\r'+str(m+1)+'/'+str(n_files))
        sys.stdout.flush()
    except:
        pass
    bg, mask = optspex.fitbg(diffdata, diffmask, x1, x2, deg=bgdeg,
                             threshold=p3thresh, isrotate=2, isplots=isplots)
    return (bg, mask, m, n)

# Replace bad pixels
def replacePixels(shiftdata, shiftmask, m, n, i, j, k, ktot, ny, nx, sy, sx):
    try:
        sys.stdout.write('\r'+str(k+1)+'/'+str(ktot))
        sys.stdout.flush()
    except:
        pass
    #Pad image initially with zeros
    newim = np.zeros(np.array(shiftdata.shape) + 2*np.array((ny, nx)))
    newim[ny:-ny, nx:-nx] = shiftdata
    #Calculate kernel
    gk = smoothing.gauss_kernel_mask2((ny,nx), (sy,sx), (m,i), shiftmask)
    shift = np.sum(gk * newim[m:m+2*ny+1, i:i+2*nx+1])
    return (shift, m, n, i, j)

# Calculate spectrum
def calcSpectrum(filename, mask, bias_master, flat_master, slitshift, xwindow, ywindow, gain, v0, spec_width, fitbghw, m, n, diffthresh=5, p3thresh=5, p5thresh=10, p7thresh=10, fittype='smooth', window_len=150, deg=3, expand=1, isplots=False, eventdir='.', bgdeg=1):
    '''
    Driver routine for optimal spectral extraction

    Parameters
    ----------

    deg         : Degree of polynomial fit of profile
    isplots     : Set True to produce plots

    Returns
    -------


    History
    -------
    Written by Kevin Stevenson        November 2012

    '''
    """
    filename    = ev.obj_list[0]
    bias_master = ev.bias_master
    flat_master = flat_master[0]
    xwindow     = ev.xwindow[0]
    ywindow     = ev.ywindow[0]
    mask        = mask[0][0]
    expand        = ev.expand
    spec_width    = ev.spec_width
    slitshift     = ev.slitshift
    gain        = ev.gain
    v0            = ev.v0
    """
    sys.stdout.write('\r'+str(m+1))
    sys.stdout.flush()
    #Read file
    frame, frameerr = hst.read(filename, returnHdr=False)
    # Calculate reduced image
    reddata    = ((frame - bias_master)/flat_master)[0]     #.squeeze()
    nreads     = reddata.shape[0]
    subny    = ywindow[1] - ywindow[0]
    subnx    = xwindow[1] - xwindow[0]
    subdata    = reddata[:,ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
    #suberr     = frameerr.squeeze()[:,ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
    suberr     = frameerr[0,:,ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
    submask    = mask[ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]
    if nreads > 1:
        # Subtract pairs of subframes
        diffdata = np.zeros((nreads-1,subny,subnx))
        diffmask = np.zeros((diffdata.shape))
        for i in range(nreads-1):
            diffmask[i] = np.copy(submask)
            diffmask[i][np.where(suberr[i    ] > diffthresh*np.std(suberr[i    ]))] = 0
            diffmask[i][np.where(suberr[i+1] > diffthresh*np.std(suberr[i+1]))] = 0
            diffdata[i] = (subdata[i+1]-subdata[i])*diffmask[i]
    else:
        # FLT data has already been differenced
        nreads        = 2
        diffdata    = subdata
        diffmask    = np.zeros((diffdata.shape))
        diffmask[0] = np.copy(submask)
    #FINDME: Do not shift data
    #cordata = diffdata
    #cormask = diffmask

    # Shift data
    if expand > 1:
        cordata = np.zeros((nreads-1,subny*expand,subnx*expand))
        cormask = np.zeros((cordata.shape))
        for i in range(nreads-1):
            # Increase resolution of data, mask and slitshift by expansion factor
            interpdata        = spni.zoom(diffdata[i], expand)
            interpmask        = np.round(spni.zoom(1.*diffmask[i], expand)).astype(int)
            interpslitshift = spni.zoom(slitshift, expand)*expand
            cordata[i], cormask[i]    = hst.correct_slitshift2(interpdata, interpslitshift, mask=interpmask, isreverse=False)
            '''
            # test plot
            if isplots >= 3:
                plt.figure(1+i)
                plt.clf()
                plt.suptitle(str(m) + "," + str(n) + "," + str(i))
                plt.subplot(211)
                plt.imshow(interpdata*interpmask, origin='lower', aspect='auto', vmax=200)
                plt.subplot(212)
                plt.imshow(cordata[i]*cormask[i], origin='lower', aspect='auto', vmax=200)
                plt.pause(0.1)
            '''
    # Do not apply shift
    else:
        #for i in range(nreads-1):
        #    cordata[i], cormask[i]    = hst.correct_slitshift2(diffdata[i], slitshift, mask=diffmask[i])
        cordata = diffdata
        cormask = diffmask

    # Determine initial guess position for spectrum
    #guess    = np.argmax(np.sum(cordata*cormask, axis=2),axis=1)
    foo = np.sum(cordata*cormask, axis=2)
    guess = []
    for i in range(nreads-1):
        guess.append(np.median(np.where(foo[i] > np.mean(foo[i]))[0]).astype(int))
    guess = np.array(guess)
    # Guess may be skewed if first file is zeros
    if guess[0] < 0 or guess[0] > subny:
        guess[0] = guess[1] - (guess[2] - guess[1])
    '''
    plt.figure(1)
    plt.clf()
    plt.imshow(cordata[0]*cormask[0], origin='lower', aspect='auto', interpolation='nearest')
    plt.figure(3)
    plt.clf()
    plt.plot(foo[0],'-')
    #for i in range(nreads-1):
        #plt.plot(np.median(cordata[i]*cormask[i], axis=1),'-')
        #plt.plot(foo[i],'-')
    plt.pause(1)
    '''
    # Set limits on the spectrum
    x1 = (guess - fitbghw*expand).astype(int)
    x2 = (guess + fitbghw*expand).astype(int)
    # STEP 3: Fit sky background with out-of-spectra data
    corbg     = np.zeros((cordata.shape))
    for i in range(nreads-1):
        corbg[i], cormask[i] = optspex.fitbg(cordata[i], cormask[i], x1[i], x2[i],
                                     deg=bgdeg, threshold=p3thresh, isrotate=2, isplots=isplots)
    #FINDME: Do not shift data
    #subdata    = cordata
    #submask    = cormask
    #bg         = corbg

    # Decrease resolution, if increased
    if expand > 1:
        subdata = np.zeros((nreads-1,subny,subnx))
        submask = np.zeros((nreads-1,subny,subnx),dtype=int)
        bg        = np.zeros((nreads-1,subny,subnx))
        for i in range(nreads-1):
            subdata[i]    = spni.zoom(cordata[i], zoom=1./expand)
            tempmask    = spni.zoom(cormask[i], zoom=1./expand)
            tempmask[np.where(tempmask >= 0.9)] = 1
            submask[i]    = tempmask.astype(int)
            bg[i]         = spni.zoom(corbg[i], zoom=1./expand)
    else:
        subdata    = cordata
        submask    = cormask
        bg         = corbg
    '''
    # Recalculate slitshift for verification
    slitshift2, shift_values2, yfit2 = calc_slitshift2(subdata[i], np.arange(xwindow[0],xwindow[1]), ywindow, xwindow)
    plt.figure(1, figsize=(12,8))
    plt.clf()
    plt.suptitle('Model Slit Tilts/Shifts')
    plt.plot(shift_values2, yfit2, '.')
    plt.plot(slitshift2, range(np.size(slitshift2)), 'r-', lw=2)
    plt.pause(0.1)
    '''
    # STEP 2: Calculate variance
    #variance    = np.zeros((subny,subnx))
    bgerr         = np.std(bg, axis=1)/np.sqrt(np.sum(submask, axis=1))
    bgerr[np.where(np.isnan(bgerr))] = 0.
    v0         += np.mean(bgerr**2)
    variance    = abs(subdata) / gain + v0
    #variance    = abs(subdata*submask) / gain + v0
    # Perform background subtraction
    subdata    -= bg
    #subdata = (subdata-bg)*submask
    #variance = abs((subdata+bg)*submask) / gain + v0 + bg

    # STEP 4: Extract standard spectrum and its variance
    #guess     = np.argmax(np.median(subdata*submask, axis=2),axis=1)
    foo = np.sum(subdata*submask, axis=2)
    guess = []
    for i in range(nreads-1):
        guess.append(np.median(np.where(foo[i] > np.mean(foo[i]))[0]).astype(int))
    guess     = np.array(guess)
    # Guess may be skewed if first file is zeros
    if guess[0] < 0 or guess[0] > subny:
        guess[0] = guess[1] - (guess[2] - guess[1])
    y1            = guess - spec_width
    y2            = guess + spec_width
    stdspec     = np.zeros((subdata.shape[0],subdata.shape[2]))
    stdvar        = np.zeros((subdata.shape[0],subdata.shape[2]))
    stdbg         = np.zeros((subdata.shape[0],subdata.shape[2]))
    fracMaskReg = np.zeros(nreads-1)
    for i in range(nreads-1):
        stdspec[i]        = np.sum((subdata[i] *submask[i])[y1[i]:y2[i]], axis=0)
        stdvar[i]         = np.sum((variance[i]*submask[i])[y1[i]:y2[i]], axis=0)
        stdbg[i]        = np.sum((bg[i]        *submask[i])[y1[i]:y2[i]], axis=0)
        # Compute fraction of masked pixels within regular spectral extraction window
        numpixels         = 1.*submask[i].size
        fracMaskReg[i]    = (numpixels - submask[i].sum())/numpixels

    if isplots >= 3:
        for i in range(nreads-1):
            plt.figure(1010)
            plt.clf()
            plt.suptitle(str(m) + "," + str(n) + "," + str(i))
            plt.subplot(211)
            plt.imshow(subdata[i]*submask[i], origin='lower', aspect='auto', vmin=-100,vmax=500)
            plt.subplot(212)
            #plt.imshow(submask[i], origin='lower', aspect='auto', vmax=1)
            plt.imshow(bg[i], origin='lower', aspect='auto')
            plt.savefig(eventdir+'/figs/fig1010-'+str(m)+'-'+str(i)+'-Image+Background.png')
            plt.pause(0.1)

    # Compute full scan length
    scannedData = np.sum(subdata*submask, axis=(0,2))
    xmin        = int(np.mean(guess)-spec_width*(nreads-1)/2.)
    xmax        = int(np.mean(guess)+spec_width*(nreads-1)/2.)
    scannedData/= np.median(scannedData[xmin:xmax+1])
    scannedData-= 0.5
    #leftEdge    = np.where(scannedData > 0)/2)[0][0]
    #rightEdge     = np.where(scannedData > 0)/2)[0][-1]
    #yrng        = range(leftEdge-5, leftEdge+5, 1)
    yrng        = range(subny)
    spline        = spi.UnivariateSpline(yrng, scannedData[yrng], k=3, s=0)
    roots         = spline.roots()
    scanHeight    = roots[1]-roots[0]
    '''
    plt.figure(1)
    plt.clf()
    plt.plot(yrng, scannedData[yrng], 'b.')
    plt.plot(yrng, spline(yrng), 'g-')
    '''
    # Extract optimal spectrum with uncertainties
    spectrum    = np.zeros((stdspec.shape))
    specstd     = np.zeros((stdspec.shape))
    fracMaskOpt = np.zeros(nreads-1)
    for i in range(nreads-1):
        #smoothspec    = smooth.medfilt(stdspec[i], window_len)
        spectrum[i], specstd[i], foomask = optspex.optimize(subdata[i,y1[i]:y2[i]], submask[i,y1[i]:y2[i]], bg[i,y1[i]:y2[i]], stdspec[i], gain, v0, p5thresh=p5thresh, p7thresh=p7thresh, fittype=fittype, window_len=window_len, deg=deg, n=m, iread=i, isplots=isplots, eventdir=eventdir)
        # Compute fraction of masked pixels within optimal spectral extraction window
        numpixels         = 1.*foomask.size
        fracMaskOpt[i]    = (submask[i,y1[i]:y2[i]].sum() - foomask.sum())/numpixels

    if isplots >= 3:
        for i in range(nreads-1):
            plt.figure(1011)
            plt.clf()
            plt.suptitle(str(m) + "," + str(n) + "," + str(i))
            #plt.errorbar(ev.wave[m], stdspec, np.sqrt(stdvar), fmt='-')
            plt.errorbar(range(subnx), stdspec[i], np.sqrt(stdvar[i]), fmt='b-')
            plt.errorbar(range(subnx), spectrum[i], specstd[i], fmt='g-')
            plt.savefig(eventdir+'/figs/fig1011-'+str(m)+'-'+str(i)+'-Spectrum.png')
            plt.pause(0.1)
    '''
    plt.figure(5)
    plt.clf()
    plt.suptitle(str(m) + "," + str(n))
    plt.errorbar(range(subnx), stdspec[i], np.sqrt(stdvar[i]), fmt='-')
    plt.errorbar(range(subnx), spectrum[i], specstd[i], fmt='-')
    plt.pause(0.01)
    '''

    return [spectrum, specstd, stdbg, fracMaskReg, fracMaskOpt, scanHeight, m, n]
"""
# Wavelength calibration fitting routine
def wavecal_fit(arc, pix, wl, deg=2, isplots=False):
    '''

    '''
    # Fit Gaussian to each spectral line to determine center
    n_lines = pix.shape[0]
    centers = np.zeros(n_lines)
    for n in range(n_lines):
        #print(m,n)
        x     = np.arange(pix[n,0],pix[n,1]+1)
        y     = arc[pix[n,0]:pix[n,1]+1]
        params, err = g.fitgaussian(y, x, guess=[len(x)/3., np.median(x), y.max()])
        centers[n] = params[1]
        if isplots:
            plt.figure(1)
            plt.clf()
            plt.plot(x, y, 'bo')
            x2 = np.arange(x.min(), x.max(), 0.1)
            plt.plot(x2, g.gaussian(x2, params[0], params[1], params[2]), 'g-')
            plt.pause(0.1)

    #lineCenters.append(centers)
    # Fit wavelength-pixel position dependence
    coeffs = np.polyfit(centers, wl, deg=deg)
    fit    = np.polyval(coeffs, centers)

    return coeffs, fit, centers
"""

# Measure spectrum drift over all frames and all non-destructive reads.
def drift_fit2D(ev, data, validRange=9):
    '''
    Measures the spectrum drift over all frames and all non-destructive reads.

    Parameters
    ----------
    ev            : Event object
    data        : 4D data frames
    preclip     : Ignore first preclip values of spectrum
    postclip    : Ignore last postclip values of spectrum
    width         : Half-width in pixels used when fitting Gaussian
    deg         : Degree of polynomial fit
    validRange    : Trim spectra by +/- pixels to compute valid region of cross correlation

    Returns
    -------
    drift         : Array of measured drift values
    model         : Array of model drift values

    History
    -------
    Written by Kevin Stevenson        January 2017

    '''
    import scipy.interpolate as spi
    import scipy.signal as sps

    #if postclip != None:
    #    postclip = -postclip
    if ev.n_reads > 2:
        istart = 1
    else:
        istart = 0
    drift         = np.zeros((ev.n_files, ev.n_reads-1))
    #model         = np.zeros((ev.n_files, ev.n_reads-1))
    #goodmask    = np.zeros((ev.n_files, ev.n_reads-1),dtype=int)
    for n in range(istart,ev.n_reads-1):
        ref_data = np.copy(data[-1,n])
        ref_data[np.where(np.isnan(ref_data) == True)] = 0
        for m in range(ev.n_files):
            #Trim data to achieve accurate cross correlation without assumptions over interesting region
            #http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
            fit_data    = np.copy(data[m,n,:,validRange:-validRange])
            fit_data[np.where(np.isnan(fit_data) == True)] = 0
            # Cross correlate, result should be 1D
            vals        = sps.correlate2d(ref_data, fit_data, mode='valid').squeeze()
            xx_t        = range(len(vals))
            # Find the B-spline representation
            spline        = spi.splrep(xx_t, vals, k=4)
            # Compute the spline representation of the derivative
            deriv         = spi.splder(spline)
            # Find the maximum with a derivative.
            maximum     = spi.sproot(deriv)
            # Multiple derivatives, take one closest to argmax(vals)
            if len(maximum) > 1:
                #print(m,n,maximum,np.argmax(vals))
                maximum = maximum[np.argmin(np.abs(maximum-np.argmax(vals)))]
            drift[m,n]    = len(vals)/2 - maximum
            '''
            try:
                vals        = np.correlate(ref_spec, fit_spec, mode='valid')
                argmax        = np.argmax(vals)
                subvals     = vals[argmax-width:argmax+width+1]
                params, err = g.fitgaussian(subvals/subvals.max(), guess=[width/5., width*1., 1])
                drift[n,m,i]= len(vals)/2 - params[1] - argmax + width
                goodmask[n,m,i] = 1
            except:
                print('Spectrum ' +str(n)+','+str(m)+','+str(i)+' marked as bad.')
            '''

    return drift#, goodmask

# Measure spectrum drift over all frames and all non-destructive reads.
def drift_fit2(ev, preclip=0, postclip=None, width=3, deg=2, validRange=9, istart=0, iref=-1):
    '''
    Measures the 1D spectrum drift over all frames and all non-destructive reads.

    Parameters
    ----------
    ev            : Event object
    preclip     : Ignore first preclip values of spectrum
    postclip    : Ignore last postclip values of spectrum
    width         : Half-width in pixels used when fitting Gaussian
    deg         : Degree of polynomial fit
    validRange    : Trim spectra by +/- pixels to compute valid region of cross correlation

    Returns
    -------
    drift         : Array of measured drift values
    model         : Array of model drift values

    History
    -------
    Written by Kevin Stevenson        Nov/Dec 2013

    '''
    if postclip != None:
        postclip = -postclip
    drift         = np.zeros((ev.n_files, ev.n_reads-1))
    goodmask    = np.zeros((ev.n_files, ev.n_reads-1),dtype=int)
    ref_spec    = np.copy(ev.spectra[iref,istart,preclip:postclip])
    #ref_spec = spni.zoom(ev.spectra[0,n,preclip:postclip], 5)
    #correlate.py performs better when the mean is not subtracted!!!
    #Zero-mean for cross correlation
    #ref_spec-= np.mean(ref_spec[validRange:-validRange][np.where(np.isnan(ref_spec[validRange:-validRange]) == False)])
    ref_spec[np.where(np.isnan(ref_spec) == True)] = 0
    nx            = len(ref_spec)
    for m in range(ev.n_files):
        for n in range(istart,ev.n_reads-1):
            fit_spec    = np.copy(ev.spectra[m,n,preclip:postclip])
            #Trim data to achieve accurate cross correlation without assumptions over interesting region
            #http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
            fit_spec    = fit_spec[validRange:-validRange]
            #correlate.py performs better when the mean is not subtracted!!!
            #fit_spec     -= np.mean(fit_spec[np.where(np.isnan(fit_spec) == False)])
            fit_spec[np.where(np.isnan(fit_spec) == True)] = 0
            try:
                vals        = np.correlate(ref_spec, fit_spec, mode='valid')
                argmax        = np.argmax(vals)
                subvals     = vals[argmax-width:argmax+width+1]
                params, err = g.fitgaussian(subvals/subvals.max(), guess=[width/5., width*1., 1])
                drift[m,n]= len(vals)/2 - params[1] - argmax + width
                #drift[n,m,i]    = nx//2. - argmax - params[1] + width
                '''
                vals        = np.correlate(ref_spec, fit_spec, mode='valid')
                params, err = g.fitgaussian(vals, guess=[width/5., width*1., vals.max()-np.median(vals)])
                drift[n,m,i]    = len(vals)/2 - params[1]
                #FINMDE
                plt.figure(4)
                plt.clf()
                plt.plot(vals/vals.max(),'o')
                ymin,ymax=plt.ylim()
                plt.vlines(params[1], ymin, ymax, colors='k')
                plt.figure(5)
                plt.clf()
                plt.plot(range(nx),ref_spec,'-k')
                plt.plot(range(validRange,nx-validRange), fit_spec,'-r')
                plt.pause(0.1)
                '''
                goodmask[m,n] = 1
            except:
                print('Spectrum ' +str(m)+','+str(n)+' marked as bad.')


    return drift, goodmask

# Measure spectrum drift over all frames and all non-destructive reads.
def drift_fit(ev, preclip=0, postclip=None, width=3, deg=2, validRange=9):
    '''
    Measures the spectrum drift over all frames and all non-destructive reads.

    Parameters
    ----------
    ev            : Event object
    preclip     : Ignore first preclip values of spectrum
    postclip    : Ignore last postclip values of spectrum
    width         : Half-width in pixels used when fitting Gaussian
    deg         : Degree of polynomial fit
    validRange    : Trim spectra by +/- pixels to compute valid region of cross correlation

    Returns
    -------
    drift         : Array of measured drift values
    model         : Array of model drift values

    History
    -------
    Written by Kevin Stevenson        Nov/Dec 2013

    '''
    if postclip != None:
        postclip = -postclip
    drift         = np.zeros((ev.n_spec, ev.n_files, ev.n_reads-1))
    model         = np.zeros((ev.n_spec, ev.n_files, ev.n_reads-1))
    goodmask    = np.zeros((ev.n_spec, ev.n_files, ev.n_reads-1),dtype=int)
    if ev.n_reads > 2:
        #print('WARNING: Marking all first reads as bad.')
        istart = 1
    else:
        #print('Using first reads.')
        istart = 0
    for n in range(ev.n_spec):
        ref_spec = np.copy(ev.spectra[istart,n,istart,preclip:postclip])
        #ref_spec = spni.zoom(ev.spectra[0,n,preclip:postclip], 5)
        #correlate.py performs better when the mean is not subtracted!!!
        #Zero-mean for cross correlation
        #ref_spec-= np.mean(ref_spec[validRange:-validRange][np.where(np.isnan(ref_spec[validRange:-validRange]) == False)])
        ref_spec[np.where(np.isnan(ref_spec) == True)] = 0
        nx         = len(ref_spec)
        for m in range(ev.n_files):    #FINDME
        #for m in [13]:
            for i in range(istart,ev.n_reads-1):
                fit_spec    = np.copy(ev.spectra[m,n,i,preclip:postclip])
                #Trim data to achieve accurate cross correlation without assumptions over interesting region
                #http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
                fit_spec    = fit_spec[validRange:-validRange]
                #correlate.py performs better when the mean is not subtracted!!!
                #fit_spec     -= np.mean(fit_spec[np.where(np.isnan(fit_spec) == False)])
                fit_spec[np.where(np.isnan(fit_spec) == True)] = 0
                try:
                    vals        = np.correlate(ref_spec, fit_spec, mode='valid')
                    argmax        = np.argmax(vals)
                    subvals     = vals[argmax-width:argmax+width+1]
                    params, err = g.fitgaussian(subvals/subvals.max(), guess=[width/5., width*1., 1])
                    drift[n,m,i]= len(vals)/2 - params[1] - argmax + width
                    #drift[n,m,i]    = nx//2. - argmax - params[1] + width
                    '''
                    vals        = np.correlate(ref_spec, fit_spec, mode='valid')
                    params, err = g.fitgaussian(vals, guess=[width/5., width*1., vals.max()-np.median(vals)])
                    drift[n,m,i]    = len(vals)/2 - params[1]
                    #FINMDE
                    plt.figure(4)
                    plt.clf()
                    plt.plot(vals/vals.max(),'o')
                    ymin,ymax=plt.ylim()
                    plt.vlines(params[1], ymin, ymax, colors='k')
                    plt.figure(5)
                    plt.clf()
                    plt.plot(range(nx),ref_spec,'-k')
                    plt.plot(range(validRange,nx-validRange), fit_spec,'-r')
                    plt.pause(0.1)
                    '''
                    goodmask[n,m,i] = 1
                except:
                    print('Spectrum ' +str(n)+','+str(m)+','+str(i)+' marked as bad.')

            isbadframes = True
            while isbadframes == True:
                igood         = np.where(goodmask[n,m])[0]
                if len(igood) >= deg:
                    #Fit model to drift
                    coeffs        = np.polyfit(np.arange(ev.n_reads-1,dtype=int)[igood], drift[n,m,igood], deg=deg)
                    model[n,m]    = np.polyval(coeffs, range(ev.n_reads-1))
                    #Look for 5*sigma outliers
                    residuals     = (drift[n,m]-model[n,m])[igood]
                    stdres        = np.std(residuals)
                    stdevs        = np.abs(residuals) / stdres
                    loc         = np.argmax(stdevs)
                    if np.abs(residuals[loc]) >= 5*stdres:
                        print('Spectrum ' +str(n)+','+str(m)+','+str(igood[loc])+' marked as bad.')
                        goodmask[n,m,igood[loc]] = 0
                    else:
                        isbadframes = False
                else:
                    isbadframes = False
            #print(np.where(goodmask[n]==0)[0])

    return drift, model, goodmask

# Determine observation time in JD
def date_obs(hdr):
    '''
    Determines observation time in JD.

    Parameters
    ----------
    hdr         : Header file

    Returns
    -------
    jd            : Julian day

    History
    -------
    Written by Kevin Stevenson        June 2012

    '''
    jd = 2400000.5 + 0.5*(hdr['EXPSTART'] + hdr['EXPEND'])

    return jd

def checkDates(directory):
    '''

    '''
    obj_list = []
    for fname in os.listdir(directory):
        obj_list.append(directory +'/'+ fname)
    obj_list = sn.sort_nicely(obj_list)

    jd = []
    rootname = []
    for fname in obj_list:
        hdr = fits.getheader(fname)
        jd.append(hst.date_obs(hdr))
        rootname.append(hdr['ROOTNAME'])
    one = np.ones(len(jd))
    mjd = np.array(jd) - 2456600    #np.floor(jd[0])

    plt.figure(10)
    #plt.clf()
    plt.plot(mjd,one,'o')

    return
