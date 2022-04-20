import numpy as np
from astropy.io import fits
import scipy.interpolate as spi
import scipy.signal as sps
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
try:
    import image_registration as imr
    imported_image_registration = True
except ModuleNotFoundError as e:
    imported_image_registration = False
from ..lib import gaussian as g
from ..lib import smooth, centroid, smoothing


def imageCentroid(filenames, guess, trim, ny, CRPIX1, CRPIX2, POSTARG1, POSTARG2):
    '''
    Calculate centroid for a list of direct images.

    Parameters
    ----------
    filenames     : List of direct image filenames
    guess         : Paired list, centroid guess
    trim        : Trim image when calculating centroid
    ny          : The value of NAXIS2
    CRPIX1: The value of CRPIX1 in the main FITS header
    CRPIX2:  The value of CRPIX2 in the main FITS header
    POSTARG1:    The value of POSTARG1 in the science FITS header
    POSTARG2:   The value of POSTARG2 in the science FITS header

    Returns
    -------
    center        : Centroids

    History
    -------
    Written by Kevin Stevenson        November 2013
    Added IRSUB256                    March 2016
    Updated for Eureka by Taylor Bell   December 8, 2021
    '''
    nfiles = len(filenames)
    centers     = []
    #images         = []
    for i in range(nfiles):
        #images.append(fits.getdata(filenames[i].rstrip()))
        image = fits.getdata(filenames[i].rstrip())
        #hdr0 = fits.getheader(filenames[i],0)
        #hdr1 = fits.getheader(filenames[i],1)
        calhdr0 = fits.getheader(filenames[i].rstrip(),0)
        calhdr1 = fits.getheader(filenames[i].rstrip(),1)
        #Calculate centroid, correct for difference in image size, if any
        #centers.append(centroid.ctrgauss(images[i], guess=guess, trim=trim) - (images[i].shape[0]-ny)/2.)
        centers.append(centroid.ctrgauss(image, guess=guess, trim=trim) - (image.shape[0]-ny)/2.)
        xoffset    = CRPIX1 - calhdr1['CRPIX1'] + (POSTARG1[i] - calhdr0['POSTARG1'])/0.135
        yoffset    = CRPIX2 - calhdr1['CRPIX2'] + (POSTARG2[i] - calhdr0['POSTARG2'])/0.121
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
    return centers#, images

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
    hdulist     = fits.open(flatfile)
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
def calcDrift2D(im1, im2, m, n, n_files):
    if not imported_image_registration:
        raise ModuleNotFoundError("The image-registration package was not installed with Eureka and is required for HST analyses.\n"+
                                  "You can install all HST-related dependencies with `pip install .[hst]`")
    drift2D = imr.chi2_shift(im1, im2, boundary='constant', nthreads=1,
                             zeromean=False, return_error=False)
    return (drift2D, m, n)

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
