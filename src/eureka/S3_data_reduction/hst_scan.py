import numpy as np
from astropy.io import fits
try:
    import image_registration as imr
    imported_image_registration = True
except ModuleNotFoundError:
    imported_image_registration = False
from ..lib import centroid


def imageCentroid(filenames, guess, trim, ny, CRPIX1, CRPIX2, POSTARG1,
                  POSTARG2, meta, log):
    '''Calculate centroid for a list of direct images from HST.

    Parameters
    ----------
    filenames : list
        List of direct image filenames
    guess : array_like
        The initial guess of the position of the star.  Has the form
        (x, y) of the guess center.
    trim : int
        If trim!=0, trims the image in a box of 2*trim pixels around
        the guess center. Must be !=0 for 'col' method.
    ny : int
        The value of NAXIS2
    CRPIX1 : float
        The value of CRPIX1 in the main FITS header
    CRPIX2 : float
        The value of CRPIX2 in the main FITS header
    POSTARG1 : float
        The value of POSTARG1 in the science FITS header
    POSTARG2 : float
        The value of POSTARG2 in the science FITS header
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    centers : list
        Centroids
    '''
    nfiles = len(filenames)
    centers = []

    # Swap the x-y order for the other, older code which used to have (y,x)
    guess = guess[::-1]

    for i in range(nfiles):
        image = fits.getdata(filenames[i].rstrip())
        calhdr0 = fits.getheader(filenames[i].rstrip(), 0)
        calhdr1 = fits.getheader(filenames[i].rstrip(), 1)
        # Calculate centroid, correct for difference in image size, if any
        centers.append(centroid.ctrgauss(image, guess=guess, trim=trim) -
                       (image.shape[0]-ny)/2.)
        xoffset = (CRPIX1 - calhdr1['CRPIX1'] +
                   (POSTARG1[i] - calhdr0['POSTARG1'])/0.135)
        yoffset = (CRPIX2 - calhdr1['CRPIX2'] +
                   (POSTARG2[i] - calhdr0['POSTARG2'])/0.121)
        centers[i][0] += yoffset
        centers[i][1] += xoffset
        log.writelog(f"Adding {np.round(xoffset, 3)}, {np.round(yoffset, 3)}"
                     f" pixels to x, y centroid position for integrations "
                     f"related to staring-mode image #{i}.",
                     mute=(not meta.verbose))
        log.writelog(f"Final centroid is {np.round(centers[i][0], 3)},"
                     f" {np.round(centers[i][1], 3)}",
                     mute=(not meta.verbose))
    return centers


def groupFrames(dates):
    '''Group frames by HST orbit and batch number.

    Parameters
    ----------
    dates : ndarray (1D)
        Time in days

    Returns
    -------
    framenum : ndarray (1D)
        The frame numbers.
    batchnum : ndarray (1D)
        The batch numbers.
    orbitnum : ndarray (1D)
        The orbit numbers.
    '''
    n_frames = len(dates)
    framenum = np.zeros(n_frames)
    batchnum = np.zeros(n_frames)
    orbitnum = np.zeros(n_frames)
    frame = 0
    batch = 0
    orbit = 0
    framegap = np.median(np.ediff1d(dates))
    orbitgap = np.max(np.ediff1d(dates))
    for i in range(1, n_frames):
        if dates[i]-dates[i-1] < 2*framegap:
            # New frames, same batch, same orbit
            frame += 1
        elif dates[i]-dates[i-1] > 0.5*orbitgap:
            # Reset frame, new batch, rest orbit
            frame = 0
            batch = 0
            orbit += 1
        else:  # dates[i]-dates[i-1] > 3*exptime[i]/86400.:
            # Reset frame, new batch, same orbit
            frame = 0
            batch += 1
        framenum[i] = frame
        batchnum[i] = batch
        orbitnum[i] = orbit

    return framenum, batchnum, orbitnum


def calcTrace(x, centroid, grism):
    '''Calculates the WFC3 trace given the position of the direct image in
    physical pixels.

    Parameters
    ----------
    x : ndarray
        Physical pixel values along dispersion direction over which the trace
        is calculated
    centroid : list
        [y,x] pair describing the centroid of the direct image
    grism : str
        The grism being used.

    Returns
    -------
    y : ndarray
        Computed trace.
    '''
    yref, xref = centroid

    if not isinstance(yref, float):
        yref = yref[:, np.newaxis]
        x = x[np.newaxis]

    if grism == 'G141':
        # WFC3-2009-17.pdf
        # Table 1: Field dependent trace descriptions for G141.
        # Term      a0           a1(X)         a2(Y)         a3(X^2)
        #           a4(X*Y)      a5(Y^2)
        DYDX_A_0 = [1.96882E+00, 9.09159E-05,  -1.93260E-03]
        DYDX_A_1 = [1.04275E-02, -7.96978E-06, -2.49607E-06, 1.45963E-09,
                    1.39757E-08, 4.84940E-10]
    elif grism == 'G102':
        # WFC3-2009-18.pdf
        # Table 1: Field dependent trace descriptions for G102.
        # Term      a0            a1(X)         a2(Y)         a3(X^2)
        #           a4(X*Y)       a5(Y^2)
        DYDX_A_0 = [-3.55018E-01, 3.28722E-05,  -1.44571E-03]
        DYDX_A_1 = [1.42852E-02,  -7.20713E-06, -2.42542E-06, 1.18294E-09,
                    1.19634E-08,  6.17274E-10]
    else:
        print("Unknown filter/grism: " + grism)
        return 0

    DYDX_0 = DYDX_A_0[0] + DYDX_A_0[1]*xref + DYDX_A_0[2]*yref
    DYDX_1 = (DYDX_A_1[0] + DYDX_A_1[1]*xref + DYDX_A_1[2]*yref +
              DYDX_A_1[3]*xref**2 + DYDX_A_1[4]*xref*yref +
              DYDX_A_1[5]*yref**2)

    y = DYDX_0 + DYDX_1*(x-xref) + yref

    return y


def calibrateLambda(x, centroid, grism):
    '''Calculates the wavelength solution for WFC3 observations.

    Parameters
    ----------
    x : ndarray
        Physical pixel values along dispersion direction over which the trace
        is calculated
    centroid : list
        [y,x] pair describing the centroid of the direct image
    grism : str
        The grism being used.

    Returns
    -------
    y : ndarray
        Computed wavelength values
    '''
    yref, xref = centroid

    if not isinstance(yref, float):
        yref = yref[:, np.newaxis]
        x = x[np.newaxis]

    if grism == 'G141':
        # WFC3-2009-17.pdf
        # Table 5: Field dependent wavelength solution for G141.
        # Term      a0           a1(X)         a2(Y)        a3(X^2)
        #           a4(X*Y)      a5(Y^2)
        DLDP_A_0 = [8.95431E+03, 9.35925E-02,  0.0,         0.0,
                    0.0,         0.0]
        DLDP_A_1 = [4.51423E+01, 3.17239E-04,  2.17055E-03, -7.42504E-07,
                    3.48639E-07, 3.09213E-07]
    elif grism == 'G102':
        # WFC3-2009-18.pdf
        # Table 5: Field dependent wavelength solution for G102.
        # FINDME: y^2 term not given in Table 5, assuming 0.
        # Term      a0            a1(X)        a2(Y)        a3(X^2)
        #           a4(X*Y)       a5(Y^2)
        DLDP_A_0 = [6.38738E+03,  4.55507E-02, 0.0]
        DLDP_A_1 = [2.35716E+01,  3.60396E-04, 1.58739E-03, -4.25234E-07,
                    -6.53726E-08, 0.0]
    else:
        print("Unknown filter/grism: " + grism)
        return 0

    DLDP_0 = DLDP_A_0[0] + DLDP_A_0[1]*xref + DLDP_A_0[2]*yref
    DLDP_1 = (DLDP_A_1[0] + DLDP_A_1[1]*xref + DLDP_A_1[2]*yref +
              DLDP_A_1[3]*xref**2 + DLDP_A_1[4]*xref*yref +
              DLDP_A_1[5]*yref**2)

    y = DLDP_0 + DLDP_1*(x-xref) + yref

    return y


def makeflats(flatfile, wave, xwindow, ywindow, flatoffset, n_spec, ny, nx,
              sigma=5, isplots=0):
    '''Makes master flatfield image and new mask for WFC3 data.

    Parameters
    ----------
    flatfile : list
        List of files containing flatfiles images.
    wave : ndarray
        Wavelengths.
    xwindow : list
        Array containing image limits in wavelength direction.
    ywindow : list
        Array containing image limits in spatial direction.
    n_spec : int
        Number of spectra.
    sigma : float
        Sigma rejection level.

    Returns
    -------
    flat_master : list
        Single master flatfield image.
    mask_master : list
        Single bad-pixel mask image. Boolean, where True values
        should be masked.
    '''
    # Read in flat frames
    hdulist = fits.open(flatfile)
    flat_mhdr = hdulist[0].header
    wmin = float(flat_mhdr['wmin'])/1e4
    wmax = float(flat_mhdr['wmax'])/1e4
    # nx = flat_mhdr['naxis1']
    # ny = flat_mhdr['naxis2']
    # Build flat field, only compute for subframe
    flat_master = []
    mask_master = []
    for i in range(n_spec):
        # Read and assemble flat field
        # Select windowed region containing the data
        x = (wave[i] - wmin)/(wmax - wmin)

        ylower = int(ywindow[i][0]+flatoffset[i][0])
        yupper = int(ywindow[i][1]+flatoffset[i][0])
        xlower = int(xwindow[i][0]+flatoffset[i][1])
        xupper = int(xwindow[i][1]+flatoffset[i][1])
        # flat_window += hdulist[j].data[ylower:yupper,xlower:xupper]*x**j

        if flatfile[-19:] == 'sedFFcube-both.fits':
            # sedFFcube-both
            flat_window = hdulist[1].data[ylower:yupper, xlower:xupper]
            for j in range(2, len(hdulist)):
                flat_window += hdulist[j].data[ylower:yupper,
                                               xlower:xupper]*x**(j-1)
        else:
            # WFC3.IR.G141.flat.2 OR WFC3.IR.G102.flat.2
            flat_window = hdulist[0].data[ylower:yupper, xlower:xupper]
            for j in range(1, len(hdulist)):
                flat_window += hdulist[j].data[ylower:yupper,
                                               xlower:xupper]*x**j

        # Initialize bad-pixel mask
        mask_window = np.zeros(flat_window.shape, dtype=bool)

        flat_norm = flat_window

        if sigma is not None and sigma > 0:
            # Reject points from flat and flag them in the mask
            # Points that are outliers, do this for the high and low
            # sides separately
            # 1. Reject points < 0
            index = np.nonzero(flat_norm < 0)
            flat_norm[index] = 1.
            mask_window[index] = True
            # 2. Reject outliers from low side
            ilow = np.nonzero(flat_norm < 1)
            # Make distribution symetric about 1
            dbl = np.concatenate((flat_norm[ilow], 1+(1-flat_norm[ilow])))
            # MAD
            std = 1.4826*np.median(np.abs(dbl - np.median(dbl)))
            ibadpix = np.nonzero((1 - flat_norm[ilow]) > sigma * std)[0]
            flat_norm[ilow[0][ibadpix], ilow[1][ibadpix]] = 1.
            mask_window[ilow[0][ibadpix], ilow[1][ibadpix]] = True
            # 3. Reject outliers from high side
            ihi = np.nonzero(flat_norm > 1)
            # Make distribution symetric about 1
            dbl = np.concatenate((flat_norm[ihi], 2-flat_norm[ihi]))
            # MAD
            std = 1.4826*np.median(np.abs(dbl - np.median(dbl)))
            ibadpix = np.nonzero((flat_norm[ihi] - 1) > sigma * std)[0]
            flat_norm[ihi[0][ibadpix], ihi[1][ibadpix]] = 1.
            mask_window[ihi[0][ibadpix], ihi[1][ibadpix]] = True

        # Put the subframes back in the full frames
        flat_new = np.ones((ny, nx), dtype=np.float32)
        mask = np.ones((ny, nx), dtype=bool)
        flat_new[ywindow[i][0]:ywindow[i][1],
                 xwindow[i][0]:xwindow[i][1]] = flat_norm
        mask[ywindow[i][0]:ywindow[i][1],
             xwindow[i][0]:xwindow[i][1]] = mask_window
        flat_master.append(flat_new)
        mask_master.append(mask)

    return flat_master, mask_master


def makeBasicFlats(flatfile, xwindow, ywindow, flatoffset, ny, nx, sigma=5,
                   isplots=0):
    '''Makes master flatfield image (with no wavelength correction) and new
    mask for WFC3 data.

    Parameters
    ----------
    flatfile : list
        List of files containing flatfiles images
    xwindow : ndarray
        Array containing image limits in wavelength direction
    ywindow : ndarray
        Array containing image limits in spatial direction
    n_spec : int
        Number of spectra
    sigma : float
        Sigma rejection level

    Returns
    -------
    flat_master : list
        Single master flatfield image
    mask_master : list
        Single bad-pixel mask image. Boolean, where True values
        should be masked.
    '''
    # Read in flat frames
    hdulist = fits.open(flatfile)
    # flat_mhdr = hdulist[0].header
    # wmin = float(flat_mhdr['wmin'])/1e4
    # wmax = float(flat_mhdr['wmax'])/1e4
    # nx = flat_mhdr['naxis1']
    # ny = flat_mhdr['naxis2']
    # Build flat field, only compute for subframe
    flat_master = []
    mask_master = []
    # Read and assemble flat field
    # Select windowed region containing the data
    # x = (wave[i] - wmin)/(wmax - wmin)
    ylower = int(ywindow[0]+flatoffset[0])
    yupper = int(ywindow[1]+flatoffset[0])
    xlower = int(xwindow[0]+flatoffset[1])
    xupper = int(xwindow[1]+flatoffset[1])
    if flatfile[-19:] == 'sedFFcube-both.fits':
        # sedFFcube-both
        flat_window = hdulist[1].data[ylower:yupper, xlower:xupper]
    else:
        # WFC3.IR.G141.flat.2
        flat_window = hdulist[0].data[ylower:yupper, xlower:xupper]

    # Initialize bad-pixel mask
    mask_window = np.zeros(flat_window.shape, dtype=bool)
    flat_norm = flat_window

    if sigma is not None and sigma > 0:
        # Reject points from flat and flag them in the mask
        # Points that are outliers, do this for the high and low
        # sides separately
        # 1. Reject points < 0
        index = np.nonzero(flat_norm < 0)
        flat_norm[index] = 1.
        mask_window[index] = True
        # 2. Reject outliers from low side
        ilow = np.nonzero(flat_norm < 1)
        # Make distribution symetric about 1
        dbl = np.concatenate((flat_norm[ilow], 1+(1-flat_norm[ilow])))
        # MAD
        std = 1.4826*np.median(np.abs(dbl - np.median(dbl)))
        ibadpix = np.nonzero((1 - flat_norm[ilow]) > sigma * std)[0]
        flat_norm[ilow[0][ibadpix], ilow[1][ibadpix]] = 1.
        mask_window[ilow[0][ibadpix], ilow[1][ibadpix]] = True
        # 3. Reject outliers from high side
        ihi = np.nonzero(flat_norm > 1)
        # Make distribution symetric about 1
        dbl = np.concatenate((flat_norm[ihi], 2-flat_norm[ihi]))
        # MAD
        std = 1.4826*np.median(np.abs(dbl - np.median(dbl)))
        ibadpix = np.nonzero((flat_norm[ihi] - 1) > sigma * std)[0]
        flat_norm[ihi[0][ibadpix], ihi[1][ibadpix]] = 1.
        mask_window[ihi[0][ibadpix], ihi[1][ibadpix]] = True

    # Put the subframes back in the full frames
    flat_new = np.ones((ny, nx), dtype=np.float32)
    mask = np.ones((ny, nx), dtype=bool)
    flat_new[ywindow[0]:ywindow[1], xwindow[0]:xwindow[1]] = flat_norm
    mask[ywindow[0]:ywindow[1], xwindow[0]:xwindow[1]] = mask_window
    flat_master.append(flat_new)
    mask_master.append(mask)

    return flat_master, mask_master


def calcDrift2D(im1, im2, n):
    """Calculates 2D drift of im2 with respect to im1 for diagnostic use,
    to align the images, and/or for decorrelation.

    Parameters
    ----------
    im1 : ndarray (2D)
        The reference image.
    im2 : ndarray (2D)
        The current image.
    n : int
        The current integration number.

    Returns
    -------
    drift2D : list
        The x and y offset of im2 with respect to im1.
    n : int
        The current integration number.

    Raises
    ------
    ModuleNotFoundError
        image_registration wasn't installed with Eureka.
    """
    if not imported_image_registration:
        raise ModuleNotFoundError('The image-registration package was not '
                                  'installed with Eureka and is required for '
                                  'HST analyses.\nYou can install all '
                                  'HST-related dependencies with '
                                  '`pip install eureka-bang[hst]`')
    drift2D = imr.chi2_shift(im1, im2, boundary='constant', nthreads=1,
                             zeromean=False, return_error=False)
    return drift2D, n
