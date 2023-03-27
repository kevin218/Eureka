import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.ndimage as spn
from astropy.stats import sigma_clip
from tqdm import tqdm
import warnings

from ..lib import gaussian as g
from ..lib import smooth
from . import plots_s3


def standard_spectrum(data, apdata, aperr):
    """Compute the standard box spectrum.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    apdata : ndarray
        The pixel values in the aperture region.
    aperr : ndarray
        The noise values in the aperture region.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object in which the spectrum data will stored.
    """
    data['stdspec'] = (['time', 'x'], np.sum(apdata, axis=1))
    data['stdvar'] = (['time', 'x'], np.sum(aperr ** 2, axis=1))
    data['stdspec'].attrs['flux_units'] = \
        data.flux.attrs['flux_units']
    data['stdspec'].attrs['time_units'] = \
        data.flux.attrs['time_units']
    data['stdvar'].attrs['flux_units'] = \
        data.flux.attrs['flux_units']
    data['stdvar'].attrs['time_units'] = \
        data.flux.attrs['time_units']

    return data


def profile_poly(subdata, mask, deg=3, threshold=10, isplots=0):
    '''Construct normalized spatial profile using polynomial fits along the
    wavelength direction.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask.
    deg : int; optional
        Polynomial degree, defaults to 3.
    threshold : float; optional
        Sigma threshold for outlier rejection while constructing
        spatial profile, defaults to 10.
    isplots : int; optional
        The plotting verbosity. Defaults to 0.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.
    '''
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))
    maxiter = nx
    for j in range(ny):
        nobadpixels = False
        iternum = 0
        while not nobadpixels and iternum < maxiter:
            # Do not want to alter original data
            dataslice = np.copy(subdata[j])
            # Replace masked points with median of nearby points
            for ind in np.where(submask[j] == 0)[0]:
                dataslice[ind] = \
                    np.median(dataslice[np.max((0, ind-10)):ind+11])

            # Smooth each row
            coeffs = np.polyfit(range(nx), dataslice, deg)
            model = np.polyval(coeffs, range(nx))
            if isplots == 7:
                plt.figure(3703)
                plt.clf()
                plt.suptitle(str(j) + "," + str(iternum))
                plt.plot(dataslice, 'ro')
                plt.plot(dataslice*submask[j], 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.1)

            # Calculate residuals and number of sigma from the model
            residuals = submask[j]*(dataslice - model)
            stdevs = np.abs(residuals) / np.std(residuals)
            # Find worst data point
            loc = np.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                nobadpixels = False
                submask[j, loc] = 0
            else:
                nobadpixels = True  # exit while loop
            iternum += 1

        profile[j] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for '
                  'dataslice ' + str(j))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_smooth(subdata, mask, threshold=10, window_len=21,
                   windowtype='hanning', isplots=0):
    '''Construct normalized spatial profile using a smoothing function.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask.
    threshold : float; optional
        Sigma threshold for outlier rejection while constructing
        spatial profile.
    window_len : int; optional
        The dimension of the smoothing window.
    windowtype : str; optional
        UNUSED. One of {'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'}. The type of window. A flat window will
        produce a moving average smoothing. Defaults to 'hanning'.
    isplots : int; optional
        The plotting verbosity. Defaults to 0.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.
    '''
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))
    maxiter = nx
    for j in range(ny):
        # Check for good pixels in row
        if np.sum(submask[j]) > 0:
            nobadpixels = False
            iternum = 0
            maxiter = np.sum(submask[j])
            while not nobadpixels and iternum < maxiter:
                # Do not want to alter original data
                dataslice = np.copy(subdata[j])
                # Replace masked points with median of nearby points
                # dataslice[np.where(submask[j] == 0)] = 0
                # FINDME: Code below appears to be effective, but is slow for
                # lots of masked points
                for ind in np.where(submask[j] == 0)[0]:
                    dataslice[ind] = \
                        np.median(dataslice[np.max((0, ind-10)):ind+11])

                # Smooth each row
                # model = smooth.smooth(dataslice, window_len=window_len,
                #                       window=windowtype)
                model = smooth.medfilt(dataslice, window_len)
                if isplots == 7:
                    plt.figure(3703)
                    plt.clf()
                    plt.suptitle(str(j) + "," + str(iternum))
                    plt.plot(dataslice, 'ro')
                    plt.plot(dataslice*submask[j], 'bo')
                    plt.plot(model, 'g-')
                    plt.pause(0.1)

                # Calculate residuals and number of sigma from the model
                igoodmask = np.where(submask[j] == 1)[0]
                residuals = submask[j]*(dataslice - model)
                stdevs = (np.abs(residuals[igoodmask]) /
                          np.std(residuals[igoodmask]))
                # Find worst data point
                loc = np.argmax(stdevs)
                # Mask data point if > threshold
                if stdevs[loc] > threshold:
                    nobadpixels = False
                    submask[j, igoodmask[loc]] = 0
                else:
                    nobadpixels = True  # exit while loop
                iternum += 1
            # Copy model slice to profile
            profile[j] = model
            if iternum == maxiter:
                print('WARNING: Max number of iterations reached for '
                      'dataslice ' + str(j))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    with warnings.catch_warnings():
        # Ignore warnings about columns that are completely masked
        warnings.filterwarnings("ignore", "invalid value encountered in")
        profile /= np.nansum(profile, axis=0)

    return profile


def profile_meddata(data, mask, meddata, threshold=10, isplots=0):
    '''Construct normalized spatial profile using median of all data frames.

    Parameters
    ----------
    data : ndarray
        Unused. Image data.
    mask : ndarray
        Unused. Outlier mask.
    meddata : ndarray
        The median of all data frames.
    threshold : float; optional
        Unused. Sigma threshold for outlier rejection while constructing
        spatial profile. Defaults to 10.
    isplots : int; optional
        Unused. The plotting verbosity. Defaults to 0.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.
    '''
    # profile = np.copy(meddata*mask)
    profile = np.copy(meddata)
    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.sum(profile, axis=0)

    return profile


# Construct normalized spatial profile using wavelets
def profile_wavelet(subdata, mask, wavelet, numlvls, isplots=0):
    '''This function performs 1D image denoising using BayesShrink
    soft thresholding.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask.
    wavelet : Wavelet object or name string
        qWavelet to use
    numlvls : int
        Decomposition levels to consider (must be >= 0).
    isplots : int; optional
        The plotting verbosity. Defaults to 0.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.

    References
    ----------
    Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and
    Compression", 2000
    '''
    import pywt
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))

    for j in range(ny):
        # Perform wavelet decomposition
        dec = pywt.wavedec(subdata[j], wavelet)
        # Estimate noise variance
        noisevar = np.inf
        for i in range(-1, -numlvls-1, -1):
            noisevar = np.min([(np.median(np.abs(dec[i]))/0.6745)**2,
                               noisevar])
        # At each level of decomposition...
        for i in range(-1, -numlvls-1, -1):
            # Estimate variance at level i then compute the threshold value
            # sigmay2 = np.mean(dec[i]*dec[i])
            # sigmax = np.sqrt(np.max([sigmay2-noisevar, 0]))
            threshold = np.max(np.abs(dec[i]))
            # if sigmax == 0 or i == -1:
            #     threshold = np.max(np.abs(dec[i]))
            # else:
            #     threshold = noisevar/sigmax
            # Compute less noisy coefficients by applying soft thresholding
            dec[i] = map(lambda x: pywt.thresholding.soft(x, threshold),
                         dec[i])

        profile[j] = pywt.waverec(dec, wavelet)[:nx]
        if isplots == 7:
            plt.figure(3703)
            plt.clf()
            plt.suptitle(str(j))
            plt.plot(subdata[j], 'ro')
            plt.plot(subdata[j]*submask[j], 'bo')
            plt.plot(profile[j], 'g-')
            plt.pause(0.1)

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_wavelet2D(subdata, mask, wavelet, numlvls, isplots=0):
    '''Construct normalized spatial profile using wavelets

    This function performs 2D image denoising using BayesShrink
    soft thresholding.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask.
    wavelet : Wavelet object or name string
        qWavelet to use
    numlvls : int
        Decomposition levels to consider (must be >= 0).
    isplots : int; optional
        The plotting verbosity. Defaults to 0.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.

    References
    ----------
    Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and
    Compression", 2000
    '''
    import pywt
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))

    # Perform wavelet decomposition
    dec = pywt.wavedec2(subdata, wavelet)
    # Estimate noise variance
    noisevar = np.inf
    for i in range(-1, -numlvls-1, -1):
        noisevar = np.min([(np.median(np.abs(dec[i]))/0.6745)**2, noisevar])
    # At each level of decomposition...
    for i in range(-1, -numlvls-1, -1):
        # Estimate variance at level i then compute the threshold value
        # sigmay2 = np.mean((dec[i][0]*dec[i][0]+dec[i][1]*dec[i][1] +
        #                    dec[i][2]*dec[i][2])/3.)
        # sigmax = np.sqrt(np.max([sigmay2-noisevar, 0]))
        threshold = np.max(np.abs(dec[i]))
        # if sigmax == 0:
        #     threshold = np.max(np.abs(dec[i]))
        # else:
        #     threshold = noisevar/sigmax
        # Compute less noisy coefficients by applying soft thresholding
        dec[i] = map(lambda x: pywt.thresholding.soft(x, threshold), dec[i])

    profile = pywt.waverec2(dec, wavelet)[:ny, :nx]
    if isplots == 7:
        plt.figure(3703)
        plt.clf()
        # plt.suptitle(str(j) + "," + str(iternum))
        plt.plot(subdata[ny/2], 'ro')
        plt.plot(subdata[ny/2]*submask[ny/2], 'bo')
        plt.plot(profile[ny/2], 'g-')
        plt.figure(3704)
        plt.clf()
        # plt.suptitle(str(j) + "," + str(iternum))
        plt.plot(subdata[:, nx/2], 'ro')
        plt.plot(subdata[:, nx/2]*submask[:, nx/2], 'bo')
        plt.plot(profile[:, nx/2], 'g-')
        plt.pause(0.1)

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_gauss(subdata, mask, threshold=10, guess=None, isplots=0):
    '''Construct normalized spatial profile using Gaussian smoothing function.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask.
    threshold : float; optional
        Sigma threshold for outlier rejection while constructing
        spatial profile. Defaults to 10.
    guess : list; optional
        UNUSED. The initial guess for the Gaussian parameters.
        Defaults to None.
    isplots : int; optional
        The plotting verbosity. Defaults to 0.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.
    '''
    submask = np.copy(mask)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))
    maxiter = ny
    for i in range(nx):
        nobadpixels = False
        iternum = 0
        # Do not want to alter original data
        dataslice = np.copy(subdata[:, i])
        # Set initial guess if none given
        guess = [ny/10., np.argmax(dataslice), dataslice.max()]
        while not nobadpixels and iternum < maxiter:
            # Fit Gaussian to each column
            if sum(submask[:, i]) >= 3:
                params, err = g.fitgaussian(dataslice, np.arange(ny),
                                            mask=submask[:, i], fitbg=0,
                                            guess=guess)
            else:
                params = guess
                err = None
            # Create model
            model = g.gaussian(np.arange(ny), params[0], params[1], params[2])
            if isplots == 7:
                plt.figure(3703)
                plt.clf()
                plt.suptitle(str(i) + "," + str(iternum))
                plt.plot(dataslice, 'ro')
                plt.plot(dataslice*submask[:, i], 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.1)

            # Calculate residuals and number of sigma from the model
            residuals = submask[:, i]*(dataslice - model)
            if np.std(residuals) == 0:
                stdevs = np.zeros(residuals.shape)
            else:
                stdevs = np.abs(residuals) / np.std(residuals)
            # Find worst data point
            loc = np.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                # Check for bad fit, possibly due to a bad pixel
                if i > 0 and (err is None or
                              np.abs(params[0]) < np.abs(0.2*guess[0])):
                    # print(i, params)
                    # Remove brightest pixel within region of fit
                    loc = (params[1]-3 +
                           np.argmax(dataslice[params[1]-3:params[1]+4]))
                    # print(loc)
                else:
                    guess = np.abs(params)
                submask[loc, i] = 0
            else:
                nobadpixels = True  # exit while loop
                guess = np.abs(params)
            iternum += 1

        profile[:, i] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for dataslice ' +
                  str(i))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def clean_median_flux(data, meta, log, m):
    """Computes a median flux frame that is free of bad pixels.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.

    Notes
    -----
    History:

    - 2022-08-03, KBS
        Inital version
    """
    log.writelog('  Computing clean median frame...', mute=(not meta.verbose))

    # Compute median flux using masked arrays
    flux_ma = np.ma.masked_where(data.mask.values == 0, data.flux.values)
    medflux = np.ma.median(flux_ma, axis=0)
    ny, nx = medflux.shape

    # Interpolate over masked regions
    interp_med = np.zeros((ny, nx))
    xx = np.arange(nx)
    for j in range(ny):
        x1 = xx[~np.ma.getmaskarray(medflux[j])]
        goodrow = medflux[j][~np.ma.getmaskarray(medflux[j])]
        if len(goodrow) > 0:
            f = spi.interp1d(x1, goodrow, 'linear',
                             fill_value='extrapolate')
            # f = spi.UnivariateSpline(x1, goodmed, k=1, s=None)
            interp_med[j] = f(xx)
        else:
            log.writelog(f'    Row {j}: Interpolation failed. No good pixels.')
            interp_med[j] = medflux[j]

    if meta.window_len > 1:
        # Apply smoothing filter along dispersion direction
        smoothflux = spn.median_filter(interp_med, size=(1, meta.window_len))
        # Compute median error array
        err_ma = np.ma.masked_where(data.mask.values == 0, data.err.values)
        mederr = np.ma.median(err_ma, axis=0)
        # Compute residuals in units of std dev
        residuals = (medflux - smoothflux)/mederr

        # Flag outliers
        if not hasattr(meta, 'median_thresh'):
            log.writelog('  Using a default value of median_thresh=5',
                         mute=(not meta.verbose))
            meta.median_thresh = 5
        outliers = sigma_clip(residuals, sigma=meta.median_thresh,
                              maxiters=None, axis=1, cenfunc='median')

        # Interpolate over bad pixels
        clean_med = np.zeros((ny, nx))
        xx = np.arange(nx)
        for j in range(ny):
            x1 = xx[~np.ma.getmaskarray(outliers[j]) *
                    ~np.ma.getmaskarray(medflux[j])]
            goodrow = medflux[j][~np.ma.getmaskarray(outliers[j]) *
                                 ~np.ma.getmaskarray(medflux[j])]
            f = spi.interp1d(x1, goodrow, 'linear', fill_value='extrapolate')
            # f = spi.UnivariateSpline(x1, goodmed, k=1, s=None)
            clean_med[j] = f(xx)

        # Assign cleaned median frame to data object
        data['medflux'] = (['y', 'x'], clean_med)
    else:
        # Assign uncleaned median frame to data object
        data['medflux'] = (['y', 'x'], medflux.data)
    data['medflux'].attrs['flux_units'] = data.flux.attrs['flux_units']

    if meta.isplots_S3 >= 3:
        plots_s3.median_frame(data, meta, m)

    return data


def optimize_wrapper(data, meta, log, apdata, apmask, apbg, apv0, gain=1,
                     windowtype='hanning', m=0):
    '''Extract optimal spectrum with uncertainties for many frames.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    apdata : ndarray
        Background subtracted data.
    apmask : ndarray
        Outlier mask.
    apbg : ndarray
        Background array.
    apv0 : ndarray
        Variance array for data.
    gain : float
        The gain factor. Defaults to 1 as the flux should already be in
        electrons.
    windowtype : str; optional
        UNUSED. One of {'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'}. The type of window. A flat window will
        produce a moving average smoothing. Defaults to 'hanning'.
    m : int; optional
        File number. Defaults to 0.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.

    Notes
    -----
    History:

    - 2022-07-18, Taylor J Bell
        Added optimize_wrapper to iterate over each frame.
    '''
    # Extract optimal spectrum with uncertainties
    log.writelog("  Performing optimal spectral extraction...",
                 mute=(not meta.verbose))
    data['optspec'] = (['time', 'x'], np.zeros(data.stdspec.shape))
    data['opterr'] = (['time', 'x'], np.zeros(data.stdspec.shape))
    data['optmask'] = (['time', 'x'], np.zeros(data.stdspec.shape, dtype=bool))
    data['optspec'].attrs['flux_units'] = data.flux.attrs['flux_units']
    data['optspec'].attrs['time_units'] = data.flux.attrs['time_units']
    data['optspec'].attrs['wave_units'] = data.wave_1d.attrs['wave_units']
    data['opterr'].attrs['flux_units'] = data.flux.attrs['flux_units']
    data['opterr'].attrs['time_units'] = data.flux.attrs['time_units']
    data['opterr'].attrs['wave_units'] = data.wave_1d.attrs['wave_units']
    data['optmask'].attrs['flux_units'] = 'None'
    data['optmask'].attrs['time_units'] = data.flux.attrs['time_units']
    data['optmask'].attrs['wave_units'] = data.wave_1d.attrs['wave_units']

    # Select median frame over aperture region
    ap_y1 = int(meta.src_ypos-meta.spec_hw)
    ap_y2 = int(meta.src_ypos+meta.spec_hw)
    apmedflux = data.medflux[ap_y1:ap_y2].values

    # Perform optimal extraction on each of the frames
    iterfn = range(meta.int_start, meta.n_int)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        data['optspec'][n], data['opterr'][n], _ = \
            optimize(meta, apdata[n], apmask[n], apbg[n],
                     data.stdspec[n].values, gain, apv0[n],
                     p5thresh=meta.p5thresh,
                     p7thresh=meta.p7thresh,
                     fittype=meta.fittype,
                     window_len=meta.window_len,
                     deg=meta.prof_deg, windowtype=windowtype,
                     n=n, m=m, meddata=apmedflux)

    # Mask out NaNs and Infs
    optspec_ma = np.ma.masked_invalid(data.optspec.values)
    opterr_ma = np.ma.masked_invalid(data.opterr.values)
    optmask = np.logical_or(np.ma.getmaskarray(optspec_ma),
                            np.ma.getmaskarray(opterr_ma))
    data.optmask.values = optmask

    return data, meta, log


def optimize(meta, subdata, mask, bg, spectrum, Q, v0, p5thresh=10,
             p7thresh=10, fittype='smooth', window_len=21, deg=3,
             windowtype='hanning', n=0, m=0, meddata=None):
    '''Extract optimal spectrum with uncertainties for a single frame.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask.
    bg : ndarray
        Background array.
    spectrum : ndarray
        Standard spectrum.
    Q : float
        The gain factor.
    v0 : ndarray
        Variance array for data.
    p5thresh : float; optional
        Sigma threshold for outlier rejection while constructing
        spatial profile. Defaults to 10.
    p7thresh : float; optional
        Sigma threshold for outlier rejection during optimal
        spectral extraction. Defaukts to 10.
    fittype : str; optional
        One of {'smooth', 'meddata', 'wavelet2D', 'wavelet',
        'gauss', 'poly'}. The type of profile fitting
        you want to do. Defaults to 'smooth'.
    window_len : int; optional
        The dimension of the smoothing window. Defaults to 21.
    deg : int; optional
        Polynomial degree. Defaults to 3.
    windowtype : str; optional
        UNUSED. One of {'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'}. The type of window. A flat window will
        produce a moving average smoothing. Defaults to 'hanning'.
    n : int; optional
        Integration number. Defaults to 0.
    m : int; optional
        File number. Defaults to 0.
    meddata : ndarray; optional
        The median of all data frames. Defaults to None.

    Returns
    -------
    spectrum : ndarray
        The optimally extracted spectrum.
    specunc : ndarray
        The standard deviation on the spectrum.
    submask : ndarray
        The mask array.
    '''
    submask = np.copy(mask)
    ny, nx = subdata.shape
    isnewprofile = True
    # Loop through steps 5-8 until no more bad pixels are uncovered
    while isnewprofile:
        # STEP 5: Construct normalized spatial profile
        if fittype == 'smooth':
            profile = profile_smooth(subdata, submask, threshold=p5thresh,
                                     window_len=window_len,
                                     windowtype=windowtype,
                                     isplots=meta.isplots_S3)
        elif fittype == 'meddata':
            profile = profile_meddata(subdata, submask, meddata,
                                      threshold=p5thresh,
                                      isplots=meta.isplots_S3)
        elif fittype == 'wavelet2D':
            profile = profile_wavelet2D(subdata, submask, wavelet='bior5.5',
                                        numlvls=3, isplots=meta.isplots_S3)
        elif fittype == 'wavelet':
            profile = profile_wavelet(subdata, submask, wavelet='bior5.5',
                                      numlvls=3, isplots=meta.isplots_S3)
        elif fittype == 'gauss':
            profile = profile_gauss(subdata, submask, threshold=p5thresh,
                                    guess=None, isplots=meta.isplots_S3)
        elif fittype == 'poly':
            profile = profile_poly(subdata, submask, deg=deg,
                                   threshold=p5thresh)
        else:
            print("Unknown normalized spatial profile method.")
            return

        if meta.isplots_S3 >= 3 and n < meta.int_end:
            plots_s3.profile(meta, profile, submask, n, m)

        isnewprofile = False
        isoutliers = True
        # Loop through steps 6-8 until no more bad pixels are uncovered
        while isoutliers:
            # STEP 6: Revise variance estimates
            expected = profile*spectrum
            variance = np.abs(expected + bg) / Q + v0
            # STEP 7: Mask cosmic ray hits
            stdevs = np.abs(subdata - expected)*submask / np.sqrt(variance)
            if meta.isplots_S3 >= 5 and n < meta.int_end:
                plots_s3.stddev_profile(meta, n, m, stdevs, p7thresh)
            isoutliers = False
            if len(stdevs) > 0:
                # Find worst data point in each column
                loc = np.argmax(stdevs, axis=0)
                # Mask data point if std is > p7thresh
                for i in range(nx):
                    if meta.isplots_S3 == 8:
                        try:
                            plt.figure(3803)
                            plt.clf()
                            plt.suptitle(str(i) + "/" + str(nx))
                            plt.errorbar(np.arange(ny), subdata[:, i],
                                         yerr=np.sqrt(variance[:, i]),
                                         fmt='.', color='b')
                            plt.plot(expected[:, i], 'g-')
                            plt.pause(0.01)
                        except:
                            # FINDME: Need to only catch the expected exception
                            pass
                    if stdevs[loc[i], i] > p7thresh:
                        isnewprofile = True
                        isoutliers = True
                        submask[loc[i], i] = 0
                        # Generate plot
                        if meta.isplots_S3 >= 5 and n < meta.int_end:
                            plots_s3.subdata(meta, i, n, m, subdata, submask,
                                             expected, loc, variance)
                        # Check for insufficient number of good points
                        if sum(submask[:, i]) < ny/2.:
                            submask[:, i] = 0
            # STEP 8: Extract optimal spectrum
            denom = np.sum(profile*profile*submask/variance, axis=0)
            denom[np.where(denom == 0)] = np.inf
            spectrum = np.sum(profile*submask*subdata/variance, axis=0) / denom

    # Calculate variance of optimal spectrum
    specvar = np.sum(profile*submask, axis=0) / denom

    # Return spectrum and uncertainties
    return spectrum, np.sqrt(specvar), submask
