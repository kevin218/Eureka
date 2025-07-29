import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.ndimage as spn
from astropy.stats import sigma_clip
from tqdm import tqdm
import multiprocessing as mp

from ..lib import gaussian as g
from ..lib import smooth, plots
from . import plots_s3


def standard_spectrum(apdata, apmask, aperr):
    """Compute the standard box spectrum.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    apdata : ndarray
        The pixel values in the aperture region.
    apmask : ndarray
        The outlier mask in the aperture region. True where pixels should be
        masked.
    aperr : ndarray
        The noise values in the aperture region.

    Returns
    -------
    stdspec : 2D array
        Time-series of stellar spectra
    stdvar : 2D array
        Time-series of stellar variances
    """
    # Replace masked pixels with spectral neighbors
    apdata_cleaned = np.copy(apdata)
    aperr_cleaned = np.copy(aperr)

    for t, y, x in np.array(np.where(apmask)).T:
        # Do not extend to negative indices (short and long wavelengths
        # do not have similar profiles)
        lower = x-2
        if lower < 0:
            lower = 0
        # Get mask for current neighbors
        mask_temp = np.append(apmask[t, y, lower:x],
                              apmask[t, y, x+1:x+3])

        # Gather current data neighbors and apply mask
        replacement_val = ~mask_temp*np.append(apdata_cleaned[t, y, lower:x],
                                               apdata_cleaned[t, y, x+1:x+3])
        # Figure out how many data neighbors are being used
        denom = np.sum(~mask_temp)
        # Compute the mean of the unmasked data neighbors
        replacement_val = np.nansum(replacement_val)/denom
        # Replace masked value with the newly computed data value
        apdata_cleaned[t, y, x] = replacement_val

        # Gather current err neighbors and apply mask
        replacement_val = ~mask_temp*np.append(aperr_cleaned[t, y, lower:x],
                                               aperr_cleaned[t, y, x+1:x+3])
        # Compute the mean of the unmasked err neighbors
        replacement_val = np.nansum(replacement_val)/denom
        # Replace masked value with the newly computed err value
        aperr_cleaned[t, y, x] = replacement_val

    # Compute standard spectra
    stdspec = np.nansum(apdata_cleaned, axis=1)
    stdvar = np.nansum(aperr_cleaned**2, axis=1)

    return stdspec, stdvar


@plots.apply_style
def profile_poly(subdata, mask, deg=3, threshold=10, isplots=0):
    '''Construct normalized spatial profile using polynomial fits along the
    wavelength direction.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask, with True values being masked.
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
            dataslice = np.ma.masked_where(submask[j], subdata[j])
            # Replace masked points with median of nearby points
            for ind in np.where(submask[j])[0]:
                dataslice[ind] = \
                    np.ma.median(dataslice[np.max((0, ind-10)):ind+11])

            # Smooth each row
            coeffs = np.ma.polyfit(range(nx), dataslice, deg)
            model = np.polyval(coeffs, range(nx))
            if isplots == 7:
                plt.figure(3703)
                plt.clf()
                plt.suptitle(str(j) + "," + str(iternum))
                plt.plot(dataslice.data, 'ro')
                plt.plot(dataslice, 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.1)

            # Calculate residuals and number of sigma from the model
            residuals = dataslice - model
            stdevs = np.ma.abs(residuals) / np.ma.std(residuals)
            # Find worst data point
            loc = np.ma.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                nobadpixels = False
                submask[j, loc] = True
            else:
                nobadpixels = True  # exit while loop
            iternum += 1

        profile[j] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for '
                  'dataslice ' + str(j))

    # Enforce positivity
    profile[profile < 0] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.nansum(profile, axis=0)

    return profile


@plots.apply_style
def profile_smooth(subdata, mask, threshold=10, window_len=21,
                   windowtype='hanning', isplots=0):
    '''Construct normalized spatial profile using a smoothing function.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask, with True values being masked.
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
        if np.sum(~submask[j]) > 0:
            nobadpixels = False
            iternum = 0
            maxiter = np.sum(~submask[j])
            while not nobadpixels and iternum < maxiter:
                # Do not want to alter original data
                dataslice = np.ma.masked_where(submask[j], subdata[j])
                # Replace masked points with median of nearby points
                # dataslice[np.where(submask[j] == 0)] = 0
                # FINDME: Code below appears to be effective, but is slow for
                # lots of masked points
                for ind in np.where(submask[j])[0]:
                    dataslice[ind] = \
                        np.ma.median(dataslice[np.max((0, ind-10)):ind+11])

                # Smooth each row
                # model = smooth.smooth(dataslice, window_len=window_len,
                #                       window=windowtype)
                model = smooth.medfilt(dataslice, window_len)
                if isplots == 7:
                    plt.figure(3703)
                    plt.clf()
                    plt.suptitle(str(j) + "," + str(iternum))
                    plt.plot(dataslice.data, 'ro')
                    plt.plot(dataslice, 'bo')
                    plt.plot(model, 'g-')
                    plt.pause(0.1)

                # Calculate residuals and number of sigma from the model
                residuals = dataslice - model
                stdevs = np.ma.abs(residuals) / np.ma.std(residuals)
                # Find worst data point
                loc = np.ma.argmax(stdevs)
                # Mask data point if > threshold
                if stdevs[loc] > threshold:
                    nobadpixels = False
                    submask[j, loc] = True
                else:
                    nobadpixels = True  # exit while loop
                iternum += 1
            # Copy model slice to profile
            profile[j] = model
            if iternum == maxiter:
                print('WARNING: Max number of iterations reached for '
                      'dataslice ' + str(j))

    # Enforce positivity
    profile[profile < 0] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.nansum(profile, axis=0)

    return profile


def profile_meddata(meddata):
    '''Construct normalized spatial profile using median of all data frames.

    Parameters
    ----------
    meddata : ndarray
        The median of all data frames.

    Returns
    -------
    profile : ndarray
        Fitted profile in the same shape as the input data array.
    '''
    profile = np.ma.copy(meddata)
    # Enforce positivity
    profile[profile < 0] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.ma.sum(profile, axis=0)

    return profile


# Construct normalized spatial profile using wavelets
@plots.apply_style
def profile_wavelet(subdata, mask, wavelet, numlvls, isplots=0):
    '''This function performs 1D image denoising using BayesShrink
    soft thresholding.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask, with True values being masked.
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
    subdata = np.ma.masked_where(submask, subdata)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))

    for j in range(ny):
        # Perform wavelet decomposition
        dec = pywt.wavedec(subdata[j], wavelet)
        # Estimate noise variance
        noisevar = np.inf
        for i in range(-1, -numlvls-1, -1):
            noisevar = np.min([(np.ma.median(np.ma.abs(dec[i]))/0.6745)**2,
                               noisevar])
        # At each level of decomposition...
        for i in range(-1, -numlvls-1, -1):
            # Estimate variance at level i then compute the threshold value
            # sigmay2 = np.mean(dec[i]*dec[i])
            # sigmax = np.sqrt(np.max([sigmay2-noisevar, 0]))
            threshold = np.ma.max(np.ma.abs(dec[i]))
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
            plt.plot(subdata[j].data, 'ro')
            plt.plot(subdata[j], 'bo')
            plt.plot(profile[j], 'g-')
            plt.pause(0.1)

    # Enforce positivity
    profile[profile < 0] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.nansum(profile, axis=0)

    return profile


@plots.apply_style
def profile_wavelet2D(subdata, mask, wavelet, numlvls, isplots=0):
    '''Construct normalized spatial profile using wavelets

    This function performs 2D image denoising using BayesShrink
    soft thresholding.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask, with True values being masked.
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
    subdata = np.ma.masked_where(submask, subdata)
    ny, nx = np.shape(subdata)
    profile = np.zeros((ny, nx))

    # Perform wavelet decomposition
    dec = pywt.wavedec2(subdata, wavelet)
    # Estimate noise variance
    noisevar = np.inf
    for i in range(-1, -numlvls-1, -1):
        noisevar = np.min([(np.ma.median(np.ma.abs(dec[i]))/0.6745)**2,
                           noisevar])
    # At each level of decomposition...
    for i in range(-1, -numlvls-1, -1):
        # Estimate variance at level i then compute the threshold value
        # sigmay2 = np.mean((dec[i][0]*dec[i][0]+dec[i][1]*dec[i][1] +
        #                    dec[i][2]*dec[i][2])/3.)
        # sigmax = np.sqrt(np.max([sigmay2-noisevar, 0]))
        threshold = np.ma.max(np.ma.abs(dec[i]))
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
        plt.plot(subdata[ny//2].data, 'ro')
        plt.plot(subdata[ny//2], 'bo')
        plt.plot(profile[ny//2], 'g-')

        plt.figure(3704)
        plt.clf()
        # plt.suptitle(str(j) + "," + str(iternum))
        plt.plot(subdata[:, int(nx/2)].data, 'ro')
        plt.plot(subdata[:, int(nx/2)], 'bo')
        plt.plot(profile[:, int(nx/2)], 'g-')
        plt.pause(0.1)

    # Enforce positivity
    profile[profile < 0] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.nansum(profile, axis=0)

    return profile


@plots.apply_style
def profile_gauss(subdata, mask, threshold=10, guess=None, isplots=0):
    '''Construct normalized spatial profile using Gaussian smoothing function.

    Parameters
    ----------
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask, with True values being masked.
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
        dataslice = np.ma.masked_where(submask[:, i], subdata[:, i])
        # Set initial guess if none given
        guess = [ny/10., np.ma.argmax(dataslice), np.ma.max(dataslice)]
        while not nobadpixels and iternum < maxiter:
            # Fit Gaussian to each column
            if sum(~submask[:, i]) >= 3:
                # If there are 3 or more good elements, fit the gaussian
                params, err = g.fitgaussian(dataslice, np.arange(ny),
                                            mask=submask[:, i], fitbg=0,
                                            guess=guess)
            else:
                # If there are fewer than 3 elements, don't fit gaussian
                params = guess
                err = None
            # Create model
            model = g.gaussian(np.arange(ny), params[0], params[1], params[2])
            if isplots == 7:
                plt.figure(3703)
                plt.clf()
                plt.suptitle(str(i) + "," + str(iternum))
                plt.plot(dataslice.data, 'ro')
                plt.plot(dataslice, 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.1)

            # Calculate residuals and number of sigma from the model
            residuals = dataslice - model
            if np.ma.std(residuals) == 0:
                stdevs = np.zeros(residuals.shape)
            else:
                stdevs = np.ma.abs(residuals) / np.ma.std(residuals)
            # Find worst data point
            loc = np.ma.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                # Check for bad fit, possibly due to a bad pixel
                if i > 0 and (err is None or
                              np.abs(params[0]) < np.abs(0.2*guess[0])):
                    # print(i, params)
                    # Remove brightest pixel within region of fit
                    loc = (params[1]-3 +
                           np.ma.argmax(dataslice[params[1]-3:params[1]+4]))
                    # print(loc)
                else:
                    guess = np.abs(params)
                submask[loc, i] = True
            else:
                nobadpixels = True  # exit while loop
                guess = np.abs(params)
            iternum += 1

        profile[:, i] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for dataslice ' +
                  str(i))

    # Enforce positivity
    profile[profile < 0] = 0
    # Normalize along spatial direction
    with np.errstate(divide='ignore', invalid='ignore'):
        profile /= np.nansum(profile, axis=0)

    return profile


def get_clean(data, meta, log, medflux, mederr):
    """Computes a median flux frame that is free of bad pixels.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    medflux : array
        2D array of median flux
    mederr : array
        2D array of median flux uncertainties

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.
    """
    nx = len(data.x)
    ny = len(data.y)

    # Interpolate over masked regions
    interp_med = np.zeros((ny, nx))
    xx = np.arange(nx)
    for j in range(ny):
        x1 = xx[~np.ma.getmaskarray(medflux[j])]
        goodrow = medflux[j][~np.ma.getmaskarray(medflux[j])]
        if len(goodrow) > 0:
            f = spi.interp1d(x1, goodrow, 'linear',
                             fill_value='extrapolate')
            interp_med[j] = f(xx)
        else:
            log.writelog(f'    Row {j}: Interpolation failed. No good pixels.')
            interp_med[j] = medflux[j]

    if meta.window_len > 1:
        # Apply smoothing filter along dispersion direction
        smoothflux = spn.median_filter(interp_med, size=(1, meta.window_len))
        # Smooth error array along dispersion direction
        # to enable flagging bad points with large uncertainties
        smooth_mederr = spn.median_filter(mederr, size=(1, meta.window_len))
        # Compute residuals in units of std dev
        residuals = (medflux - smoothflux)/smooth_mederr

        # Flag outliers
        outliers = sigma_clip(residuals, sigma=meta.median_thresh, maxiters=5,
                              axis=1, cenfunc=np.ma.median, stdfunc=np.ma.std)

        # Interpolate over bad pixels
        clean_med = np.zeros((ny, nx))
        xx = np.arange(nx)
        for j in range(ny):
            x1 = xx[~np.ma.getmaskarray(outliers[j]) *
                    ~np.ma.getmaskarray(medflux[j])]
            goodrow = medflux[j][~np.ma.getmaskarray(outliers[j]) *
                                 ~np.ma.getmaskarray(medflux[j])]
            if len(goodrow) > 0:
                f = spi.interp1d(x1, goodrow, 'linear',
                                 fill_value='extrapolate')
                clean_med[j] = f(xx)

        return clean_med
    else:
        return medflux.data


def optimize_wrapper(data, meta, log, apdata, apmask, apbg, apv0, apmedflux,
                     gain=1, windowtype='hanning', m=0):
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
        Outlier mask, with True values being masked.
    apbg : ndarray
        Background array.
    apv0 : ndarray
        Variance array for data.
    apmedflux : ndarray
        Median flux array.
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
    '''
    # Extract optimal spectrum with uncertainties
    log.writelog("  Performing optimal spectral extraction...",
                 mute=(not meta.verbose))

    coords = list(data.stdspec.coords.keys())
    data['optspec'] = (coords, np.zeros_like(data.stdspec))
    data['opterr'] = (coords, np.zeros_like(data.stdspec))
    data['optmask'] = (coords, np.zeros_like(data.stdspec, dtype=bool))
    data['optspec'].attrs['flux_units'] = data.flux.attrs['flux_units']
    data['optspec'].attrs['time_units'] = data.flux.attrs['time_units']
    data['optspec'].attrs['wave_units'] = data.wave_1d.attrs['wave_units']
    data['opterr'].attrs['flux_units'] = data.flux.attrs['flux_units']
    data['opterr'].attrs['time_units'] = data.flux.attrs['time_units']
    data['opterr'].attrs['wave_units'] = data.wave_1d.attrs['wave_units']
    data['optmask'].attrs['flux_units'] = 'None'
    data['optmask'].attrs['time_units'] = data.flux.attrs['time_units']
    data['optmask'].attrs['wave_units'] = data.wave_1d.attrs['wave_units']

    # Write optimal extraction results
    def writeOptSpex(arg):
        optspec, opterr, _, n, _ = arg
        data['optspec'][n] = optspec
        data['opterr'][n] = opterr
        return

    # Write optimal extraction results for detectors with multiple orders
    def writeOptSpexMultiOrder(arg):
        optspec, opterr, _, n, k = arg
        data['optspec'][n, :, k] = optspec
        data['opterr'][n, :, k] = opterr
        return

    # Perform optimal extraction on each of the integrations
    iterfn = range(meta.int_start, meta.n_int)
    if meta.ncpu == 1:
        # Only 1 CPU
        if meta.verbose:
            iterfn = tqdm(iterfn)
        if meta.orders is None:
            for n in iterfn:
                writeOptSpex(optimize(
                    meta, apdata[n], apmask[n], apbg[n],
                    data.stdspec[n].values, gain, apv0[n],
                    p5thresh=meta.p5thresh, p7thresh=meta.p7thresh,
                    fittype=meta.fittype, window_len=meta.window_len,
                    deg=meta.prof_deg, windowtype=windowtype,
                    n=n, m=m, meddata=apmedflux))
        else:
            norders = len(meta.orders)
            for n in iterfn:
                for k in range(norders):
                    writeOptSpexMultiOrder(optimize(
                        meta, apdata[n, :, :, k], apmask[n, :, :, k],
                        apbg[n, :, :, k], data.stdspec[n, :, k].values,
                        gain, apv0[n, :, :, k], p5thresh=meta.p5thresh,
                        p7thresh=meta.p7thresh, fittype=meta.fittype,
                        window_len=meta.window_len, deg=meta.prof_deg,
                        windowtype=windowtype, n=n, m=m,
                        meddata=apmedflux[:, :, k], order=meta.orders[k]))
    else:
        # Multiple CPU threads
        pool = mp.Pool(meta.ncpu)
        jobs = []
        if meta.orders is None:
            for n in iterfn:
                job = pool.apply_async(func=optimize, args=(
                    meta, apdata[n], apmask[n], apbg[n],
                    data.stdspec[n].values, gain, apv0[n],
                    meta.p5thresh, meta.p7thresh,
                    meta.fittype, meta.window_len,
                    meta.prof_deg, windowtype,
                    n, m, apmedflux), callback=writeOptSpex)
                jobs.append(job)
        else:
            norders = len(meta.orders)
            for n in iterfn:
                for k in range(norders):
                    job = pool.apply_async(func=optimize, args=(
                        meta, apdata[n, :, :, k], apmask[n, :, :, k],
                        apbg[n, :, :, k], data.stdspec[n, :, k].values,
                        gain, apv0[n, :, :, k], meta.p5thresh,
                        meta.p7thresh, meta.fittype, meta.window_len,
                        meta.prof_deg, windowtype, n, m, apmedflux[:, :, k],
                        meta.orders[k]), callback=writeOptSpexMultiOrder)
                    jobs.append(job)
        pool.close()
        iterfn = jobs
        if meta.verbose:
            iterfn = tqdm(iterfn)
        for job in iterfn:
            job.get()

    # Mask out NaNs and Infs
    optspec_ma = np.ma.masked_invalid(data.optspec.values)
    opterr_ma = np.ma.masked_invalid(data.opterr.values)
    optmask = np.ma.getmaskarray(optspec_ma) + np.ma.getmaskarray(opterr_ma)
    data['optmask'][:] = optmask

    return data, meta, log


@plots.apply_style
def optimize(meta, subdata, mask, bg, spectrum, Q, v0, p5thresh=10,
             p7thresh=10, fittype='smooth', window_len=21, deg=3,
             windowtype='hanning', n=0, m=0, meddata=None, order=None):
    '''Extract optimal spectrum with uncertainties for a single frame.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    subdata : ndarray
        Background subtracted data.
    mask : ndarray
        Outlier mask, with True values being masked.
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
    order : int; optional
        Spectral order. Default is None

    Returns
    -------
    spectrum : ndarray
        The optimally extracted spectrum.
    specunc : ndarray
        The standard deviation on the spectrum.
    submask : ndarray
        The mask array.
    n : int
        The input integration number (useful for multiprocessing)
    order : int
        The input spectral order number (useful for multiprocessing)
    '''
    submask = np.copy(mask)
    ny, nx = subdata.shape
    subdata = np.ma.masked_invalid(subdata)
    subdata = np.ma.masked_where(submask, subdata)
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
            profile = profile_meddata(meddata)
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

        isnewprofile = False
        isoutliers = True
        # Loop through steps 6-8 until no more bad pixels are uncovered
        while isoutliers:
            # Mask any known bad points
            subdata = np.ma.masked_where(submask, subdata)
            # STEP 6: Revise variance estimates
            expected = np.ma.masked_invalid(profile*spectrum)
            variance = np.ma.abs(expected + bg) / Q + v0
            # STEP 7: Mask cosmic ray hits
            stdevs = np.ma.abs(subdata - expected)/np.ma.sqrt(variance)
            submask[np.ma.getmaskarray(stdevs)] = True
            # Mask any known bad points
            subdata = np.ma.masked_where(submask, subdata)
            if meta.isplots_S3 >= 5 and n < meta.int_end:
                plots_s3.stddev_profile(meta, n, m, stdevs, p7thresh)
            isoutliers = False
            for i in range(nx):
                # Only continue if there are unmasked values
                if np.any(~submask[:, i]):
                    # Find worst data point in each column
                    loc = np.ma.argmax(stdevs[:, i])

                    if meta.isplots_S3 == 8:
                        try:
                            plt.figure(3803)
                            plt.clf()
                            plt.suptitle(str(i) + "/" + str(nx))
                            plt.errorbar(np.arange(ny), subdata[:, i],
                                         yerr=np.ma.sqrt(variance[:, i]),
                                         fmt='.', color='b')
                            plt.plot(expected[:, i], 'g-')
                            plt.pause(0.01)
                        except:
                            # FINDME: Need to only catch the expected exception
                            pass
                    # Mask data point if std is > p7thresh
                    if stdevs[loc, i] > p7thresh:
                        isnewprofile = True
                        isoutliers = True
                        submask[loc, i] = True
                        # Generate plot
                        if meta.isplots_S3 >= 5 and n < meta.int_end:
                            plots_s3.subdata(meta, i, n, m, subdata, submask,
                                             expected, loc, variance)
                        # Check for insufficient number of good points
                        if np.sum(~submask[:, i]) < ny/2.:
                            submask[:, i] = True
            # STEP 8: Extract optimal spectrum
            with np.errstate(divide='ignore', invalid='ignore'):
                # Ignore warnings about columns that are completely masked
                denom = np.ma.sum(profile*profile*~submask/variance, axis=0)
            denom = np.ma.masked_where(denom == 0, denom)
            spectrum = np.ma.sum(profile*~submask*subdata/variance,
                                 axis=0)/denom

    if meta.isplots_S3 >= 3 and n < meta.int_end:
        plots_s3.profile(meta, profile, submask, n, m, order=order)

    # Calculate variance of optimal spectrum
    specvar = np.ma.sum(profile*~submask, axis=0) / denom

    # Return spectrum and uncertainties
    return spectrum, np.sqrt(specvar), submask, n, order
