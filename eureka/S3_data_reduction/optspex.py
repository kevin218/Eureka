
import numpy as np
import matplotlib.pyplot as plt
from ..lib import gaussian as g
from ..lib import smooth
from . import plots_s3

def profile_poly(subdata, mask, deg=3, threshold=10, isplots=0):
    '''Construct normalized spatial profile using polynomial fits along the wavelength direction.

    Parameters
    ----------
    subdata:    ndarray
        Background subtracted data.
    mask:   ndarray
        Outlier mask.
    deg:    int
        Polynomial degree.
    threshold:  float
        Sigma threshold for outlier rejection while constructing spatial profile.
    isplots:    int
        The amount of plots saved; set in ecf.

    Returns
    -------
    profile:    ndarray
        Fitted profile in the same shape as the input data array.
    '''
    submask  = np.copy(mask)
    ny, nx   = np.shape(subdata)
    profile  = np.zeros((ny, nx))
    maxiter  = nx
    for j in range(ny):
        nobadpixels = False
        iternum     = 0
        while (nobadpixels == False) and (iternum < maxiter):
            dataslice = np.copy(subdata[j])     #Do not want to alter original data
            # Replace masked points with median of nearby points
            for ind in np.where(submask[j] == 0)[0]:
                dataslice[ind] = np.median(dataslice[np.max((0,ind-10)):ind+11])

            # Smooth each row
            coeffs    = np.polyfit(range(nx), dataslice, deg)
            model     = np.polyval(coeffs, range(nx))
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
            stdevs    = np.abs(residuals) / np.std(residuals)
            # Find worst data point
            loc       = np.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                nobadpixels = False
                submask[j,loc] = 0
            else:
                nobadpixels = True      #exit while loop
            iternum += 1

        profile[j] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for dataslice ' + str(j))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_smooth(subdata, mask, threshold=10, window_len=21, windowtype='hanning', isplots=False):
    '''Construct normalized spatial profile using a smoothing function.

    Parameters
    ----------
    subdata:    ndarray
        Background subtracted data.
    mask:   ndarray
        Outlier mask.
    threshold:  float
        Sigma threshold for outlier rejection while constructing spatial profile.
    window_len: int
        The dimension of the smoothing window.
    windowtype: {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
        UNUSED. The type of window. A flat window will produce a moving average smoothing.
    isplots:    int
        The amount of plots saved; set in ecf.

    Returns
    -------
    profile:    ndarray
        Fitted profile in the same shape as the input data array.
    '''
    submask  = np.copy(mask)
    ny, nx   = np.shape(subdata)
    profile  = np.zeros((ny, nx))
    maxiter  = nx
    for j in range(ny):
        # Check for good pixels in row
        if np.sum(submask[j]) > 0:
            nobadpixels = False
            iternum     = 0
            maxiter     = np.sum(submask[j])
            while (nobadpixels == False) and (iternum < maxiter):
                dataslice = np.copy(subdata[j])     #Do not want to alter original data
                # Replace masked points with median of nearby points
                #dataslice[np.where(submask[j] == 0)] = 0
                #FINDME: Code below appears to be effective, but is slow for lots of masked points
                for ind in np.where(submask[j] == 0)[0]:
                    dataslice[ind] = np.median(dataslice[np.max((0,ind-10)):ind+11])

                # Smooth each row
                #model = smooth.smooth(dataslice, window_len=window_len, window=windowtype)
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
                stdevs    = np.abs(residuals[igoodmask]) / np.std(residuals[igoodmask])
                # Find worst data point
                loc       = np.argmax(stdevs)
                # Mask data point if > threshold
                if stdevs[loc] > threshold:
                    nobadpixels = False
                    submask[j,igoodmask[loc]] = 0
                else:
                    nobadpixels = True      #exit while loop
                iternum += 1
            # Copy model slice to profile
            profile[j] = model
            if iternum == maxiter:
                print('WARNING: Max number of iterations reached for dataslice ' + str(j))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_meddata(data, mask, meddata, threshold=10, isplots=0):
    '''Construct normalized spatial profile using median of all data frames.

    Parameters
    ----------
    data:    ndarray
        UNUSED. Image data.
    mask:   ndarray
        UNUSED. Outlier mask.
    meddata:    ndarray
        The median of all data frames.
    threshold:  float
        UNUSED. Sigma threshold for outlier rejection while constructing spatial profile.
    isplots:    int
        UNUSED. The amount of plots saved; set in ecf.

    Returns
    -------
    profile:    ndarray
        Fitted profile in the same shape as the input data array.
    '''
    #profile = np.copy(meddata*mask)
    profile = np.copy(meddata)
    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


# Construct normalized spatial profile using wavelets
def profile_wavelet(subdata, mask, wavelet, numlvls, isplots=0):
    '''This function performs 1D image denoising using BayesShrink soft thresholding.

    Parameters
    ----------
    subdata:    ndarray
        Background subtracted data.
    mask:   ndarray
        Outlier mask.
    wavelet:    Wavelet object or name string
        qWavelet to use
    numlvls:    int
        Decomposition levels to consider (must be >= 0).
    isplots:    int
        The amount of plots saved; set in ecf.

    Returns
    -------
    profile:    ndarray
        Fitted profile in the same shape as the input data array.

    References
    ----------
    Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and Compression", 2000
    '''
    import pywt
    submask  = np.copy(mask)
    ny, nx   = np.shape(subdata)
    profile  = np.zeros((ny, nx))

    for j in range(ny):
        #Perform wavelet decomposition
        dec = pywt.wavedec(subdata[j],wavelet)
        #Estimate noise variance
        noisevar = np.inf
        for i in range(-1,-numlvls-1,-1):
            noisevar = np.min([(np.median(np.abs(dec[i]))/0.6745)**2,noisevar])
        #At each level of decomposition...
        for i in range(-1,-numlvls-1,-1):
            #Estimate variance at level i then compute the threshold value
            sigmay2 = np.mean(dec[i]*dec[i])
            sigmax  = np.sqrt(np.max([sigmay2-noisevar,0]))
            threshold = np.max(np.abs(dec[i]))
            #if sigmax == 0 or i == -1:
            #    threshold = np.max(np.abs(dec[i]))
            #else:
            #    threshold = noisevar/sigmax
            #Compute less noisy coefficients by applying soft thresholding
            dec[i] = map (lambda x: pywt.thresholding.soft(x,threshold), dec[i])

        profile[j] = pywt.waverec(dec,wavelet)[:nx]
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

# Construct normalized spatial profile using wavelets
def profile_wavelet2D(subdata, mask, wavelet, numlvls, isplots=0):
    '''This function performs 2D image denoising using BayesShrink soft thresholding.
    
    Parameters
    ----------
    subdata:    ndarray
        Background subtracted data.
    mask:   ndarray
        Outlier mask.
    wavelet:    Wavelet object or name string
        qWavelet to use
    numlvls:    int
        Decomposition levels to consider (must be >= 0).
    isplots:    int
        The amount of plots saved; set in ecf.

    Returns
    -------
    profile:    ndarray
        Fitted profile in the same shape as the input data array.

    References
    ----------
    Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and Compression", 2000
    '''
    import pywt
    submask  = np.copy(mask)
    ny, nx   = np.shape(subdata)
    profile  = np.zeros((ny, nx))

    #Perform wavelet decomposition
    dec = pywt.wavedec2(subdata,wavelet)
    #Estimate noise variance
    noisevar = np.inf
    for i in range(-1,-numlvls-1,-1):
        noisevar = np.min([(np.median(np.abs(dec[i]))/0.6745)**2,noisevar])
    #At each level of decomposition...
    for i in range(-1,-numlvls-1,-1):
        #Estimate variance at level i then compute the threshold value
        sigmay2 = np.mean((dec[i][0]*dec[i][0]+dec[i][1]*dec[i][1]+dec[i][2]*dec[i][2])/3.)
        sigmax  = np.sqrt(np.max([sigmay2-noisevar,0]))
        threshold = np.max(np.abs(dec[i]))
        #if sigmax == 0:
        #    threshold = np.max(np.abs(dec[i]))
        #else:
        #    threshold = noisevar/sigmax
        #Compute less noisy coefficients by applying soft thresholding
        dec[i] = map (lambda x: pywt.thresholding.soft(x,threshold), dec[i])

    profile = pywt.waverec2(dec,wavelet)[:ny,:nx]
    if isplots == 7:
        plt.figure(3703)
        plt.clf()
        #plt.suptitle(str(j) + "," + str(iternum))
        plt.plot(subdata[ny/2], 'ro')
        plt.plot(subdata[ny/2]*submask[ny/2], 'bo')
        plt.plot(profile[ny/2], 'g-')
        plt.figure(3704)
        plt.clf()
        #plt.suptitle(str(j) + "," + str(iternum))
        plt.plot(subdata[:,nx/2], 'ro')
        plt.plot(subdata[:,nx/2]*submask[:,nx/2], 'bo')
        plt.plot(profile[:,nx/2], 'g-')
        plt.pause(0.1)

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def profile_gauss(subdata, mask, threshold=10, guess=None, isplots=0):
    '''Construct normalized spatial profile using a Gaussian smoothing function.

    Parameters
    ----------
    subdata:    ndarray
        Background subtracted data.
    mask:   ndarray
        Outlier mask.
    threshold:  float
        Sigma threshold for outlier rejection while constructing spatial profile.
    guess: list
        UNUSED. The initial guess for the Gaussian parameters.
    isplots:    int
        The amount of plots saved; set in ecf.

    Returns
    -------
    profile:    ndarray
        Fitted profile in the same shape as the input data array.
    '''
    submask  = np.copy(mask)
    ny, nx   = np.shape(subdata)
    profile  = np.zeros((ny, nx))
    maxiter  = ny
    for i in range(nx):
        nobadpixels = False
        iternum     = 0
        dataslice = np.copy(subdata[:,i])     #Do not want to alter original data
        # Set initial guess if none given
        guess = [ny/10.,np.argmax(dataslice),dataslice.max()]
        while (nobadpixels == False) and (iternum < maxiter):
            #if guess == None:
                #guess = g.old_gaussianguess(dataslice, np.arange(ny), mask=submask[:,i])
            # Fit Gaussian to each column
            if sum(submask[:,i]) >= 3:
                params, err = g.fitgaussian(dataslice, np.arange(ny), mask=submask[:,i], fitbg=0, guess=guess)
            else:
                params = guess
                err    = None
            # Create model
            model  = g.gaussian(np.arange(ny), params[0], params[1], params[2])
            if isplots == 7:
                plt.figure(3703)
                plt.clf()
                plt.suptitle(str(i) + "," + str(iternum))
                plt.plot(dataslice, 'ro')
                plt.plot(dataslice*submask[:,i], 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.1)

            # Calculate residuals and number of sigma from the model
            residuals  = submask[:,i]*(dataslice - model)
            if np.std(residuals) == 0:
                stdevs = np.zeros(residuals.shape)
            else:
                stdevs = np.abs(residuals) / np.std(residuals)
            # Find worst data point
            loc        = np.argmax(stdevs)
            # Mask data point if > threshold
            if stdevs[loc] > threshold:
                # Check for bad fit, possibly due to a bad pixel
                if i > 0 and (err == None or abs(params[0]) < abs(0.2*guess[0])):
                    #print(i, params)
                    # Remove brightest pixel within region of fit
                    loc = params[1]-3 + np.argmax(dataslice[params[1]-3:params[1]+4])
                    #print(loc)
                else:
                    guess = abs(params)
                submask[loc,i] = 0
            else:
                nobadpixels = True      #exit while loop
                guess = abs(params)
            iternum += 1

        profile[:,i] = model
        if iternum == maxiter:
            print('WARNING: Max number of iterations reached for dataslice ' + str(i))

    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)

    return profile


def optimize(subdata, mask, bg, spectrum, Q, v0, p5thresh=10, p7thresh=10, fittype='smooth', window_len=21, deg=3, windowtype='hanning', n=0, isplots=0, eventdir='.',meddata=None, hide_plots=False):
    '''Extract optimal spectrum with uncertainties.

    Parameters
    ----------
    subdata:    ndarray
        Background subtracted data.
    mask:   ndarray
        Outlier mask.
    bg: ndarray
        Background array.
    spectrum:   ndarray
        Standard spectrum.
    Q:  float
        The gain factor.
    v0:     ndarray
        Variance array for data.
    p5thresh:   float
        Sigma threshold for outlier rejection while constructing spatial profile.
    p7thresh:   float
        Sigma threshold for outlier rejection during optimal spectral extraction.
    fittype:    {'smooth', 'meddata', 'wavelet2D', 'wavelet', 'gauss', 'poly'}
        The type of profile fitting you want to do.
    window_len: int
        The dimension of the smoothing window.
    deg:    int
        Polynomial degree.
    windowtype: {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
        UNUSED. The type of window. A flat window will produce a moving average smoothing.
    n:  int
        Integration number.
    isplots:    int
        The amount of plots saved; set in ecf.
    eventdir:   str
        Directory in which to save outupts.
    meddata:    ndarray
        The median of all data frames.
    hide_plots: 
        If True, plots will automatically be closed rather than popping up.

    Returns
    -------
    spectrum:   ndarray
        The optimally extracted spectrum.
    specunc:    ndarray
        The standard deviation on the spectrum.
    submask:    ndarray
        The mask array.
    '''
    submask      = np.copy(mask)
    ny, nx       = subdata.shape
    isnewprofile = True
    # Loop through steps 5-8 until no more bad pixels are uncovered
    while(isnewprofile == True):
        # STEP 5: Construct normalized spatial profile
        if fittype == 'smooth':
            profile = profile_smooth(subdata, submask, threshold=p5thresh, window_len=window_len, windowtype=windowtype, isplots=isplots)
        elif fittype == 'meddata':
            profile = profile_meddata(subdata, submask, meddata, threshold=p5thresh, isplots=isplots)
        elif fittype == 'wavelet2D':
            profile = profile_wavelet2D(subdata, submask, wavelet='bior5.5', numlvls=3, isplots=isplots)
        elif fittype == 'wavelet':
            profile = profile_wavelet(subdata, submask, wavelet='bior5.5', numlvls=3, isplots=isplots)
        elif fittype == 'gauss':
            profile = profile_gauss(subdata, submask, threshold=p5thresh, guess=None, isplots=isplots)
        elif fittype == 'poly':
            profile = profile_poly(subdata, submask, deg=deg, threshold=p5thresh)
        else:
            print("Unknown normalized spatial profile method.")
            return
        #
        if isplots >= 3:
            plots_s3.profile(eventdir, profile, submask, n, hide_plots=hide_plots)
            # try:
            #     plots_s3.profile(eventdir, profile, submask)
            # except:
            #     pass

        isnewprofile = False
        isoutliers   = True
        # Loop through steps 6-8 until no more bad pixels are uncovered
        while(isoutliers == True):
            # STEP 6: Revise variance estimates
            expected    = profile*spectrum
            variance    = np.abs(expected + bg) / Q + v0
            # STEP 7: Mask cosmic ray hits
            stdevs      = np.abs(subdata - expected)*submask / np.sqrt(variance)
            if isplots == 8:
                try:
                    plt.figure(3801)
                    plt.clf()
                    plt.plot(variance[20])
                    plt.figure(3802)
                    plt.clf()
                    plt.plot(variance[:,10])
                    plt.pause(1)
                except:
                    pass
            isoutliers  = False
            if len(stdevs) > 0:
                # Find worst data point in each column
                loc         = np.argmax(stdevs, axis=0)
                # Mask data point if std is > p7thresh
                for i in range(nx):
                    if isplots == 8:
                        try:
                            plt.figure(3803)
                            plt.clf()
                            plt.suptitle(str(i) + "/" + str(nx))
                            plt.plot( subdata[:,i], 'bo')
                            plt.plot(expected[:,i], 'g-')
                            plt.pause(0.01)
                        except:
                            pass
                    if stdevs[loc[i],i] > p7thresh:
                        isnewprofile = True
                        isoutliers   = True
                        submask[loc[i],i] = 0
                        # Generate plot
                        if isplots >= 5:
                            plt.figure(3501)
                            plt.clf()
                            plt.suptitle(f'Integration {n}, Columns {i}/{nx}')
                            #plt.suptitle(str(n) + ", " + str(i) + "/" + str(nx))
                            #print(np.where(submask[:,i])[0])
                            plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], subdata[np.where(submask[:,i])[0],i], 'bo')
                            plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], expected[np.where(submask[:,i])[0],i], 'g-')
                            plt.plot((loc[i]), (subdata[loc[i],i]), 'ro')
                            plt.savefig(eventdir + "figs/fig3501-"+str(n)+"-"+str(i)+"-Subdata.png")
                            if hide_plots:
                                plt.close()
                            else:
                                plt.pause(0.1)
                        # Check for insufficient number of good points
                        if sum(submask[:,i]) < ny/2.:
                            submask[:,i] = 0
            # STEP 8: Extract optimal spectrum
            denom    = np.sum(profile*profile*submask/variance, axis=0)
            denom[np.where(denom == 0)] = np.inf
            spectrum = np.sum(profile*submask*subdata/variance, axis=0) / denom

    # Calculate variance of optimal spectrum
    specvar  = np.sum(profile*submask, axis=0) / denom

    # Return spectrum and uncertainties
    return spectrum, np.sqrt(specvar), submask
