import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm

__all__ = ['BGsubtraction', 'fitbg', 'fitbg2', 'fitbg3']

def BGsubtraction(data, meta, log, isplots):
    """Does background subtraction using inst.fit_bg & background.fitbg

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of MJy/sr or DN/s.
    meta:   MetaClass
        The metadata object.
    log:    logedit.Logedit
        The open log in which notes from this step can be added.
    isplots:    int
        The amount of plots saved; set in ecf.

    Returns
    -------
    data:   DataClass
        Data object containing background subtracted data.
    """
    n_int, bg_y1, bg_y2, subdata, submask = meta.n_int, meta.bg_y1, meta.bg_y2, data.subdata, data.submask

    # Load instrument module
    if meta.inst == 'miri':
        from . import miri as inst
    elif meta.inst == 'nircam':
        from . import nircam as inst
    elif meta.inst == 'nirspec':
        from . import nirspec as inst
    elif meta.inst == 'niriss':
        raise ValueError('NIRISS observations are currently unsupported!')
    else:
        raise ValueError('Unknown instrument {}'.format(meta.inst))

    # Write background
    def writeBG(arg):
        bg_data, bg_mask, n = arg
        subbg[n] = bg_data
        submask[n] = bg_mask
        return

    # Compute background for each integration
    log.writelog('  Performing background subtraction')
    subbg = np.zeros((subdata.shape))
    if meta.ncpu == 1:
        # Only 1 CPU
        for n in tqdm(range(meta.int_start,n_int)):
            # Fit sky background with out-of-spectra data
            writeBG(inst.fit_bg(subdata[n], meta, submask[n], bg_y1, bg_y2, meta.bg_deg, meta.p3thresh, n, isplots))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        args_list = []
        for n in range(meta.int_start,n_int):
            args_list.append((subdata[n], meta, submask[n], bg_y1, bg_y2, meta.bg_deg, meta.p3thresh, n, isplots))
        jobs = [pool.apply_async(func=inst.fit_bg, args=(*args,), callback=writeBG) for args in args_list]
        pool.close()
        for job in tqdm(jobs):
            res = job.get()

    # 9.  Background subtraction
    # Perform background subtraction
    subdata -= subbg

    data.subbg, data.submask, data.subdata = subbg, submask, subdata

    return data

# STEP 3: Fit sky background with out-of-spectra data
def fitbg(dataim, meta, mask, x1, x2, deg=1, threshold=5, isrotate=False, isplots=0):
    '''Fit sky background with out-of-spectra data.

    Parameters
    ----------
    dataim: ndarray
        The data array
    meta:   MetaClass
        The metadata object.
    mask:   ndarray
        A mask array
    x1:     ndarray
    x2:     ndarray
    deg:    int, optional
        Polynomial order for column-by-column background subtraction
        Default is 1.
    threshold:  int, optional
        Sigma threshold for outlier rejection during background subtraction.
        Defaullt is 5.
    isrotate:   bool, optional
        Default is False.
    isplots:    int, optional
        The amount of plots saved; set in ecf. Default is 0.

    Notes
    ------
    History:

    - May 2013
        Removed [::-1] for LDSS3

    - Feb 2014
        Modified x1 and x2 to allow for arrays
    '''
    # Assume x is the spatial direction and y is the wavelength direction
    # Otherwise, rotate array
    if isrotate == 1:
        dataim = dataim[::-1].T
        mask   = mask[::-1].T
    elif isrotate == 2:
        dataim = dataim.T
        mask   = mask.T

    #Convert x1 and x2 to array, if need be
    ny, nx   = np.shape(dataim)
    if type(x1) == int or type(x1) == np.int64:
        x1 = np.zeros(ny,dtype=int)+x1
    if type(x2) == int or type(x2) == np.int64:
        x2 = np.zeros(ny,dtype=int)+x2

    if deg < 0:
        # Calculate median background of entire frame
        # Assumes all x1 and x2 values are the same
        submask = np.concatenate((  mask[:,    :x1[0]].T,  mask[:,x2[0]+1:nx].T)).T
        subdata = np.concatenate((dataim[:,    :x1[0]].T,dataim[:,x2[0]+1:nx].T)).T
        bg      = np.zeros((ny,nx)) + np.median(subdata[np.where(submask)])
    elif deg == None :
        # No background subtraction
        bg      = np.zeros((ny,nx))
    else:
        degs = np.ones(ny)*deg
        # Initiate background image with zeros
        bg      = np.zeros((ny,nx))
        # Fit polynomial to each column
        for j in range(ny):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals    = np.concatenate((range(x1[j]), range(x2[j]+1,nx))).astype(int)
            # If too few good pixels then average
            if (np.sum(mask[j,:x1[j]]) < deg) or (np.sum(mask[j,x2[j]+1:nx]) < deg):
                degs[j] = 0
            while (nobadpixels == False):
                try:
                    goodxvals = xvals[np.where(mask[j,xvals])]
                except:
                    print("****Warning: Background subtraction failed!****")
                    print(j)
                    print(xvals)
                    print(np.where(mask[j,xvals]))
                    return
                dataslice = dataim[j,goodxvals]
                # Check for at least 1 good x value
                if len(goodxvals) == 0:
                    nobadpixels = True      #exit while loop
                    #Use coefficients from previous row
                else:
                    # Fit along spatial direction with a polynomial of degree 'deg'
                    coeffs    = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model     = np.polyval(coeffs, goodxvals)
                    # Calculate residuals and number of sigma from the model
                    residuals = dataslice - model
                    # Simple standard deviation (faster but prone to missing scanned background stars)
                    #stdres = np.std(residuals)
                    # Median Absolute Deviation (slower but more robust)
                    #stdres  = np.median(np.abs(np.ediff1d(residuals)))
                    # Mean Absolute Deviation (good comprimise)
                    stdres  = np.mean(np.abs(np.ediff1d(residuals)))
                    if stdres == 0:
                        stdres = np.inf
                    stdevs    = np.abs(residuals) / stdres
                    # Find worst data point
                    loc       = np.argmax(stdevs)
                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        mask[j,goodxvals[loc]] = 0
                    else:
                        nobadpixels = True      #exit while loop

            # Evaluate background model at all points, write model to background image
            if len(goodxvals) != 0:
                bg[j] = np.polyval(coeffs, range(nx))
                if isplots >= 6:
                    plt.figure(3601)
                    plt.clf()
                    plt.title(str(j))
                    plt.plot(goodxvals, dataslice, 'bo')
                    plt.plot(range(nx), bg[j], 'g-')
                    plt.savefig(meta.outputdir + 'figs/Fig6_BG_'+str(j)+'.png')
                    plt.pause(0.01)

    if isrotate == 1:
        bg   = (bg.T)[::-1]
        mask = (mask.T)[::-1]
    elif isrotate == 2:
        bg   = (bg.T)
        mask = (mask.T)

    return bg, mask

# STEP 3: Fit sky background with out-of-spectra data
def fitbg2(dataim, meta, mask, bgmask, deg=1, threshold=5, isrotate=False, isplots=0):
    '''Fit sky background with out-of-spectra data.

    fitbg2 uses bgmask, a mask for the background region which enables fitting more complex
    background regions than simply above or below a given distance from the trace. This will
    help mask the 2nd and 3rd orders of NIRISS.

    Parameters
    ----------
    dataim: ndarray
        The data array
    meta:   MetaClass
        The metadata object.
    mask:   ndarray
        A mask array
    bgmask: ndarray
        A background mask array.
    deg:    int, optional
        Polynomial order for column-by-column background subtraction.
        Default is 1.
    threshold:  int, optional
        Sigma threshold for outlier rejection during background subtraction.
        Default is 5.
    isrotate:   bool, optional
        Default is False.
    isplots:    int, optional
        The amount of plots saved; set in ecf. Default is 0.

    Notes
    ------
    History:

    - September 2016 Kevin Stevenson
        Initial version
    '''
    # Assume x is the spatial direction and y is the wavelength direction
    # Otherwise, rotate array
    if isrotate == 1:
        dataim = dataim[::-1].T
        mask   = mask[::-1].T
        bgmask = bgmask[::-1].T

    elif isrotate == 2:
        dataim = dataim.T
        mask   = mask.T
        bgmask = bgmask.T

    # Initiate background image with zeros
    ny, nx  = np.shape(dataim)
    bg      = np.zeros((ny,nx))

    if deg < 0:
        # Calculate median background of entire frame
        bg  += np.median(dataim[np.where(mask2)])

    elif deg == None :
        # No background subtraction
        pass
    else:
        degs = np.ones(ny)*deg
        # Fit polynomial to each column
        for j in tqdm(range(ny)):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals   = np.where(bgmask[j] == 1)[0]
            # If too few good pixels on either half of detector then compute average
            if (np.sum(bgmask[j,:int(nx/2)]) < deg) or (np.sum(bgmask[j,int(nx/2):nx]) < deg):
                degs[j] = 0
            while (nobadpixels == False):
                try:
                    goodxvals = xvals[np.where(bgmask[j,xvals])]
                except:
                    print('column: ', j, 'xvals: ', xvals)
                    print(np.where(mask[j,xvals]))
                    return
                dataslice = dataim[j,goodxvals]
                # Check for at least 1 good x value
                if len(goodxvals) == 0:
                    nobadpixels = True      #exit while loop
                    #Use coefficients from previous row

                else:
                    # Fit along spatial direction with a polynomial of degree 'deg'
                    coeffs    = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model     = np.polyval(coeffs, goodxvals)

                    #model = smooth.smooth(dataslice, window_len=window_len, window=windowtype)
                    #model = sps.medfilt(dataslice, window_len)
                    if isplots == 6:
                        plt.figure(3601)
                        plt.clf()
                        plt.title(str(j))
                        plt.plot(goodxvals, dataslice, 'bo')
                        plt.plot(goodxvals, model, 'g-')
                        plt.savefig(meta.outputdir + 'figs/Fig6_BG_'+str(j)+'.png')
                        plt.pause(0.01)

                    # Calculate residuals
                    residuals = dataslice - model

                    # Find worst data point
                    loc         = np.argmax(np.abs(residuals))
                    # Calculate standard deviation of points excluding worst point
                    ind = np.arange(0,len(residuals),1)
                    ind = np.delete(ind, loc)
                    stdres = np.std(residuals[ind])
                    
                    if stdres == 0:
                        stdres = np.inf
                    # Calculate number of sigma from the model
                    stdevs    = np.abs(residuals) / stdres
                    #print(stdevs)

                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        bgmask[j,goodxvals[loc]] = 0
                    else:
                        nobadpixels = True      #exit while loop


                    if isplots == 6:
                        plt.figure(3601)
                        plt.clf()
                        plt.title(str(j))
                        plt.plot(goodxvals, dataslice, 'bo')
                        plt.plot(goodxvals, model, 'g-')
                        plt.pause(0.01)
                        plt.show()

            # Evaluate background model at all points, write model to background image
            if len(goodxvals) != 0:
                bg[j] = np.polyval(coeffs, range(nx))
                #bg[j] = np.interp(range(nx), goodxvals, model)

    if isrotate == 1:
        bg      = (bg.T)[::-1]
        mask    = (mask.T)[::-1]
        bgmask  = (bgmask.T)[::-1]
    elif isrotate == 2:
        bg      = (bg.T)
        mask    = (mask.T)
        bgmask  = (bgmask.T)

    return bg, bgmask#, mask #,variance


def fitbg3(data, isplots=0):
    """
    Fit sky background with out-of-spectra data. Hopefully this is a faster
    routine than fitbg2. (optimized to fit across the x-direction)

    Parameters
    ----------
    dataim : np.ndarray                                          
       Data image to fit the background to.                                                            
    isplots : int, optional                                    
       The amount of plots saved; set in ecf. Default is 0.                                          

    Returns
    -------
    bg : np.ndarray
       Background model.
    """
