import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from importlib import reload

__all__ = ['BGsubtraction', 'fitbg', 'fitbg2', 'fitbg3']

def BGsubtraction(data, meta, log, isplots):
    """
    Does background subtraction using inst.fit_bg & optspex.fitbg

    Args:
        dat: Data object
        md: Metadata object
        log: log file
        isplots: amount of plots saved; set in ecf

    Returns:
        Corrects subdata with the background
    """
    n_int, bg_y1, bg_y2, subdata, submask = meta.n_int, meta.bg_y1, meta.bg_y2, data.subdata, data.submask

    # Load instrument module
    exec('from eureka.S3_data_reduction import ' + meta.inst + ' as inst', globals())
    reload(inst)


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
            writeBG(inst.fit_bg(subdata[n], submask[n], bg_y1, bg_y2, meta.bg_deg, meta.p3thresh, n, isplots))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        args_list = []
        for n in range(meta.int_start,n_int):
            args_list.append((subdata[n], submask[n], bg_y1, bg_y2, meta.bg_deg, meta.p3thresh, n, isplots))
        jobs = [pool.apply_async(func=inst.fit_bg, args=(*args,), callback=writeBG) for args in args_list]
        pool.close()
        for job in tqdm(jobs):
            res = job.get()

    # Calculate variance
    # bgerr       = np.std(bg, axis=1)/np.sqrt(np.sum(mask, axis=1))
    # bgerr[np.where(np.isnan(bgerr))] = 0.
    # v0[np.where(np.isnan(v0))] = 0.   # FINDME: v0 is all NaNs
    # v0         += np.mean(bgerr**2)
    # variance    = abs(data) / gain + ev.v0    # FINDME: Gain reference file: 'crds://jwst_nircam_gain_0056.fits'
    # variance    = abs(subdata*submask) / gain + v0

    # 9.  Background subtraction
    # Perform background subtraction
    subdata -= subbg

    data.subbg, data.submask, data.subdata = subbg, submask, subdata

    return data

# STEP 3: Fit sky background with out-of-spectra data
def fitbg(dataim, mask, x1, x2, deg=1, threshold=5, isrotate=False, isplots=False):
    '''
    Fit sky background with out-of-spectra data

    HISTORY
    -------
    Written by Kevin Stevenson
    Removed [::-1] for LDSS3                May 2013
    Modified x1 and x2 to allow for arrays  Feb 2014
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
        #degs[np.where(np.sum(mask[:,    :x1],axis=1) < deg)] = 0
        #degs[np.where(np.sum(mask[:,x2+1:nx],axis=1) < deg)] = 0
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
                    #print(j,ny)
                    nobadpixels = True      #exit while loop
                    #Use coefficients from previous row
                else:
                    # Fit along spatial direction with a polynomial of degree 'deg'
                    coeffs    = np.polyfit(goodxvals, dataslice, deg=degs[j])
                    # Evaluate model at goodexvals
                    model     = np.polyval(coeffs, goodxvals)
                    #model = smooth.smooth(dataslice, window_len=window_len, window=windowtype)
                    #model = sps.medfilt(dataslice, window_len)
                    '''
                    if isplots == 6:
                        plt.figure(4)
                        plt.clf()
                        plt.title(str(j))
                        plt.plot(goodxvals, dataslice, 'bo')
                        plt.plot(goodxvals, model, 'g-')
                        #plt.savefig('Fig6_BG_'+str(j)+'.png')
                        plt.pause(0.01)
                    '''
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
                #bg[j] = np.interp(range(nx), goodxvals, model)
                if isplots == 6:
                    plt.figure(3601)
                    plt.clf()
                    plt.title(str(j))
                    plt.plot(goodxvals, dataslice, 'bo')
                    plt.plot(range(nx), bg[j], 'g-')
                    #plt.savefig('Fig6_BG_'+str(j)+'.png')
                    plt.pause(0.01)

    if isrotate == 1:
        bg   = (bg.T)[::-1]
        mask = (mask.T)[::-1]
    elif isrotate == 2:
        bg   = (bg.T)
        mask = (mask.T)

    return bg, mask #,variance

# STEP 3: Fit sky background with out-of-spectra data
def fitbg2(dataim, mask, bgmask, deg=1, threshold=5, isrotate=0, isplots=False):
    '''
    Fit sky background with out-of-spectra data. Fits background model in
    the y-direction by default.

    Parameters
    ----------
    dataim : np.ndarray
       Data image to fit the background to.
    mask : np.ndarray
       Mask of shape dataim that marks where the orders are.
    bgmask : np.ndarray
       Background mask that marks where the background is. The background
       pixels should equal 0 in the mask, while non-background regions should
       equal > 0 (the exact value does not matter).
    deg : int, optional
       The number of degree polynomial to fit to the background.
       Default is 1 (linear fit).
    threshold : float, optional
       The standard deviation threshold to remove bad background
       pixels. Default is 5.
    isrotate : int, optional
       Rotates the image. Default is 0. isrotate=1 rotates the
       image 270 degrees. isrotate=2 rotates the image 90 degrees.
    isplots : bool, optional
       Plots intermediate steps for the background fitting routine.
       Default is False.

    HISTORY
    -------
    Written by Kevin Stevenson                 September 2016
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

    if deg <= 0:
        # Calculate median background of entire frame
        bg  += np.nanmedian(dataim*bgmask)
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
                        bgmask[j,goodxvals[loc]] = np.nan#0
                    else:
                        nobadpixels = True      #exit while loop


                    if isplots == True:
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

    return bg*bgmask#, mask #,variance


def fitbg3(data, omask, bgmask, deg=1, threshold=5, isrotate=0, isplots=False):
    """
    Fit sky background with out-of-spectra data. Hopefully this is a faster
    routine than fitbg2. (optimized to fit across the x-direction)

    Parameters
    ----------
    dataim : np.ndarray                                          
       Data image to fit the background to.                      
    omask : np.ndarray                                            
       Mask of shape dataim that marks where the orders are.      
    bgmask : np.ndarray                                                      
       Background mask that marks where the background is. The background    
       pixels should equal 0 in the mask, while non-background regions should
       equal > 0 (the exact value does not matter).                   
    deg : int, optional                                               
       The number of degree polynomial to fit to the background.      
       Default is 1 (linear fit).                                     
    threshold : float, optional                                      
       The standard deviation threshold to remove bad background     
       pixels. Default is 5.                                       
    isplots : bool, optional                                      
       Plots intermediate steps for the background fitting routine.
       Default is False.                                          

    Returns
    -------
    bg : np.ndarray
       Background model.
    """
    bg = np.zeros(data.shape)

    # Takes a median background model
    if deg <= 0:
        bg = np.full(bgmask.shape, np.nanmedian(data*bgmask))
        return bg
    # No background modeling
    elif deg == None:
        return bg

    # Fitting the background model with some degree polynomial
    else:

        # quickly masking cosmic rays
        maxim = np.nanmax(data*bgmask, axis=0)
        x,y   = np.where(maxim>=np.nanmedian(maxim)+np.nanstd(maxim))
        bg[:,x,y] = np.nan

        for n in tqdm(range(data.shape[0])):
            for i in range(data.shape[1]):

                nobadpixels = False
                goodyvals = np.where((np.isnan(bgmask[n,i])==False) & 
                                     (np.isnan(data[n,i])==False) )[0]
                x, y = goodyvals+0.0, data[n,i][goodyvals]+0.0
                
                while nobadpixels == False:
                    
                    coeffs    = np.polyfit(x, y, deg=deg)
                    model     = np.polyval(coeffs, x)
                    
                    residuals = y-model
                    
                    outliers = np.abs(residuals) > np.nanstd(residuals)*threshold
                    
                    if isplots:
                        plt.plot(x, residuals, '.')
                        plt.plot(x[outliers], residuals[outliers], '.')
                        plt.title(i)
                        plt.show()
                        
                    if len(residuals[outliers])==0:
                        nobadpixels = True
                    else:
                        x, y = x[~outliers], y[~outliers]
                    
                bg[n,i] = np.polyval(coeffs, np.arange(0,bgmask[n].shape[0],1))

            rx, ry = np.where(np.isnan(bgmask))
            bgmask[rx,ry]=0
        

        flipped = np.ones(omask.shape)
        flipped[omask>0] = 0

        return bg*flipped + data
