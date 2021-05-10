
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.signal as sps
import gaussian as g
# reload(g)
import smooth

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
                    plt.figure(4)
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
def fitbg2(dataim, mask, bgmask, deg=1, threshold=5, isrotate=False, isplots=False):
    '''
    Fit sky background with out-of-spectra data
    
    HISTORY
    -------
    Written by Kevin Stevenson      September 2016
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
    mask2   = mask*bgmask
    if deg < 0:
        # Calculate median background of entire frame
        bg  += np.median(data[np.where(mask2)])
    elif deg == None :
        # No background subtraction
        pass
    else:
        degs = np.ones(ny)*deg
        # Fit polynomial to each column
        for j in range(ny):
            nobadpixels = False
            # Create x indices for background sections of frame
            xvals   = np.where(bgmask[j] == 1)[0]
            # If too few good pixels on either half of detector then compute average
            if (np.sum(mask2[j,:nx/2]) < deg) or (np.sum(mask2[j,nx/2:nx]) < deg):
                degs[j] = 0
            while (nobadpixels == False):
                try:
                    goodxvals = xvals[np.where(mask[j,xvals])]
                except:
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
                    if isplots == 6:
                        plt.figure(4)
                        plt.clf()
                        plt.title(str(j))
                        plt.plot(goodxvals, dataslice, 'bo')
                        plt.plot(goodxvals, model, 'g-')
                        plt.pause(0.01)
                    
                    # Calculate residuals
                    residuals   = dataslice - model
                    # Find worst data point
                    loc         = np.argmax(np.abs(residuals))
                    # Calculate standard deviation of points excluding worst point
                    ind         = range(len(residuals))
                    ind.remove(loc)
                    stdres      = np.std(residuals[ind])
                    if stdres == 0:
                        stdres = np.inf
                    # Calculate number of sigma from the model
                    stdevs    = np.abs(residuals) / stdres
                    # Mask data point if > threshold
                    if stdevs[loc] > threshold:
                        mask[j,goodxvals[loc]] = 0
                    else:
                        nobadpixels = True      #exit while loop
            
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
    
    return bg, mask #,variance

# Construct normalized spatial profile using polynomial fits along the wavelength direction
def profile_poly(subdata, mask, deg=3, threshold=10, isplots=False):
    '''
    
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
                plt.figure(3)
                plt.clf()
                plt.suptitle(str(j) + "," + str(iternum))
                plt.plot(dataslice, 'ro')
                plt.plot(dataslice*submask[j], 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.001)
            
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

# Construct normalized spatial profile using a smoothing function
def profile_smooth(subdata, mask, threshold=10, window_len=21, windowtype='hanning', isplots=False):
    '''
    
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
                    plt.figure(3)
                    plt.clf()
                    plt.suptitle(str(j) + "," + str(iternum))
                    plt.plot(dataslice, 'ro')
                    plt.plot(dataslice*submask[j], 'bo')
                    plt.plot(model, 'g-')
                    plt.pause(0.2)
                
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

#Construct normalized spatial profile using median of all data frames
def profile_meddata(data, mask, meddata, threshold=10, isplots=0):
    '''
    
    '''
    #profile = np.copy(meddata*mask)
    profile = np.copy(meddata)
    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)
    
    return profile
    

# Construct normalized spatial profile using wavelets
def profile_wavelet(subdata, mask, wavelet, numlvls, isplots=False):
    '''
    This function performs 1D image denoising using BayesShrink soft thresholding.
    Ref: Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and Compression", 2000
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
            plt.figure(7)
            plt.clf()
            plt.suptitle(str(j))
            plt.plot(subdata[j], 'ro')
            plt.plot(subdata[j]*submask[j], 'bo')
            plt.plot(profile[j], 'g-')
            plt.pause(0.5)
    
    # Enforce positivity
    profile[np.where(profile < 0)] = 0
    # Normalize along spatial direction
    profile /= np.sum(profile, axis=0)
    
    return profile

# Construct normalized spatial profile using wavelets
def profile_wavelet2D(subdata, mask, wavelet, numlvls, isplots=False):
    '''
    This function performs 2D image denoising using BayesShrink soft thresholding.
    Ref: Chang et al. "Adaptive Wavelet Thresholding for Image Denoising and Compression", 2000
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
        plt.figure(7)
        plt.clf()
        #plt.suptitle(str(j) + "," + str(iternum))
        plt.plot(subdata[ny/2], 'ro')
        plt.plot(subdata[ny/2]*submask[ny/2], 'bo')
        plt.plot(profile[ny/2], 'g-')
        plt.figure(8)
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

# Construct normalized spatial profile using a Gaussian smoothing function
def profile_gauss(subdata, mask, threshold=10, guess=None, isplots=False):
    '''
    
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
                plt.figure(3)
                plt.clf()
                plt.suptitle(str(i) + "," + str(iternum))
                plt.plot(dataslice, 'ro')
                plt.plot(dataslice*submask[:,i], 'bo')
                plt.plot(model, 'g-')
                plt.pause(0.5)
            
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

# Extract optimal spectrum with uncertainties
def optimize(subdata, mask, bg, spectrum, Q, v0, p5thresh=10, p7thresh=10, fittype='smooth', window_len=21, deg=3, windowtype='hanning', n=0, iread=0, isplots=False, eventdir='.',meddata=None):
    '''
    
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
        else:
            profile = profile_poly(subdata, submask, deg=deg, threshold=p5thresh)
        #
        if isplots >= 3:
            try:
                plt.figure(1015)
                plt.clf()
                plt.suptitle(str(n)+', '+str(iread))
                plt.imshow(profile*submask, aspect='auto', origin='lower',vmax=0.01)
                #print(profile.min(),profile.max())
                plt.savefig(eventdir+'/figs/fig1015-'+str(n)+'-'+str(iread)+'-Profile.png')
                #plt.pause(0.2)
            except:
                pass
        
        isnewprofile = False
        isoutliers   = True
        # Loop through steps 6-8 until no more bad pixels are uncovered
        while(isoutliers == True):
            # STEP 6: Revise variance estimates
            expected    = profile*spectrum
            variance    = np.abs(expected + bg) / Q + v0
            
            # STEP 7: Mask cosmic ray hits
            stdevs      = np.abs(subdata - expected)*submask / np.sqrt(variance)
            '''
            plt.figure(5)
            plt.clf()
            plt.plot(variance[20])
            plt.figure(6)
            plt.clf()
            plt.plot(variance[:,10])
            plt.pause(1)
            print(variance.shape)
            '''
            isoutliers  = False
            if len(stdevs) > 0:
                # Find worst data point in each column
                loc         = np.argmax(stdevs, axis=0)
                # Mask data point if std is > p7thresh
                for i in range(nx):
                    '''
                    if isplots:
                        plt.figure(2)
                        plt.clf()
                        plt.suptitle(str(i) + "/" + str(nx))
                        plt.plot( subdata[:,i], 'bo')
                        plt.plot(expected[:,i], 'g-')
                        plt.pause(0.01)
                    '''
                    if stdevs[loc[i],i] > p7thresh:
                        isnewprofile = True
                        isoutliers   = True
                        submask[loc[i],i] = 0
                        # Generate plot
                        if isplots >= 5:
                            plt.figure(2)
                            plt.clf()
                            plt.suptitle(str(n) + ", " + str(i) + "/" + str(nx))
                            #print(np.where(submask[:,i])[0])
                            plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], subdata[np.where(submask[:,i])[0],i], 'bo')
                            plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], expected[np.where(submask[:,i])[0],i], 'g-')
                            plt.plot((loc[i]), (subdata[loc[i],i]), 'ro')
                            plt.savefig(eventdir + "/figs/subdata-"+str(n)+"-"+str(i)+".png")
                            plt.pause(0.1)
                        # Check for insufficient number of good points
                        if sum(submask[:,i]) < ny/2.:
                            submask[:,i] = 0
                    """
                    else:
                        if isplots >= 5 and n == 1:
                            plt.figure(2)
                            plt.clf()
                            plt.suptitle(str(n) + ", " + str(i) + "/" + str(nx) + ' GOOD')
                            #print(np.where(submask[:,i])[0])
                            plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], subdata[np.where(submask[:,i])[0],i], 'bo')
                            plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], expected[np.where(submask[:,i])[0],i], 'g-')
                            #plt.plot((loc[i]), (subdata[loc[i],i]), 'ro')
                            #plt.savefig(eventdir + "/figs/subdata-"+str(n)+"-"+str(i)+".png")
                            plt.pause(0.1)
                    """
            # STEP 8: Extract optimal spectrum
            denom    = np.sum(profile*profile*submask/variance, axis=0)
            denom[np.where(denom == 0)] = np.inf
            spectrum = np.sum(profile*submask*subdata/variance, axis=0) / denom
    
    # Calculate variance of optimal spectrum
    specvar  = np.sum(profile*submask, axis=0) / denom
        
    # Return spectrum and uncertainties
    return spectrum, np.sqrt(specvar), submask

'''
plt.figure(3)
plt.clf()
plt.plot(subdata[60], 'o')
plt.plot(profile[60], '-')

plt.figure(4)
plt.clf()
plt.plot(subdata[:,498], 'o')
plt.plot((profile*spectrum)[:,498], '-')

'''
