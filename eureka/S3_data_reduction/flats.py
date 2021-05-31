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
