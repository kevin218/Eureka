
import numpy as np

def pbilinint(ipparams, posflux, etc = [], retbinflux = False, retbinstd = False):
    """
  This function fits the intra-pixel sensitivity effect using bilinear interpolation to fit mean binned flux vs position.

    Parameters
    ----------
    ipparams :  tuple
                unused
    y :         1D array, size = # of measurements
                Pixel position along y
    x :         1D array, size = # of measurements
                Pixel position along x
    flux :      1D array, size = # of measurements
                Observed flux at each position
    wherebinflux :  1D array, size = # of bins
                    Measurement number assigned to each bin
    gridpt :    1D array, size = # of measurements
                Bin number in which each measurement is assigned
    dy1 :       1D array, size = # of measurements
                (y - y1)/(y2 - y1)
    dy2 :       1D array, size = # of measurements
                (y2 - y)/(y2 - y1)
    dx1 :       1D array, size = # of measurements
                (x - x1)/(x2 - x1)
    dx2 :       1D array, size = # of measurements
                (x2 - x)/(x2 - x1)
    ysize :     int
                Number of bins along y direction
    xsize :     int
                Number of bins along x direction
    smoothing:  boolean
                Turns smoothing on/off

    Returns
    -------
    output :    1D array, size = # of measurements
                Normalized intrapixel-corrected flux multiplier

    Optional
    --------
    binflux :   1D array, size = # of bins
                Binned Flux values

    Notes
    -----
    When there are insufficient points for bilinear interpolation, nearest-neighbor interpolation is used.  The code that handles this is in p6models.py.

    Examples
    --------
    None

    Revisions
    ---------
    2010-06-11  Kevin Stevenson, UCF
                kevin218@knights.ucf.edu
                Original version
    2010-07-07  Kevin
                Added wbfipmask
    """

    y, x, flux, wbfipmask, binfluxmask, kernel, [ny, nx, sy, sx], [binlocnni, binlocbli], \
    [dy1, dy2, dx1, dx2], [ysize, xsize], issmoothing = posflux
    binflux = np.zeros(len(wbfipmask))
    binstd  = np.zeros(len(wbfipmask))
    ipflux  = flux / etc
    wbfm    = np.where(binfluxmask == 1)
    if retbinstd == True:
        for i in wbfm[0]:
            binflux[i] = np.mean(ipflux[wbfipmask[i]])
            binstd[i]  = np.std(ipflux[wbfipmask[i]])
        meanbinflux = np.mean(binflux[wbfm])
        binflux    /= meanbinflux
        binstd     /= meanbinflux
    else:
        for i in wbfm[0]:
            binflux[i] = np.mean(ipflux[wbfipmask[i]])
        binflux /= np.mean(binflux[wbfm])

    #Perform smoothing
    if issmoothing == True:
        binflux = smoothing.smoothing(np.reshape(binflux,    (ysize,xsize)), (ny,nx), (sy,sx),
                                      np.reshape(binfluxmask,(ysize,xsize)), gk=kernel).flatten()
    #Calculate ip-corrected flux using bilinear interpolation
    output = binflux[binlocbli      ]*dy2*dx2 + binflux[binlocbli      +1]*dy2*dx1 + \
             binflux[binlocbli+xsize]*dy1*dx2 + binflux[binlocbli+xsize+1]*dy1*dx1
    #Return fit with or without binned flux
    if retbinflux == False and retbinstd == False:
        return output
    elif retbinflux == True and retbinstd == True:
        return [output, binflux, binstd]
    elif retbinflux == True:
        return [output, binflux]
    else:
        return [output, binstd]

