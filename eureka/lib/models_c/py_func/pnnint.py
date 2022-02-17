
import numpy as np

def pnnint(ipparams, posflux, etc = [], retbinflux = False, retbinstd = False):
    """
  This function fits the intra-pixel sensitivity effect using the mean
   within a given binned position (nearest-neighbor interpolation).

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

  Returns
  -------
    1D array, size = # of measurements
    Normalized intrapixel-corrected flux multiplier

  Revisions
  ---------
    2010-06-07    Kevin Stevenson, UCF
                kevin218@knights.ucf.edu
                Original version
    2010-07-07  Kevin
                Added wbfipmask
    """
    #ystep, xstep = ipparams
    y, x, flux, wbfipmask, binfluxmask, kernel, [ny, nx, sy, sx], [binlocnni, binlocbli], \
    [dy1, dy2, dx1, dx2], [ysize, xsize], issmoothing = posflux
    output  = np.zeros(flux.size)
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

    output = binflux[binlocnni]

    if retbinflux == False and retbinstd == False:
        return output
    elif retbinflux == True and retbinstd == True:
        return [output, binflux, binstd]
    elif retbinflux == True:
        return [output, binflux]
    else:
        return [output, binstd]

