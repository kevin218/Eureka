
import numpy as np
import scipy.interpolate as spi

def ipspline(ipparams, position, etc = []):
    """
  This function fits the intra-pixel sensitivity effect using a bicubic spline.

  Parameters
  ----------
    k#:   Knot coefficient
    x,y:  Array of x,y pixel positions and quadrant locations
    etx:  Knot locations

  Returns
  -------
    This function returns an array of y values...

  Revisions
  ---------
  2010-06-08	Kevin Stevenson, UCF  
			kevin218@knights.ucf.edu
		Original Version
    """
    y, x, q = position
    yknots, xknots = etc
    
    tck = spi.bisplrep(xknots.flatten(), yknots.flatten(), ipparams, kx=3, ky=3)
    #print(tck)
    #tck = [yknots, xknots, ipparams, 3, 3]
    #func = spi.interp2d(xknots, yknots, ipparams, kind='cubic'
    output = np.ones(y.size)
    for i in range(y.size):
        output[i] = spi.bisplev(x[i], y[i], tck, dx=0, dy=0)
    
    return output
