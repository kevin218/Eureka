
import scipy.interpolate as spi

def vsspline(visparams, y, etc = []):
    """
  This function creates a cubic spline that fits the visit sensitivity.

  Parameters
  ----------
    k#:     knot coefficient
    x:      Array of frame numbers in current visit
    knots:  knot locations

  Returns
  -------
    This function returns an array of y values...

  References
  ----------
    See SI from Harrinton et al. (2007)

  Revisions
  ---------
  2010-04-03	Kevin Stevenson, UCF
			kevin218@knights.ucf.edu
		Original version
    """
    x, knots    = y
    tck = spi.splrep(knots, visparams, k=3)
    #tck = (knots, visparams, 3)
    
    return spi.splev(x, tck, der=0)

