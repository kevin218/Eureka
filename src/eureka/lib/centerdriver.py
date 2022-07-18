# $Author: patricio $
# $Revision: 304 $
# $Date: 2010-07-13 11:36:20 -0400 (Tue, 13 Jul 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/centerdriver.py $
# $Id: centerdriver.py 304 2010-07-13 15:36:20Z patricio $

import numpy as np
from . import err_fasym_c as ctr
#from . import imageedit   as ie
from . import psf_fit     as pf
from . import gaussian    as g

def centerdriver(method, data, guess, trim, radius, size,
                 mask=None, uncd=None, fitbg=1, maskstar=True,
                 expand=5.0, psf=None, psfctr=None):
  """
  Use the center method to find the center of a star in data, starting
  from position guess.
  
  Parameters:
  -----------
  method: string
          Name of the centering method to use.
  data:   2D ndarray
          Array containing the star image.
  guess:  2 elements 1D array
          y, x initial guess position of the target.
  trim:   integer
          Semi-lenght of the box around the target that will be trimmed.
  radius: float
          least asymmetry parameter. See err_fasym_c.
  size:   float
          least asymmetry parameter. See err_fasym_c.  
  mask:   2D ndarray
          A mask array of bad pixels. Same shape of data.
  uncd:   2D ndarray
          An array containing the uncertainty values of data. Same
          shape of data.
          
  Returns:
  --------
  A y,x tuple (scalars) with the coordinates of center of the target
  in data.
  
  Example:
  --------
  nica

  Modification History:
  ---------------------
  23-11-2010 patricio   Written by Patricio Cubillos
                        pcubillos@fulbrightmail.org
  """

  # Default mask: all good
  if mask is None:
    mask = np.ones(np.shape(data))

  # Default uncertainties: flat image
  if uncd is None:
    uncd = np.ones(np.shape(data))

  # Trim the image if requested
  if trim != 0:
    # Integer part of center
    cen = np.rint(guess)
    # Center in the trimed image
    loc = (trim, trim)
    # Do the trim:
    img, msk, err = ie.trimimage(data, cen, loc, mask=mask, uncd=uncd)
  else:
    cen = np.array([0,0])
    loc = np.rint(guess)
    img, msk, err = data, mask, uncd


  # If all data is bad:
  if not np.any(msk):
    raise Exception('Bad Frame Exception!')

  weights = 1.0/np.abs(err)
  extra = []

  # Get the center with one of the methods:
  if   method == 'fgc':
    sy, sx, y, x = g.fitgaussian(img, yxguess=loc, mask=msk, weights=weights,
                         fitbg=fitbg, maskg=maskstar)[0][0:4]
    extra   = sy, sx    #Gaussian 1-sigma half-widths
  elif method == 'col':
    y, x = ctr.col(img)
  elif method == 'lag':
    y, x = ctr.actr(img, loc, asym_rad=radius,
                    asym_size=size, method='gaus')
  elif method == 'lac':
    y, x = ctr.actr(img, loc, asym_rad=radius,
                    asym_size=size, method='col')
  elif method == 'bpf' or method == 'ipf':
    y,x, flux, sky = pf.spitzer_fit(img, msk, weights, psf, psfctr, expand,
                                    method)
    extra = flux, sky

  # Make trimming correction and return
  return ((y, x) + cen - trim), extra
