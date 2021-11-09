# $Author: patricio $
# $Revision: 285 $
# $Date: 2010-06-18 17:59:25 -0400 (Fri, 18 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/centroid.py $
# $Id: centroid.py 285 2010-06-18 21:59:25Z patricio $

import sys
import numpy as np
from . import gaussian as g
import matplotlib.pyplot as plt

'''
Stellar Centroiding Routines
----------------------------
This package contains a collection of routines related
to the finding of the centroid of a stellar image.

Contents:
  ctrguess:  Guesses the initial parameters of the
             stellar centroid in a given image.
  ctrgauss:  Performs centroiding by fitting a 2D Gaussian
             function to a given image.
'''

def ctrguess(data, mask=None, guess=None):
  '''
  Calculates crude initial guesses of parameters required for
  Gaussian centroiding of stellar images.

  Speciffically, this function guesses the flux of the center
  of a star, the array indices of this location, and a rough guess
  of the width of the associated PSF.  This method is not robust
  to bad pixels or any other outlying values.

  Parameters
  ----------
  data : ndarray (2D)
      The image in the form of a 2D array containing the star
      to be centroided.  Works best if this is a small subarray
      of the actual data image.

  mask : ndarray (2D)
      The image in the form of a 2D array containing the star
      to be centroided.  Works best if this is a small subarray
      of the actual data image.

  Returns
  -------
  ght : scalar
      The rough estimate of the height (or max flux) of the
      stars PSF.
  gwd : tuple
      The guessed width of the PSF, in the form (gwdy, gwdx)
      where `gwdy` and `gwdx` are the y and x widths 
      respectively.
  gct : tuple
      The guessed center of the PSF, in the form (gcty, gctx)
      where `gcty` and `gctx` are the y and x center indices 
      respectively.

  Notes
  -----
  Logic adapted from `gaussian.py`

  Revisions
  ---------
  2009-10-27   Christopher Campo, UCF    Initial Version
               ccampo@gmail.com
  '''

  if mask is None:
    mask = np.ones(np.shape(data))

  # Center position guess, looking the max value
  if guess is None:
    gcenter = np.unravel_index(np.argmax(data*mask), np.shape(data))
  else:
    gcenter = int(guess[0]), int(guess[1])
  gheight = data[gcenter]  # height guess

  # sum of the number of pixels that are greater than two
  # sigma of the values in the x and y direction.  This 
  # gives a (very) rough guess, in pixels, how wide the PSF is.
  sigma = np.array( [np.std(data[:, gcenter[1]]),    # y std (of central column)
                     np.std(data[gcenter[0], :])] )  # x std (of central row)

  gwidth = ( np.sum( (data*mask)[:, gcenter[1]] > 2*sigma[0] ),
             np.sum( (data*mask)[gcenter[0], :] > 2*sigma[1] ))

  return (gwidth, gcenter, gheight)


def ctrgauss(data, guess=None, mask=None, indarr=None, trim=None):
  '''
  Finds and records the stellar centroid of a set of images by
  fitting a two dimensional Gaussian function to the data.

  It does not find the average centroid, but instead records
  the centroid of each image in the supplied frame parameters
  array at the supplied indices.  The frame parameters array
  is assumed to have the same number of rows as the number
  of frames in the data cube.

  Parameters
  ----------
  data   : ndarray (2D)
           The stellar image.
  guess  : array_like 
           The initial guess of the position of the star.  Has the form 
           (y, x) of the guess center.
  mask   : ndarray (2D)
           The stellar image.
  indarr : array_like
           The indices of the x and y center columns of the frame
           parameters and the width index.  Defaults to 4, 5, and 6
           respectively.
  trim   : Scalar (positive)
           If trim!=0, trims the image in a box of 2*trim pixels around 
           the guess center. Must be !=0 for 'col' method.

  Returns
  -------
  center : y, x
      The updated frame parameters array.  Contains the centers
      of each star in each image and their average width.

  Revisions
  ---------
  2010-06-23  Patricio E. Cubillos, UCF (pcubillos@fulbrightmail.org)
              Adapted to POET from Chris' routines.
  2009-10-30  Christopher J. Campo, UCF (ccampo@gmail.com)
              Initial version.
  '''

  if guess is None:
    fitguess = ctrguess(data, mask, guess)
    guess = fitguess[1]

  # the pixel of the center
  roundguess = np.round(guess)

  # Trim the image  around the star if requested
  if trim is not None:
    image = np.copy(data[roundguess[0]-trim:roundguess[0]+trim,
                         roundguess[1]-trim:roundguess[1]+trim])
    loc = (trim, trim)
  else:
    image = np.copy(data)
    loc   = roundguess

  # Subtract the median to fit a gaussian.
  image -= np.median(image)

  fitguess =  ((1.0, 1.0), loc, image[loc[0],loc[1]])

  if  indarr is None:
    indarr = np.indices(np.shape(image))

#  print(np.shape(image))
#  print(loc)
#  print(fitguess[2])

#  print(np.shape(image), roundguess)
#  print(np.shape(indarr))

  # Fit the gaussian:
  #(fw, fc, fh, err) = g.fitgaussian(image, indarr, guess=fitguess)
  p, err = g.fitgaussian(image, indarr, guess=fitguess)
  fw = p[0:2]
  fc = p[2:4]
  fh = p[4]
  #FINDME: Hack below to get denoise_cenetering.py working
  #foo = g.fitgaussian(image, indarr, guess=fitguess)
  #fc = foo[0][2:4]

  return ( fc + roundguess - loc )