# $Author: carthik $
# $Revision: 267 $
# $Date: 2010-06-08 22:33:22 -0400 (Tue, 08 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/meanerr.py $
# $Id: meanerr.py 267 2010-06-09 02:33:22Z carthik $

import numpy as np

def meanerr(data, derr, mask=None, err=False, status=False):
  """
    Calculate the error-weighted mean and the error in the
    error-weighted mean of the input data, omitting masked data, NaN
    data or errors, and data whose errors are zero.
  
    Parameters:
    -----------
    data:   ndarray
            The data to average.
    derr:   ndarray
            The 1-sigma uncertainties in data, same shape as data.
    mask:   ndarray
            A 1 indicates the corresponding element of Data is good, a
            0 indicates it is bad, same shape as data.
    err:    boolean
            Set to True to return error in the mean.
    status: boolean
            Set to True to return a bit flag.  If value=0, result is good.
            Bits: 0 = NaN(s) in data.
                  1 = some errors equal zero.
                  2 = masked pixel(s) in data.

    Return:
    -------
    This function returns the error-weighted mean of the unmasked
    elements of Data. If err or status were set to True, return a
    tuple.
  
    Notes:
    ------
    Follows maximum likelihood method (see, e.g., Bevington and
    Robinson 2003, Data Reduction and Error Analysis for the
    Physical Sciences, 3rd ed, McGraw Hill, Ch. 4.).
  
  
    EXAMPLE:
    --------
    >>> import meanerr as men
    >>> nd = 5
    >>> data = np.arange(nd) + 5.0
    >>> derr = np.sqrt(data)
    >>> mask = np.ones(nd)

    >>> print(men.meanerr(data, derr,mask=mask, err=True, status=True))
    (6.7056945183608301, 1.1580755172579058, 0)

    >>> mask[2] = 0
    >>> print(men.meanerr(data, derr,mask=mask, err=True, status=True))
    (6.6359447004608301, 1.2880163722232756, 4)

    >>> data[3] = np.nan
    >>> print(men.meanerr(data, derr,mask=mask, err=True, status=True))
    (6.279069767441861, 1.4467284665112363, 5)

    >>> derr[4] = 0.0
    >>> print(men.meanerr(data, derr,mask=mask, err=True, status=True))
    (5.4545454545454541, 1.6514456476895409, 7)


    MODIFICATION HISTORY:
    ---------------------
    2005-11-15 jh        Written by: Joseph Harrington, Cornell.
                         jh@oobleck.astro.cornell.edu
    2010-11-18 patricio  Wrote in python, docstring added.
                         pcubillos@fulbrightmail.org
  """
  retstatus = status

  if mask is None:
    mask = np.ones(np.shape(data), dtype=bool)

  # Status is good
  status = 0

  # Mask off NaNs
  fin = np.isfinite(data) * np.isfinite(derr)

  # Mask off errors = zero
  nonzero = derr != 0

  # Final Mask
  loc = np.where(fin * nonzero * mask)
  weights = (1.0/derr[loc]**2.0)

  # The returns (a tuple if err or status set to True).
  ret = (np.average(data[loc], weights=weights),)

  if err:
    ret = ret + (np.sqrt(1.0 /np.sum(weights)),)

  if retstatus:
    if not np.all(fin):      # NaNs
      status |= 1
    if not np.all(nonzero):  # errors = zero
      status |= 2
    if not np.all(mask):     # bad data
      status |= 4

    ret = ret + (status,)   

  # return statement
  if len(ret) == 1:
      return ret[0]
  return ret

