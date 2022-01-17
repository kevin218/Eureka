# $Author: carthik $
# $Revision: 267 $
# $Date: 2010-06-08 22:33:22 -0400 (Tue, 08 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/medstddev.py $
# $Id: medstddev.py 267 2010-06-09 02:33:22Z carthik $

import numpy as np

def medstddev1d(data, mask=None, medi=False):
  """
    This function computes the stddev of an N-elements masked array
    with respect to the median rather than the standard method of
    using the mean.

    Parameters:
    -----------
    data: 1D ndarray
          An N-element vector of type integer, float or double.
    mask: 1D ndarray
          Mask indicating the good values (ones) and bad values
          (zeros). Same shape of data.
    medi: boolean, optional
          If True return a tuple with (stddev, median) of data. 

    Notes:
    ------
    MEANSTDDEV calculates the median, subtracts it from each value of
    X, then uses this residual to calculate the standard deviation.

    The numerically-stable method for calculating the variance from
    moment.pro doesn't work for the median standard deviation.  It
    only works for the mean, because by definition the residuals from
    the mean add to zero.


    Example:
    --------
    >>> import medstdev as m
    >>> a  = np.array([1,3,4,5,6,7,7])
    >>> std, med = m.medstddev1d(a, medi=True)
    >>> print(median(a))
    5.0
    >>> print(med)
    5.0
    >>> print(std)
    2.2360679775
    
    >>> # use masks
    >>> a    = np.array([1,3,4,5,6,7,7])
    >>> mask = np.array([1,1,1,0,0,0,0])
    >>> std, med = m.medstddev1d(a, mask, medi=True)
    >>> print(std)
    1.58113883008
    >>> print(med)
    3.0
    
    >>> # automatically mask invalid values
    >>> a = np.array([np.nan, 1, 4, np.inf, 6])
    >>> std, med = m.medstddev1d(a, medi=True)
    >>> print(std, med)
    (2.5495097567963922, 4.0)
    
    >>> # critical cases:
    >>> # only one value, return std = 0.0
    >>> a    = np.array([1, 4, 6])
    >>> mask = np.array([0, 0, 1])
    >>> std, med = m.medstddev1d(a, mask, medi=True)
    >>> print(std, med)
    (0.0, 6.0)
    
    >>> # no good values, return std = nan, med = nan
    >>> mask[-1] = 0
    >>> std, med = m.medstddev1d(a, mask, medi=True)
    >>> print(std, med)
    (nan, nan)

    Modification history:
    ---------------------
    2005-01-18  statia    Written by Statia Luszcz.
    2005-01-19  statia    Updated variance calculation according to
                          algorithm in moment.pro, added medi keyword.
    2005-01-20  jh        Header update.  Removed algorithm from
                          moment.pro because it doesn't work for the
                          median.  Added /double. Joe Harrington, 
                          Cornell, jh@oobleck.astro.cornell.edu
    2010-11-05  patricio  Converted to python, documented.
                          pcubillos@fulbrightmail.org
  """
  # flag to return median value
  retmed = medi

  # defult mask, all good:
  if mask is None:
    mask = np.ones(len(data))
  # mask invalid values:
  finite = np.isfinite(data)
  mask *= finite
  data[np.where(finite == False)] = 0
  # number of good values:
  ngood = np.sum(mask)

  # calculate median of good values:
  medi = np.median(data[np.where(mask)])
  # residuals is data - median, masked values don't count:
  residuals = (data - medi)*mask
  # calculate standar deviation:
  std = np.sqrt( np.sum(residuals**2.0) / (ngood - 1.0) )

  # critical case fixes:
  if   ngood == 0:
    std = np.nan
  elif ngood == 1:
    std = 0.0

  # return statement:
  if retmed:
    return (std, medi)
  return std


def reduce(func, data, mask, result1, result2, axis=0):
  """
    Operates a function func on data along a given axis, store the
    results in result.

    Parameters:
    -----------
    func:   function
            the function to apply.
    data:   ndarray
    mask:   ndarray
            Same shape as data.
    result: ndarray
            Varaible to store the result of func. Its dimensions are
            same as data except the one given by axis.
    axis:   int
            The axis on data along which func will operate.

    Modification history:
    ---------------------
    2010-11-05  patricio  Written by Patricio Cubillos
                          pcubillos@fulbrightmail.org
   """
  ndim = np.ndim(data)
  if axis == 0:
    # send axis to last 
    data   = np.swapaxes(data, axis, ndim-1)
    mask   = np.swapaxes(mask, axis, ndim-1)
    result1 = np.swapaxes(result1, axis, ndim-2)
    result2 = np.swapaxes(result2, axis, ndim-2)
    axis = ndim-1

  # reduce first dimension and go over again:
  for i in np.arange(np.shape(data)[0]):
    # base case
    if ndim == 2:
      result1[i], result2[i] = func(data[i], mask[i], True)
    else:
      reduce(func, data[i], mask[i], result1[i], result2[i], axis-1)  


def medstddev(data, mask=None, medi=False, axis=0):
  """
    This function computes the stddev of an n-dimensional ndarray with
    respect to the median along a given axis.

    Parameters:
    -----------
    data: ndarray
          A n dimensional array frmo wich caculate the median standar
          deviation.
    mask: ndarray
          Mask indicating the good and bad values of data.
    medi: boolean
          If True return a tuple with (stddev, median) of data. 
    axis: int
          The axis along wich the median std deviation is calculated.
    
    Examples:
    --------
    >>> import medstddev as m
    >>> b = np.array([[1, 3, 4,  5, 6,  7, 7],
                      [4, 3, 4, 15, 6, 17, 7], 
                      [9, 8, 7,  6, 5,  4, 3]])  
    >>> c = np.array([b, 1-b, 2+b])
    >>> std, med = m.medstddev(c, medi=True, axis=2)
    >>> print(median(c, axis=2))
    [[ 5.  6.  6.]
     [-4. -5. -5.]
     [ 7.  8.  8.]]
    >>> print(med)
    [[ 5.  6.  6.]
     [-4. -5. -5.]
     [ 7.  8.  8.]]
    >>> print(std)
    [[ 2.23606798  6.05530071  2.1602469 ]
     [ 2.23606798  6.05530071  2.1602469 ]
     [ 2.23606798  6.05530071  2.1602469 ]]
    >>> # take a look at the first element of std
    >>> d = c[0,0,:]
    >>> print(d)
    [1, 3, 4, 5, 6, 7, 7]
    >>> print(m.medstddev1d(d))
    2.2360679775
    >>> # See medstddev1d for masked examples

    Modification history:
    ---------------------
    2010-11-05  patricio  Written by Patricio Cubillos
                          pcubillos@fulbrightmail.org
  """

  # flag to return median value
  retmed = medi

  # get shape
  shape = np.shape(data)
  # default mask, all good.
  if mask is None:
    mask = np.ones(shape)

  # base case: 1D
  if len(shape) == 1:
    return medstddev1d(data, mask, retmed)

  newshape = np.delete(shape, axis)
  # results
  std  = np.zeros(newshape)
  medi = np.zeros(newshape)

  # reduce dimensions until 1D case
  reduce(medstddev1d, data, mask, std, medi, axis)

  # return statement:
  if retmed:
    return (std, medi)
  return std