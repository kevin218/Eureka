# $Author: carthik $
# $Revision: 267 $
# $Date: 2010-06-08 22:33:22 -0400 (Tue, 08 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/time2phase.py $
# $Id: time2phase.py 267 2010-06-09 02:33:22Z carthik $


import numpy as np

def time2phase(time, tzero, period, etype='s'):
  """
    Converts time to phase of a  uniform circular motion.

    Parameters:
    -----------
    times:  scalar or ndarray
            Times to convert.
    tzero:  scalar
            Zero phase time.
    period: scalar  
            The duration of one cycle of the motion.

    Return:
    -------
    This function returns a double-precision array or scalar of the
    same shape as Times, giving the corresponding phases.  All phases
    are in the range [0.0,1.0].  -0.0 is sometimes seen.

    Example:
    --------
    >>> import tme2phase as tp
    >>> print(tp.time2phase(np.array([0.0, 1.2, 3.3]), 1.1, 1.0))
    [ 0.9  0.1  0.2]

    Modification History:
    ---------------------
    2005-10-14 jh        Written by Joseph Harrington, Cornell.
                         jh@oobleck.astro.cornell.edu
    2010-11-01 patricio  Converted to python. pcubillos@fulbrightmail.org
  """

  if etype == 's':
    phase = ( (time - tzero) / period ) % 1.0
    phase[np.where(phase < 0.0)] += 1.0
  elif etype == 'p' or etype == 'o':
    phase = (time - tzero) / period
    phase -= np.floor(np.amin(phase))
  return phase
