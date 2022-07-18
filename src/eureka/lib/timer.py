# $Author: patricio $
# $Revision: 304 $
# $Date: 2010-07-13 11:36:20 -0400 (Tue, 13 Jul 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/timer.py $
# $Id: timer.py 304 2010-07-13 15:36:20Z patricio $

import time
import numpy as np


def hms_time(time, hours=False):
  """
    Convert time (in seconds) to hours:minutes:seconds format. 

    Parameters:
    -----------
    time:  Scalar
           Time (in seconds) to be printed in h:m:s format.
    hours: Boolean
           If True, treat time as hours rather than seconds.

    Returns:
    --------
    A string of the time in h:m:s format.
  """
  # if hours == False: convert from seconds to hours
  scale = 1.0 if hours else 3600.0
  r1, hours   = np.modf(time/scale)
  r2, minutes = np.modf(r1*60.0)
  seconds     = r2 * 60.0

  return (" %2i"%hours + ":%02i"%minutes + ":%05.2f"%seconds)


class Timer:
  """
    Object to handle the progress of a routine and estimate the
    remaining time.

    Methods:
    --------
    init:  initialize a Timer object.
    
    check: Check if one of the progress thresholds has been
           fulfilled, print progress if so.

    Example:
    --------
    >>> import timer as t
    >>> clock = t.Timer(100)
    >>> clock.check(9.5)
    >>> clock.check(10.1)
    progress:  10%   Estimated remaining time (h:m:s):  0:07:32.87
    >>> clock.check(77)
    progress:  75%   Estimated remaining time (h:m:s):  0:00:19.80

    Revision History:
    -----------------
    2010-11-13 patricio  Written by Patricio Cubillos.
                         pcubillos@fulbrightmail.org
  """
  def __init__(self, nsteps, progress=None):
    """
      Initiate an object, set time of initialization.

      Parameters:
      -----------
      nsteps: scalar
              The number of steps to accomplish in order to end the process.
    """
    self.tini =  time.time()
    self.nsteps = nsteps
    if progress is not None:
      self.progress = progress
    else:
      self.progress = np.array([0.1, 0.25, 0.5, 0.75, 1.1])
    self.index   = 0


  def check(self, done, name=""):
    """
      Control the process advance, print the progress if a thresholds has
     been fulfilled.

      Parameters:
      -----------
      done: scalar
            The number of steps accomplished of the process.
      name: string
            An identifying name.
    """
    completed = done * 1.0 / self.nsteps
    if completed > self.progress[self.index]:
      dt   = (time.time() - self.tini)/3600.0
      left = dt/completed * (1-completed)
      tleft = hms_time(left, hours=True)

      self.index = np.sum(self.progress < completed)

      if name != "":
        name = "  (" + name + ")"
      print("progress: %3d"%(self.progress[self.index-1]*100) +
            "%   " + "Estimated remaining time (h:m:s):" + tleft  + name )

  def hms_left(self, done):
    """
      return the time left in h:m:s format.
    """
    if done == 0:
      return "-:--:--"
    completed = done * 1.0 / self.nsteps
    dt   = (time.time() - self.tini)/3600.0
    left = dt/completed * (1-completed)
    tleft = hms_time(left, hours=True)
    return tleft
