
import numpy as np

def sindecay(rampparams, x, etc = []):
   """
  This function creates a model that fits a sinusoidal decay.

  Parameters
  ----------
    x0: phase/time offset
    a:	amplitude
    b:	exponential constant
    c:	period
    d:  vertical offset
    x:	Array of time/phase points

  Returns
  -------
	This function returns an array of y values...

  Revisions
  ---------
  2009-07-26	Kevin Stevenson, UCF 
			kevin218@knights.ucf.edu
		Original version
   """

   x0    = rampparams[0]
   a     = rampparams[1]
   b     = rampparams[2]
   c     = rampparams[3]
   d     = rampparams[4]
   pi    = np.pi

   return a*np.exp(b*x)*np.cos(2*pi*(x-x0)/c) + d
