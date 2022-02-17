
from numpy import sin, cos, pi

def sincos(rampparams, t, etc = []):
   """
  This function creates a model that fits a sinusoid.

  Parameters
  ----------
    a/b:    amplitude
    p1/p2:	period
    t1/t2:  phase/time offset
    c:      vertical offset
    t:	    Array of time/phase points

  Returns
  -------
	This function returns an array of y values...

  Revisions
  ---------
  2010-08-01	Kevin Stevenson, UCF 
                kevin218@knights.ucf.edu
                Original version
   """

   a     = rampparams[0]
   p1    = rampparams[1]
   t1    = rampparams[2]
   b     = rampparams[3]
   p2    = rampparams[4]
   t2    = rampparams[5]
   c     = rampparams[6]

   return a*sin(2*pi*(t-t1)/p1) + b*cos(2*pi*(t-t2)/p2) + c

