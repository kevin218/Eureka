
import numpy as np

def cosine8(rampparams, t, etc = []):
   """
  This function creates a model that fits a superposition of up to 8 sinusoids.

  Parameters
  ----------
    a#:     amplitude
    p#:	    period
    t#:     phase/time offset
    c:      vertical offset
    t:	    Array of time/phase points

  Returns
  -------
	This function returns an array of y values...

  Revisions
  ---------
  2014-05-14	Kevin Stevenson, UChicago
                kbs@uchicago.edu
                Modified from sinnp.cos.py
   """

   a1,p1,t1 = rampparams[ 0:3]
   a2,p2,t2 = rampparams[ 3:6]
   a3,p3,t3 = rampparams[ 6:9]
   a4,p4,t4 = rampparams[ 9:12]
   a5,p5,t5 = rampparams[12:15]
   a6,p6,t6 = rampparams[15:18]
   a7,p7,t7 = rampparams[18:21]
   a8,p8,t8 = rampparams[21:24]
   c        = rampparams[24]
   pi       = np.pi

   return a1*np.cos(2*pi*(t-t1)/p1) + a2*np.cos(2*pi*(t-t2)/p2) + a3*np.cos(2*pi*(t-t3)/p3) + a4*np.cos(2*pi*(t-t4)/p4) + a5*np.cos(2*pi*(t-t5)/p5) + a6*np.cos(2*pi*(t-t6)/p6) + a7*np.cos(2*pi*(t-t7)/p7) + a8*np.cos(2*pi*(t-t8)/p8) + c

