
import numpy as np

def quadip4(ipparams, position, etc = []):
   """
  This function fits the intra-pixel sensitivity effect using a 2D quadratic in each pixel quadrant.

  Parameters
  ----------
    a#: quadratic coefficient in y
    b#: quadratic coefficient in x
    c#: coefficient for cross-term 
    d#: linear coefficient in y
    e#: linear coefficient in x
    f#: constant
    *0: first quadrant
    *1: second quadrant
    *2: third quadrant
    *3: fourth quadrant
 
  Returns
  -------
    returns the flux values for the intra-pixel model

  Revisions
  ---------
  2008-08-18	Kevin Stevenson, UCF  
			kevin218@knights.ucf.edu
		Original version
   """

   a0      = ipparams[0]
   b0      = ipparams[1]
   c0      = ipparams[2]
   d0      = ipparams[3]
   e0      = ipparams[4]
   f0      = ipparams[5]
   a1      = ipparams[6]
   b1      = ipparams[7]
   c1      = ipparams[8]
   d1      = ipparams[9]
   e1      = ipparams[10]
   f1      = ipparams[11]
   a2      = ipparams[12]
   b2      = ipparams[13]
   c2      = ipparams[14]
   d2      = ipparams[15]
   e2      = ipparams[16]
   f2      = ipparams[17]
   a3      = ipparams[18]
   b3      = ipparams[19]
   c3      = ipparams[10]
   d3      = ipparams[21]
   e3      = ipparams[22]
   f3      = ipparams[23]
   y, x, q = position

   y0      = y[np.where(q == 0)]
   x0      = x[np.where(q == 0)]
   y1      = y[np.where(q == 1)]
   x1      = x[np.where(q == 1)]
   y2      = y[np.where(q == 2)]
   x2      = x[np.where(q == 2)]
   y3      = y[np.where(q == 3)]
   x3      = x[np.where(q == 3)]

   output  = np.zeros(y.size)

   output[np.where(q == 0)] = a0*y0**2 + b0*x0**2 + c0*y0*x0 + d0*y0 + e0*x0 + f0
   output[np.where(q == 1)] = a1*y1**2 + b1*x1**2 + c1*y1*x1 + d1*y1 + e1*x1 + f1
   output[np.where(q == 2)] = a2*y2**2 + b2*x2**2 + c2*y2*x2 + d2*y2 + e2*x2 + f2
   output[np.where(q == 3)] = a3*y3**2 + b3*x3**2 + c3*y3*x3 + d3*y3 + e3*x3 + f3

   return (output)
