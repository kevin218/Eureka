

def cubicgw(ipparams, width, etc = []):
   """
  This function fits the variation in Gaussian-measured PRF half-widths using a 2D cubic.

  Parameters
  ----------
    x1: linear coefficient in x
    x2: quadratic coefficient in x
    x3: cubic coefficient in x
    y1: linear coefficient in y
    y2: quadratic coefficient in y
    y3: cubic coefficient in y
    c : constant

  Returns
  -------
    returns the flux values for the intra-pixel model

  Revisions
  ---------
  2018-11-16	Kevin Stevenson, STScI  
			    kbs@stsci.edu
		        Original version
   """

   x1       = ipparams[0]
   x2       = ipparams[1]
   x3       = ipparams[2]
   y1       = ipparams[3]
   y2       = ipparams[4]
   y3       = ipparams[5]
   c        = ipparams[6]
   s0       = ipparams[7]
   sy, sx   = width

   return x1*(sx-s0) + x2*(sx-s0)**2 + x3*(sx-s0)**3 + y1*(sy-s0) + y2*(sy-s0)**2 + y3*(sy-s0)**3 + c
