

def cubicip(ipparams, position, etc = []):
   """
  This function fits the intra-pixel sensitivity effect using a 2D cubic.

  Parameters
  ----------
    a: cubic coefficient in y
    b: cubic coefficient in x
    c: coefficient of cross-term xy^2
    d: coefficient of cross-term yx^2
    e: quadratic coefficient in y
    f: quadratic coefficient in x
    g: coefficient of cross-term xy
    h: linear coefficient in y
    i: linear coefficient in x
    j: constant

  Returns
  -------
    returns the flux values for the intra-pixel model

  Revisions
  ---------
  2008-07-08	Kevin Stevenson, UCF  
			kevin218@knights.ucf.edu
		Original version
   """

   a       = ipparams[0]
   b       = ipparams[1]
   c       = ipparams[2]
   d       = ipparams[3]
   e       = ipparams[4]
   f       = ipparams[5]
   g       = ipparams[6]
   h       = ipparams[7]
   i       = ipparams[8]
   j       = ipparams[9]
   y, x, q = position

   return a*y**3 + b*x**3 + c*y**2*x + d*y*x**2 + e*y**2 + f*x**2 + g*y*x + h*y + i*x + j

