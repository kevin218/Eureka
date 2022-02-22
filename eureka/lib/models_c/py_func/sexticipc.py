

def sexticipc(ipparams, position, etc = []):
   """
  This function fits the intra-pixel sensitivity effect using a 2D 6th-order polynomial,
   with cross terms.

  Parameters
  ----------
    y#:  #-ordered coefficient in y
    x#:  #-ordered coefficient in x
    y2x: cofficient for cross-term xy^2
    x2y: coefficient for cross-term yx^2
    xy:  coefficient for cross-term xy
  
  Returns
  -------
    returns the flux values for the intra-pixel model

  Revisions
  ---------
  2010-02-01	Kevin Stevenson, UCF  
			kevin218@knights.ucf.edu
		Original version
   """

   y6, x6, y5, x5, y4, x4, y3, x3, y2x, x2y, y2, x2, xy, y1, x1, c = ipparams
   y, x, q = position

   return y6*y**6 + x6*x**6 + y5*y**5 + x5*x**5 + y4*y**4 + x4*x**4 + y3*y**3 + x3*x**3 + \
                       y2x*y**2*x + x2y*x**2*y + y2*y**2 + x2*x**2 + xy*x*y + y1*y + x1*x + c

