"""
 NAME:
	INTEGRATE

 PURPOSE:
	This function numerically integrates the given x,y pair of data points using a basic trapezoidal integration

 CATEGORY:
	Statistics.

 CALLING SEQUENCE:

	Result = INTEGRATE(x, y, a, b)

 INPUTS:
	x:	Array of x values
	y:	Array of y values
	a:	Min x value to begin integration
	b:	Max x value to end integration

  OUTPUTS:
	This function returns the integrated area

 

 MODIFICATION HISTORY:
 	Written by:	Kevin Stevenson, UCF.  		2008-08-11
			kevin218@knights.ucf.edu
    Updated code to use bitwise_and()
                kevin                       2010-02-24
"""

def integrate(x0, y0, a=None, b=None):

    import numpy as np

    if a == None:
        a = min(x0)

    if b == None:
        b = max(x0)

    region = np.where(np.bitwise_and(x0 >= a, x0 <= b))
    x      = x0[region]
    y      = y0[region]
    i      = len(x)
   
    return (sum((x[1:i] - x[0:i-1]) * (y[0:i-1] + y[1:i])/2.0))

