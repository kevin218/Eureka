def calcTrace(x, centroid, grism):
    '''
    Calculates the WFC3 trace given the position of the direct image in physical pixels.

    Parameters
    ----------
    x             : physical pixel values along dispersion direction over which the trace is calculated
    centroid    : [y,x] pair describing the centroid of the direct image

    Returns
    -------
    y             : computed trace

    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    '''
    yref, xref = centroid

    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x         = x[np.newaxis]

    if grism == 'G141':
        #WFC3-2009-17.pdf
        #Table 1: Field dependent trace descriptions for G141.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DYDX_A_0 = [1.96882E+00,    9.09159E-05,    -1.93260E-03]
        DYDX_A_1 = [1.04275E-02,    -7.96978E-06,     -2.49607E-06,     1.45963E-09,    1.39757E-08,    4.84940E-10]
    elif grism == 'G102':
        #WFC3-2009-18.pdf
        #Table 1: Field dependent trace descriptions for G102.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DYDX_A_0 = [-3.55018E-01,    3.28722E-05,     -1.44571E-03]
        DYDX_A_1 = [ 1.42852E-02,     -7.20713E-06,     -2.42542E-06,     1.18294E-09,    1.19634E-08,    6.17274E-10
]
    else:
        print("Unknown filter/grism: " + grism)
        return 0

    DYDX_0 = DYDX_A_0[0] + DYDX_A_0[1]*xref + DYDX_A_0[2]*yref
    DYDX_1 = DYDX_A_1[0] + DYDX_A_1[1]*xref + DYDX_A_1[2]*yref + \
             DYDX_A_1[3]*xref**2 + DYDX_A_1[4]*xref*yref + DYDX_A_1[5]*yref**2

    y        = DYDX_0 + DYDX_1*(x-xref) + yref

    return y

    return

def calibrateLambda(x, centroid, grism):
    '''
    Calculates coefficients for the dispersion solution

    Parameters
    ----------
    x             : physical pixel values along dispersion direction over which the wavelength is calculated
    centroid    : [y,x] pair describing the centroid of the direct image

    Returns
    -------
    y             : computed wavelength values

    History
    -------
    Initial version by LK
    Modified by Kevin Stevenson     November 2012
    '''
    yref, xref = centroid

    if isinstance(yref, float) == False:
        yref    = yref[:,np.newaxis]
        x         = x[np.newaxis]

    if grism == 'G141':
        #WFC3-2009-17.pdf
        #Table 5: Field dependent wavelength solution for G141.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DLDP_A_0 = [8.95431E+03,    9.35925E-02,            0.0,             0.0,             0.0,            0.0]
        DLDP_A_1 = [4.51423E+01,    3.17239E-04,    2.17055E-03,    -7.42504E-07,     3.48639E-07,    3.09213E-07]
    elif grism == 'G102':
        #WFC3-2009-18.pdf
        #Table 5: Field dependent wavelength solution for G102.
        #FINDME: y^2 term not given in Table 5, assuming 0.
        #Term         a0                a1(X)             a2(Y)             a3(X^2)         a4(X*Y)         a5(Y^2)
        DLDP_A_0 = [6.38738E+03,    4.55507E-02,            0.0]
        DLDP_A_1 = [2.35716E+01,    3.60396E-04,    1.58739E-03,    -4.25234E-07,    -6.53726E-08,            0.0]
    else:
        print("Unknown filter/grism: " + grism)
        return 0

    DLDP_0 = DLDP_A_0[0] + DLDP_A_0[1]*xref + DLDP_A_0[2]*yref
    DLDP_1 = DLDP_A_1[0] + DLDP_A_1[1]*xref + DLDP_A_1[2]*yref + \
             DLDP_A_1[3]*xref**2 + DLDP_A_1[4]*xref*yref + DLDP_A_1[5]*yref**2

    y        = DLDP_0 + DLDP_1*(x-xref) + yref

    return y
