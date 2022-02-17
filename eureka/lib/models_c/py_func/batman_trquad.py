import batman
import numpy as np

def batman_trquad(params, t, etc):
    '''
    Parameters
    ----------
    t0:         Time of inferior conjunction 
    rprs:       Planet radius / stellar radius
    period:     Orbital period in same units as t
    ars:        Semi-major axis / stellar radius
    cosi:       Cosine of the inclination
    ecc:        Orbital eccentricity
    omega:      Longitude of periastron (degrees) 
    u1:         Linear limb darkening parameter  
    u2:         Quadratic limb darkening parameter  

    Returns
    ---------
    lc          transit light curve normalized to unity

    Revisions
    ---------
    2019-02-19  Laura Kreidberg 
                laura.kreidberg@gmail.com
    '''

    #DEFINE PARAMETERS
    t0, rprs, period, ars, cosi, ecc, omega, u1, u2 = params

    p = batman.TransitParams()

    p.t0 = t0
    p.rp = rprs
    p.per = period 
    p.a = ars
    p.inc = np.arccos(cosi)*180./np.pi
    p.ecc = ecc
    p.w = omega
    p.limb_dark = 'quadratic'
    p.u = [u1, u2]
	
    m = batman.TransitModel(p, t)

    return m.light_curve(p)

