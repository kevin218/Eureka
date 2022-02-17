import batman
import numpy as np

def batman_ecl(params, t, etc = []):
    '''
    Parameters
    ----------
    eclmidpt:   Eclipse midpt
    fpfs:       Planet-to-star flux ratio
    rprs:       Planet radius / stellar radius
    period:     Orbital period in same units as t
    ars:        Semi-major axis / stellar radius
    cosi:       Cosine of the inclination
    ecc:        Orbital eccentricity
    omega:      Longitude of periastron (degrees) 

    Returns
    ---------
    lc          eclipse light curve normalized to unity

    Revisions
    ---------
    2019-02-19  Laura Kreidberg
                laura.kreidberg@gmail.com
    '''

    #DEFINE PARAMETERS
    eclmidpt, fpfs, rprs, period, ars, cosi, ecc, omega = params

    p = batman.TransitParams()

    p.t_secondary = eclmidpt
    p.fp = fpfs
    p.rp = rprs
    p.per = period 
    p.a = ars
    p.inc = np.arccos(cosi)*180./np.pi
    p.ecc = ecc
    p.w = omega
    p.limb_dark = 'quadratic'
    p.u = [0.0, 0.0]
	
    m = batman.TransitModel(p, t, transittype = "secondary")

    return m.light_curve(p) 

