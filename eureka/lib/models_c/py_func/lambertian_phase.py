import numpy as np

def lambertian_phase(rampparams, t, etc = []):
    """
    This function creates a model that fits a lambertian sphere phase model.
    Uses parts of sincos2 to flatten during eclipse
    
    Parameters
    ----------
    c#a/s#a     : amplitude
    c#o/s#o     : phase/time offset
    p           : period
    c           : vertical offset
    t           : Array of time/phase points
    
    Returns
    -------
    This function returns an array of values.
    
    Revisions
    ---------
    2019-09-03 Erin May
    """

    t0      = rampparams[0]
    p       = rampparams[1]
    cosi    = rampparams[2]
    lo0     = rampparams[3]
    ampl    = rampparams[4]
    c       = rampparams[5]
    midpt   = rampparams[6]
    t14     = rampparams[7]
    t12     = rampparams[8]
    pi      = np.pi
    
    incl = np.arccos(cosi)
    offs = lo0*np.pi/180.
    z_an = np.arccos(-1.0*np.sin(incl)*np.cos(2.*np.pi*(offs+(t-t0)/p)))
    
    flux = ampl*(np.sin(z_an)+(np.pi-z_an)*np.cos(z_an))/np.pi + c
    
    #Flatten sin/cos during eclipse
    iecl = np.where(np.bitwise_or((t-midpt)%p >= p-(t14-t12)/2.,(t-midpt)%p <= (t14-t12)/2.))
    z_an_in=np.arccos(-1.0*np.sin(incl)*np.cos(2.*np.pi*(offs+(midpt-t0)/p)))
    flux[iecl] = ampl*(np.sin(z_an_in)+(np.pi-z_an_in)*np.cos(z_an_in))/np.pi + c
    return flux
