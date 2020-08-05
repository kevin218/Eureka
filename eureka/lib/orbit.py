#! /usr/bin/env python 
#orbit.py
#Author: Ryan A. Hardy
#Last revison: May 23, 2009
#A set of functions for calculating, constraining, and updating orbital parameters
#affecting the timing of the secondary eclipse.
import math
import numpy as np
#import models
import scipy.optimize

G = 6.674e-11
msun = 1.98892e30
rsun = 6.955e8
rjupiter = 71492000

'''orbit.py contains functions for predicting and deriving parameters of the transit
and secondary eclipse related to the planet's orbit.  

Contents:
+Prediction Functions
    -light_time()        calculates light-time correction
    -secondary_eclipse()    reports all known secondary eclipse parameters and errors
    -duration()        calculates transit/eclipse duration
    -error_duration()    computes transit duration with error estimate
    -limb_time()        computes transit/eclipse limb crossing time
    -ephemeris()        prints a list of eclipse dates
    inclination()        calculates inclination given trnasit and orbital parameters
    -eclipse_phase()    computes phase of secondary eclipse
    -error_eclipse()    computes error assocated with phase of secondary eclipse
+Solution Functions
    -e_duration()        solves for eccentricity given eclipse phase and transit duration 
    -error_e_duration()    same as e_duration, but with errors
    -e_transit()        solves for eccentricity given the ratio between observed and expected transit durations
    -e_transit_eclipse()    solves for eccentricity    
    -ecosomega()        solves for minimum eccentricity, or e*cosine(omega)
    -error_ecosomega()    computes error assocaed with above function
    -observed_phase_error()    adds effects of ephemeris drift to known eclipse error
    -e()            computes eccentricity given omega and phase of secondary eclipse
+Miscellaneous Functions
    -radial_velocity()    computes single-planet radial velocity
    -true_anomaly()        computes true anomaly using Newton's method
    -relativistic_precession()    predicts effect of relativitistic precession
    -GR_eclipse()        predicts change in phase of secondary eclipse given precession
+Defunct Functions and Notes
    -area()
    -phase()
    -period()
    -omega()
    -chi_square()
    -e_omega_fit()
'''
#Prediction Functions

def light_time(a, omega, e, i=np.pi/2, secondary_primary=True):
    '''Computes the light time correction given the longitude of periastron
        and eccentricity of the orbit.  Returns correction in seconds.
        a            =     semimajor axis in meters
        e             =     eccentricity
        omega             =     longitude of periastron in degrees
        i             =     inclination in radians
        secondary_primary    =    Boolean option for light-time distance
                        True for secondary-primary time
                        False for star-secondary time
    '''
    #i = (i/180.0)*np.pi
    omega = (90-omega)/(180/np.pi)
    #a*=1.496e11
    d_secondary = np.sin(i)*a*(1 - e ** 2)/(1 - (e*np.cos(omega)))
    d_total = a*np.sin(i)*((1 - e ** 2)/(1 + (e*np.cos(omega))) + d_secondary/(a*np.sin(i)))
    if secondary_primary == True:
        return d_total/299792458
    else:
        return d_secondary/299792458


def secondary_eclipse(params, b=(0, 0), i=(np.pi/2, 0)):
    '''Produces a report of expected secondary eclipse parameters given 
    what is known from transit and RV data about the planet, star, and orbit.'''
    e, sigma_e = params[0] 
    omega, sigma_o = params[1]
    period, sigma_p = params[2]
    m_star, sigma_ms = params[3]
    r_star, sigma_rs = params[4]
    r_planet, sigma_rp = params[5]
    d_observed, sigma_d=params[6]
    b, sigma_b = b
    i, sigma_i = i
    print("Known Transit Parameters")
    print("Duration: \t\t", d_observed, "+/-", sigma_d, "minutes")
    print("Ingress/Egress: ", limbtime(b, d_observed, r_star, r_planet)[0],"+/-",limbtime(b, d_observed, 
    r_star, r_planet, sigma_rs=sigma_rs, sigma_rp=sigma_rp, sigma_b=sigma_b,
    sigma_d=sigma_d)[1], "minutes")
    print("Impact Parameter: ", b, "+/-", sigma_b)
    print("rp/rs: \t\t", r_planet/r_star, "+/-", sigma_rp**2/r_star**2-sigma_rs**2*r_planet**2/r_star**4)
    primary = False
    print("")
    if b != 0:
        b_secondary = b*(1+e*np.sin(np.pi*omega/180.0))/(1-e*np.sin(np.pi*omega/180.0))
        sigma_b2=sigma_b**2
        sigma_b2+=(np.pi*sigma_o/180.0)**2*((e*np.cos(omega*np.pi/180.0)*(1-e*np.sin(omega*np.pi/180.0))
        +e*np.cos(omega*np.pi/180.0)*(1+e*np.sin(omega*np.pi/180.0)))/(1-e*np.sin(omega*np.pi/180.0))**2)**2
        sigma_b2+=sigma_e**2*(np.sin(np.pi/180.0*omega)*(1-e*np.sin(np.pi/180.0*omega)
        + np.sin(np.pi/180.0*omega)*(1+e*np.sin(np.pi/180.0*omega)))/(1-e*np.sin(np.pi/180.0*omega))**2)**2    
    print("Secondary Eclipse")
    print("Phase: ", eclipse_phase(omega, e), "+/-", error_eclipse(e, sigma_e, omega, sigma_o),"\n")
    print("Calculated Eclipse Model")
    duration1 = error_duration(e, period, omega, m_star, r_star, r_planet, i,  primary,
    b_secondary, sigma_e, sigma_p, sigma_o, sigma_ms, sigma_rs, sigma_rp, sigma_i, sigma_b)
    print("Duration:\t", duration1[0], "+/-", duration1[1], "minutes" )
    print("Ingress/Egress:\t" , limbtime(b, duration1[0], r_star, r_planet)[0], "+/-",limbtime(b, duration1[0], 
    r_star, r_planet, sigma_rs=sigma_rs, sigma_rp=sigma_rp, sigma_b=sigma_b2**0.5,
    sigma_d=duration1[1])[1], "minutes")
    print("Impact Parameter:\t", b_secondary, "+/-", sigma_b2**0.5)
    print("")
    duration2=d_observed*((1+e*np.sin(np.pi*omega/180.0))/(1-e*np.sin(np.pi*omega/180.0)))*(((1+(r_planet/r_star))**2-b_secondary**2)/((1+(r_planet/r_star))**2-b**2))**0.5
    sigma_d2= sigma_d**2
    print("Scaled Eclipse Model")
    print("Duration:\t", duration2, "minutes")
    print("Ingress/Egress:\t", limbtime(b, duration2, r_star, r_planet)[0], "+/-",limbtime(b, duration1[0], 
    r_star, r_planet, sigma_rs=sigma_rs, sigma_rp=sigma_rp, sigma_b=sigma_b2**0.5,
    sigma_d=sigma_d)[1], "minutes")
    print("Impact Parameter:\t", b_secondary, "+/-", sigma_b2**0.5)
    print("")
    print("esin(omega) Upper Limit:",  ((1-(r_planet/r_star))/b-1)/((1-(r_planet/r_star))/b+1))
    

def duration(e, period, omega, m_star, r_star, r_planet, i=np.pi/2, primary = True, b=0):
    '''Computes the duration of transit and secondary eclipse in minutes.
        Equations from Tingley and Sackett 2005.
        e    =    eccentricity
        period     =    period of orbit in days
        omega    =    longitude of periastron in degrees
        m_star    =    mass of star in solar masses
        r_star     =    radius of star in meters
        r_planet=    radius of planet in meters
        i    =    inclination of orbit in radians
        primary    =    will return the duration of the primary transit if true
                and the duration of secondary eclipse if false
        b    =    impact parameter - entered if i is not known'''
    if primary == True:    
        theta = np.pi*(90-omega)/180.0
    else:
        theta = np.pi*(90+omega)/180.0
    #i = np.pi*i/180
    m_star *= msun
    #r_star *= 6.955e8
    #r_planet *= 71492000
    period *= 86400
    G = 6.673e-11
    a = (G*m_star*(period/(2*np.pi))**2)**(1/3.0)
    r = a*(1-e**2)/(1+e*np.cos(theta))
    d = 2*(r_star+r_planet)*(1-e**2)**0.5
    d *= 1/(1 + e*np.cos(theta))
    d *= (period/(2*np.pi*G*m_star))**(1/3.0)
    if np.cos(i)*r/r_star > (r_planet+r_star)/r_star or b > (r_planet+r_star)/r_star:
        print("Occultation Impossible:  Inclination too low.")
        return 0
    elif b == 0:
        #d *= ((1+(r_planet/r_star))**2-(np.cos(i)*r/r_star)**2)**0.5
        d *= (1- ((r*np.cos(i))/(r_planet+r_star))**2)**0.5
        return d/60
    elif b != 0:
        #d *= ((1+(r_planet/r_star))**2-b**2)**0.5
        d *= (1- ((b*r_star)/(r_planet+r_star))**2)**0.5
        return d/60

def error_duration(e, period, omega, m_star, r_star, r_planet, i=np.pi/2, primary = True, 
    b=0, sigma_e=0, sigma_p=0, sigma_o=0, sigma_ms=0, sigma_rs=0, sigma_rp=0, sigma_i=0, sigma_b=0):
    '''Returns the error associated with the function duration.  Inputs are the same, except
        with errors.
        sigma_e    =    uncertainty associated with eccentricity (e)
        sigma_o        =    uncertainty associated with longitude of periastron (omega)
        sigma_p        =    uncertainty associated with period (period)
        sigma_ms    =    uncertainty associated with stellar mass (m_star)
        sigma_rs    =    uncertainty associated with stellar radius (r_star)
        sigma_rp    =    uncertainty associated with planet mass (r_planet)
        sigma_i        =    uncertainty associated with inclination (i)
        sigma_b        =    uncertainty associated with impact paramter (b)'''
    h = 1e-8
    sigma = sigma_e**2*((duration(e+h, period, omega, m_star, r_star, r_planet, i, primary, b)-duration(e-h, period, omega, m_star, r_star, r_planet, i, primary, b))/(2*h))**2
    sigma+= sigma_p**2*((duration(e, period+h, omega, m_star, r_star, r_planet, i, primary, b)-duration(e, period-h, omega, m_star, r_star, r_planet, i, primary, b))/(2*h))**2
    sigma+= sigma_o**2*((duration(e, period, omega+h, m_star, r_star, r_planet, i, primary, b)-duration(e, period, omega-h, m_star, r_star, r_planet, i, primary, b))/(2*h))**2
    sigma+= sigma_ms**2*((duration(e, period, omega, m_star+h, r_star, r_planet, i, primary, b)-duration(e, period, omega, m_star-h, r_star, r_planet, i, primary, b))/(2*h))**2    
    sigma+= sigma_rs**2*((duration(e, period, omega, m_star, r_star*(1+h), r_planet, i, primary, b)-duration(e, period, omega, m_star, r_star*(1-h), r_planet, i, primary, b))/(2*h*r_star))**2    
    sigma+= sigma_p**2*((duration(e, period, omega, m_star+h, r_star*(1+h), r_planet+h, i, primary, b)-duration(e, period, omega, m_star, r_star*(1-h), r_planet-h, i, primary, b))/(2*h*r_planet))**2    
    sigma+= sigma_p**2*((duration(e, period, omega, m_star+h, r_star, r_planet, i, primary, b)-duration(e, period, omega, m_star-h, r_star, r_planet, i, primary, b))/(2*h))**2    
    sigma+= sigma_i**2*((duration(e, period, omega, m_star+h, r_star, r_planet, i+h, primary, b)-duration(e, period, omega, m_star-h, r_star, r_planet, i-h, primary, b))/(2*h))**2    
    sigma+= sigma_b**2*((duration(e, period, omega, m_star+h, r_star, r_planet, i, primary, b+h)-duration(e, period, omega, m_star-h, r_star, r_planet, i, primary, b-h))/(2*h))**2    
    
    return duration(e, period, omega, m_star+h, r_star, r_planet, i, primary, b), sigma**0.5

def limbtime(b, duration, r_star, r_planet, sigma_b=0, sigma_d=0, sigma_rs=0, sigma_rp=0):
    try:
        r = r_planet/r_star    
        tau = 0.5*(1-(((1-r)**2-b**2)/((1+r)**2-b**2))**0.5)
        sigma_r = sigma_rp**2/r_star**2-sigma_rs**2*r_planet**2/r_star**4
        k = -1/2*(((1+r)**2-b**2)/((1-r)**2-b**2))**0.5
        sigma = sigma_r**2*k*(-2*(1-r)*((1+r)**2-b**2)+(2*(1+r)*((1-r)**2-b**2)))/((1+r)**2-b**2)**2
        sigma += sigma_b**2*k*(-2*b*((1+r)**2-b**2)+(2*b*((1-r)**2-b**2)))/((1+r)**2-b**2)**2
        sigma += (sigma_d/duration)**2
        return tau*duration, sigma**0.5
    except:
        print("Grazing transit: Limb crossing time undefined.")
        return 0, 0

def ephemeris(period, current_JD, epoch, n_predictions, eclipse_phase):
    '''Computes a list of secondary eclipse transit dates.
        period         =    period of orbit (days)
        current_JD     =    current Julian Date
        epoch          =    Julian Date of Transit
        n_predictions     =     Number of predictions in the returned list
        eclipse_phase      =    Phase of secondary eclipse'''
    dates = np.zeros(n_predictions)
    while current_JD > epoch:
            epoch+=period
    for n in range(0, n_predictions):
            dates[n] = epoch+(eclipse_phase*period)+n*period
    return dates

def inclination(e, omega, b, period, r_star, m_star, errors=np.zeros((6))):
    '''Calculates inclination.  Error calculations forthcoming.
    parameters
        e    Eccentricity
        omega    argument of periastron in degrees
        r_star    stellar raidus in solar radii
        m_star    mass of star in solar masses
    outputs
        inclination in degrees
        '''
    omega = np.pi*omega/180.0
    a = ((period*86400/(2*np.pi))**2*6.673e-11*m_star*1.989e30)**(1/3.0)
    r_star*=rsun
    r = a*(1-e**2)/(1+e*np.cos(np.pi/2-omega))
    i = np.arccos(b*r_star/r)
    return i*180.0/np.pi

def scaled_eclipse(e, omega, duration, rp_rs, b):
    '''Determines secondary eclipse parameters from known transit parameters.
    Parameters
        e    Eccentricity
        omega    Argument of Periastron
        durationDuration of transit in an units
        rp_rs    ratio of planet and star radii
        b    impact parameter
        errors    array containing respective errors
    Ouputs
        Array:
        d_secondary    Duration of eclipse
        b_secondary    Impact parameter of secondary eclipse
        limb[0]        limb crossing time
        phase        phase of secondary eclipse        
        '''
    phase = eclipse_phase(omega, e)
    omega = np.pi*omega/180.0
    k = (1+e*np.sin(omega))/(1-e*np.sin(omega))
    b_secondary = b*k
    d_secondary = duration*k*np.sqrt(((1+rp_rs)**2-b_secondary**2)/((1+rp_rs)**2-b**2))
    limb = limbtime(b_secondary, duration, 1, rp_rs)
    return d_secondary, b_secondary, limb[0], phase
        

def eclipse_phase(omega, e):
    '''Predicts phase of secondary eclipse given longitude of periastron
        and eccentricity.  Unlike phase(), this method uses Kepler's
        equations to precisely determine the phase of secondary eclipse.
        omega    =    longitude of periastron in degrees
        e    =    eccentricity
    '''
    omega = omega*np.pi/180.0
    a = ((1+e)/(1-e))**0.5
    E_primary = 2*np.arctan(np.tan((3*np.pi/2 - omega)/2)/a)
    E_secondary = 2*np.arctan(np.tan((np.pi/2 - omega)/2)/a)
    M_primary = E_primary - e*np.sin(E_primary)
    M_secondary = E_secondary - e*np.sin(E_secondary)
    phi = (M_primary/(2*np.pi) - M_secondary/(2*np.pi))
    return phi % 1

def error_eclipse(e, e_error, omega, omega_error):
    '''Computes the error associated with the function eclipse_phase().
        Numerical differentiation is used to approximate the partial derivative
        of eclipse_phase().
        e    =    eccentricity(eclipse)/(period) % period
        e_error    =    uncertainty associated with eccentricity
        omega    =    longitude of periastron
        omega_error    uncertainty associated with omega'''
    h = 1e-8
    sigma = e_error**2*((eclipse_phase(omega, e+h)-eclipse_phase(omega, e))/h)**2
    sigma += omega_error**2*((eclipse_phase(omega+h, e)-eclipse_phase(omega, e))/h)**2
    return sigma**0.5

def mandel_geom(params, x):
    midpt, width, rp_rs, b, flux = params
    ingress = limbtime(b, width, 1, rp_rs)[0]
    trpars = np.array([midpt, width, rp_rs**2, ingress, ingress, flux])
    return models.mandelecl(trpars, x)

def mandelecl_orbit(params, x):
    e, omega, i, period, rplanet, rstar, mstar, ecldepth, flux = params
    rplanet *= rjupiter
    rstar *= rsun
    mstar *= msun
    rp_rs = rplanet/rstar
    period*=86400
    a = (G*mstar*(period/(2*np.pi))**2)**(1/3.0)
    r = a*(1-e**2)/(1+e*np.cos(np.pi*(90-omega)/180.0))
    btr = r*np.cos(np.pi/180.0*i)/(rstar)
    trdur = duration(e, period/86400., omega, mstar/msun, rstar, rplanet, i*np.pi/180.)/(1440.*period/86400)
    #Fold data
    x = np.abs(x % 1)
    trlimb = limbtime(btr, trdur, 1, rp_rs)[0]
    ecldur, becl, ecllimb, midpt = scaled_eclipse(e, omega, trdur, rp_rs, btr)
    trpars = np.array([0, trdur, rp_rs**2, trlimb, trlimb, flux])
    eclpars = np.array([midpt, ecldur, ecldepth, ecllimb, ecllimb, flux])
    eclipse = models.mandelecl(eclpars, x)
    transit = models.mandelecl(trpars, x)
    return eclipse*transit/flux

#Solution Functions

def e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i=np.pi/2, primary = True, b=0):
    '''Solves for e and omega using the observed phase of secondary eclipse and 
        the measured duration of either the transit or secondary eclipse.

        eclipse_phase    =    phase of secondary eclipse
        width        =    transit/eclipse duration in minutes
        period        =    orbital period in days
        m_star        =    mass of star in solar masses
        r_star        =    radius of star in solar radii
        r_planet    =    radius of planet in jovian radii
        i        =    inclination in radians
        primary        =    Boolean.  True for transit and False for eclipse
        b        =    impact parameter'''
    
    left = -360.0
    right = 360.0
    h = 1e-8
    x = 0
    while abs(left-right) > 1e-14:
        midpoint = (left+right)/2
        test = (duration(e(eclipse_phase, left), period, left, m_star, r_star, r_planet, i, primary, b)-width)
        test*= (duration(e(eclipse_phase, midpoint), period, midpoint, m_star, r_star, r_planet, i, primary, b)-width)
        if test >= 0:
            left = midpoint
        else:
            right = midpoint
        #print(midpoint)
        x+=1
        if x > 256:
            print("Convergence failure.")
            break
    #sigma = error_e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b,
    #     sigma_phi, sigma_d, sigma_p, sigma_ms, sigma_rs, sigma_rp, sigma_i, sigma_b)
    return e(eclipse_phase, midpoint), midpoint % 360
    
def error_e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i=np.pi/2, primary = True, 
    b=0, sigma_phi=0, sigma_d=0, sigma_p=0, sigma_ms=0, sigma_rs=0, sigma_rp=0, sigma_i=0, sigma_b=0):
    '''Uses e_duration() and known uncertainties to return e and omega with new respective uncertainties.
        Because of the heavy use of numerical methods, this function may take up to 20 seconds to run.
        
        eclipse_phase    =    phase of secondary eclipse
        width        =    transit/eclipse duration in minutes
        period        =    orbital period in days
        m_star        =    mass of star in solar masses
        r_star        =    radius of star in meters
        r_planet    =    radius of planet in meters
        i        =    inclination in degrees
        primary        =    Boolean.  True for transit and False for eclipse
        b        =    impact parameter
        sigma_phi    =    uncertainty associated with eclipse phase (eclipse_phase)
        sigma_d        =    uncertainty associated with transit/eclipse duration (width)
        sigma_p        =    uncertainty associated with period (period)
        sigma_ms    =    uncertainty associated with stellar mass (m_star)
        sigma_rs    =    uncertainty associated with stellar radius (r_star)
        sigma_rp    =    uncertainty associated with planet mass (r_planet)
        sigma_i        =    uncertainty associated with inclination (i)
        sigma_b        =    uncertainty associated with impact paramter (b)
        '''    
    h = 1e-8
    sigma_e = sigma_phi**2*((e_duration(eclipse_phase+h, width, period, m_star, r_star, r_planet, i, primary, b)[0]-e_duration(eclipse_phase-h, width, period, m_star, r_star, r_planet, i, primary, b)[0])/(2*h))**2    
    sigma_e += sigma_d**2*((e_duration(eclipse_phase, width+h, period, m_star, r_star, r_planet, i, primary, b)[0]-e_duration(eclipse_phase, width-h, period, m_star, r_star, r_planet, i, primary, b)[0])/(2*h))**2    
    sigma_e += sigma_p**2*((e_duration(eclipse_phase, width, period+h, m_star, r_star, r_planet, i, primary, b)[0]-e_duration(eclipse_phase, width, period-h, m_star, r_star, r_planet, i, primary, b)[0])/(2*h))**2    
    sigma_e += sigma_ms**2*((e_duration(eclipse_phase, width, period, m_star+h, r_star, r_planet, i, primary, b)[0]-e_duration(eclipse_phase, width, period, m_star-h, r_star, r_planet, i, primary, b)[0])/(2*h))**2    
    sigma_e += sigma_rs**2*((e_duration(eclipse_phase, width, period, m_star, r_star*(1+h), r_planet, i, primary, b)[0]-e_duration(eclipse_phase, width, period, m_star, r_star*(1-h), r_planet, i, primary, b)[0])/(2*h*r_star))**2    
    sigma_e += sigma_rp**2*((e_duration(eclipse_phase, width, period, m_star, r_star, r_planet*(1+h), i, primary, b)[0]-e_duration(eclipse_phase, width, period, m_star, r_star, r_planet*(1-h), i, primary, b)[0])/(2*h*r_planet))**2    
    sigma_e += sigma_i**2*((e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i+h, primary, b)[0]-e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i-h, primary, b)[0])/(2*h))**2    
    sigma_e += sigma_b**2*((e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b+h)[0]-e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b-h)[0])/(2*h))**2    
    
    sigma_o = sigma_phi**2*((e_duration(eclipse_phase+h, width, period, m_star, r_star, r_planet, i, primary, b)[1]-e_duration(eclipse_phase-h, width, period, m_star, r_star, r_planet, i, primary, b)[1])/(2*h))**2    
    sigma_o += sigma_d**2*((e_duration(eclipse_phase, width+h, period, m_star, r_star, r_planet, i, primary, b)[1]-e_duration(eclipse_phase, width-h, period, m_star, r_star, r_planet, i, primary, b)[1])/(2*h))**2    
    sigma_o += sigma_p**2*((e_duration(eclipse_phase, width, period+h, m_star, r_star, r_planet, i, primary, b)[1]-e_duration(eclipse_phase, width, period-h, m_star, r_star, r_planet, i, primary, b)[1])/(2*h))**2    
    sigma_o += sigma_ms**2*((e_duration(eclipse_phase, width, period, m_star+h, r_star, r_planet, i, primary, b)[1]-e_duration(eclipse_phase, width, period, m_star-h, r_star, r_planet, i, primary, b)[1])/(2*h))**2    
    sigma_o += sigma_rs**2*((e_duration(eclipse_phase, width, period, m_star, r_star*(1+h), r_planet, i, primary, b)[1]-e_duration(eclipse_phase, width, period, m_star, r_star*(1-h), r_planet, i, primary, b)[1])/(2*h*r_star))**2    
    sigma_o += sigma_rp**2*((e_duration(eclipse_phase, width, period, m_star, r_star, r_planet*(1+h), i, primary, b)[1]-e_duration(eclipse_phase, width, period, m_star, r_star, r_planet*(1-h), i, primary, b)[1])/(2*h*r_planet))**2    
    sigma_o += sigma_i**2*((e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i+h, primary, b)[1]-e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i-h, primary, b)[1])/(2*h))**2    
    sigma_o += sigma_b**2*((e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b+h)[1]-e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b-h)[1])/(2*h))**2    
    
    return e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b)[0], sigma_e**0.5, e_duration(eclipse_phase, width, period, m_star, r_star, r_planet, i, primary, b)[1], sigma_o**0.5

def e_transit(omega, d_ratio):
    '''Solves for eccentricity given a value of omega and the ratio of the observed 
        transit to that of a circular orbit.
        omega    =    longitude of periastron (degrees)
        d_ratio    =    ratio of observed duration to that of a circular orbit
        '''
    omega *= np.pi/180.0
    e_plus = -2*d_ratio**2*np.sin(omega)
    e_minus = e_plus
    e_plus += (4*d_ratio**4*np.sin(omega)**2-4*((d_ratio*np.sin(omega))**2+1)*(d_ratio**2-1))**0.5    
    e_minus -= (4*d_ratio**4*np.sin(omega)**2-4*((d_ratio*np.sin(omega))**2+1)*(d_ratio**2-1))**0.5
    e_plus /= 2*((d_ratio*np.sin(omega))**2+1)
    e_minus /= 2*((d_ratio*np.sin(omega))**2+1)
    return e_plus, e_minus

def e_transit_eclipse(eclipse_phase, d_ratio):
    '''Solves for e and omega, given the phase of secondary eclipse
        and the ratio of observed transit time to that of a circular orbit.

        eclipse_phase    =    phase of secondary eclipse
        d_ratio        =    observed transit duration divided by that of a circular orbit
        '''
    left = 1e-14
    right = 360
    h = 1e-8
    i = 0
    while abs(left-right) > 1e-14:
        midpoint = (left+right)/2
        test = e(eclipse_phase, left)-e_transit(left, d_ratio)[0]
        test *= e(eclipse_phase, midpoint)-e_transit(midpoint, d_ratio)[0]
        if test > 0:
            left = midpoint % 360
        else:
            right = midpoint % 360
        i += 1
        if i > 64:
            print("Convergence Failure.")
            break
    return e(eclipse_phase, midpoint), midpoint % 360








def ecosomega(phase, emin=True):        
    '''
    Uses the observed phase of secondary eclipse to compute a minimum value of eccentricity,
    equal to e*cos(omega) when omega = 0.  This function first generates an estimate of this quantity and
    uses the secant method to refine this value.  
    
    A negative output indicates that omega is between 90 and 270 degrees while a positive output
    indicates that omega is between 270 and 90 degrees.
    '''
    
    x = np.zeros(50)
    x[0] = np.pi/2*(phase-0.5)
    x[1] = np.pi/2*(phase-0.5)-1e-8
    for i in range(2, 49):
        x[i+1] = x[i] - (eclipse_phase(0, x[i])-phase)*(x[i]-x[i-1])/((eclipse_phase(0, x[i])-phase)-(eclipse_phase(0, x[i-1]) - phase))
        if round(x[i], 12) ==round(x[i-1], 12) or x[i] =='nan' or x[i] == 'inf' or x[i] == 1:
            if emin == True:
                return np.abs(x[i-1])
            else:
                return x[i-1]


def error_ecosomega(transit, eclipse, period, transit_error=0, eclipse_error=0, period_error=0, emin=True):
    '''Computes the error associated with the function ecosomega().
        Returns a tuple containing the minimum eccentricity and the error.
        
        transit        =    time of primary transit in JD
        transit_error    =    uncertainty of primary transit time in days
        period        =    period of orbit in days
        period_error    =    uncertainty of orbital period in days
        eclipse        =    time of secondary eclipse in JD
        eclipse_error     =    uncertainty of timing of secondary eclipse in days'''
    h = 1e-8
    #transit_error += period_error*np.abs(transit - eclipse)/period
    #sigma= np.pi/2*observed_phase_error(transit, eclipse, period, transit_error, eclipse_error/period, period_error)**2
    sigma= observed_phase_error(transit, eclipse, period, transit_error, eclipse_error/period, period_error)**2
    sigma*=((ecosomega(((transit-eclipse)/period % 1)+h)-ecosomega(((transit-eclipse)/period % 1)-h))/(2*h))**2
    if emin == True:
        return ecosomega(((eclipse - transit)/period) % 1, emin =True), sigma**0.5
    else:
        return ecosomega(((eclipse - transit)/period) % 1, emin =True), sigma**0.5    

def observed_phase_error(transit, eclipse, period, sigma_t=0, sigma_phi=0, sigma_p=0):
    '''Computes the error associated with the secondary eclipse in phase units.
        transit    =    time of transit in JD
        eclipse    =    time of eclipse in JD
        period    =    period in days
        sigma_t    =    error associated with transit time
        sigma_phi    =    error associated with phi, in phase units
        sigma_p    =    error associated with period'''
    sigma=sigma_phi**2
    sigma+=(sigma_t/period)**2
    sigma+=(-sigma_p*(transit-eclipse)/period**2)**2
    return sigma**0.5


def e(phase, omega, error_phase=0, error_omega = 0, epsilon=1e-14):
    '''
    Computes the value of e that corresponds to a given value of phase and
    omega using the bisection method to circumvent the instability of Newton's
    method near asymptotes.  
        phase    =    phase of secondary eclipse as a fraction of period
        omega    =    longitude of periastron in degrees
        '''
    left = abs(np.pi/2*(phase-0.5))
    right = 1.0
    h = 1e-8
    while abs(left-right) > epsilon:
        midpoint = (left+right)/2
        test = (eclipse_phase(omega, left)-phase)*(eclipse_phase(omega, midpoint)-phase)
        if test > 0:
            left = midpoint
        else:
            right = midpoint
        #print(midpoint, eclipse_phase(omega, midpoint))
    if error_phase != 0 or error_omega != 0:
        sigma = error_phase**2*((e(phase+h, omega) - e(phase-h, omega))/(2*h))**2
        sigma += error_omega**2*((e(phase, omega+h) - e(phase, omega-h))/(2*h))**2
        return midpoint, sigma**0.5
    else:
        return midpoint

def impact_parameter(duration, limbtime, rp_rs):
    b = np.sqrt(duration**2*-rp_rs+duration*rp_rs**2*limbtime+2*duration*rp_rs*limbtime+duration*limbtime-rp_rs**2*limbtime**2-2*rp_rs*limbtime**2-limbtime**2)/np.sqrt(duration*limbtime-limbtime**2) 
    return b

    
#Miscellaneous Functions

def radial_velocity(phase, period, i, m_star, m_planet, e, omega):
    '''Predicts radial velocity for a given phase measured from the
        transit time.'''    
    omega = omega*np.pi/180.0
    x = ((1+e)/(1-e))**0.5
    E_secondary = 2*np.arctan(np.tan((np.pi/2 - omega)/2)/x)
    M_secondary = E_secondary - e*np.sin(E_secondary)
    phi = M_secondary/(2*np.pi) % 1
    phase+=phi
    #i = np.pi*i/180
    period *= 86400    
    m_star *= 1.98892e30
    m_planet *= 1.8986e27
    M = 2*np.pi*phase
    f = true_anomaly(e, M)
    a = (6.673e-11*(m_star+m_planet)*(period/(2*np.pi))**2)**(1/3.0)
    v = 2*np.pi*a*m_planet*np.sin(i)
    v /= ((m_star+m_planet)*period)
    v *= (np.cos(f+omega) + e*np.cos(omega))
    return v

def true_anomaly(e, M):
    '''x = np.zeros(1000)
    f = np.zeros(len(M))
    for j in range(0, len(M)):
            x[0] = M[j]
        i = 0            
        while abs(x[i]-x[i-1])>1e-14 or i < 512:
            x[i+1] = x[i] + (M[j]+e*np.sin(x[i])-x[i])/(1-e*np.cos(x[i]))
            i+=1
        f[j] = 2*(np.arctan(((1+e)/(1-e))**0.5*np.tan(x[i]/2)))'''
    f = M + (2*e-0.25*e**3)*np.sin(M)+5/4*e**2*np.sin(2*M)+13/12*e**3*np.sin(3*M)
    return f

def relativistic_precession(m_star, period, e):
    '''Returns approximate relativistic precession in degrees per year.
        m_star    =    mass of star in solar masses
        period    =    period of planet in days
        e    =    eccentricity of orbit'''
    m_star*=1.98892e30
    period*=86400
    c = 299792458
    G= 6.673e-11
    p = 6*180*G*m_star/(c**2*((period/(2*np.pi))**2*G*m_star)**(1/3.0)*(1-e**2))
    p*= 365.25/(period/86400)
    return p

def GR_eclipse(m_star, period, e, omega):
    return period*1440*(eclipse_phase(omega+relativistic_precession(m_star, period, e), e)
    -eclipse_phase(omega, e))

def rwprecession(m_star, m_planet, r_planet, period, e, k2p=0.3):
    k = 0.01721209885
    a = ((period*k)**2/m_star)**(1/3.)
    omega_dot = 3.26e-10*(k2p/0.3)*(m_star**1.5)*(1./m_planet)*(r_planet)**5*(a/0.025)**(-13/2.)
    precper = 2*np.pi/(omega_dot) #seconds
    return precper/(86400*365.25), e*period/np.pi*1440

#Defunct Functions
'''
def area(nu, e):
    ''''''Defunct''''''
    i = nu
    A = 0
    while i < nu+np.pi:
        r = (1 - e ** 2)/(1 + (e*np.cos(i)))
        dtheta = 0.0001*np.pi
        A += 0.5*(r ** 2)*dtheta
        i += dtheta
    i = 0
    return A

def phase(omega, e): 
    ''''''Computes phase of secondary eclipse given eccentricity and longitude
        of periastron.  This method is less robust and more computationally
        intensive than the function eclipse_phase(), and should not be used.
        omega     =     longitude of periastron in degrees
        e     =    eccentricity''''''
    omega=(90 -omega)/(180/np.pi) 
    return (area(omega, e))/(np.pi*(1-e**2)**0.5)

def omega(phase, e):
    x = np.zeros(100)
        x[0] = np.acos(ecosomega(phase)/e)
        x[1] = np.acos(ecosomega(phase)/e)-1
        for i in range(2, 70):
            x[i+1] = x[i] - (eclipse_phase(x[i], e)-phase)*(x[i]-x[i-1])/((eclipse_phase(x[i], e)-phase)-(eclipse_phase(x[i-1], e) - phase))         
        if round(x[i], 6) == round(x[i-1], 6) or x[i] =='nan' or x[i] == 'inf' or x[i] > 1:
                    return x[i-1] % 360

def e_omega_fit(eclipse_phase, duration, period, m_star, r_star, r_planet, b):
    scipy.optimize.fmin()
def chi_square(eclipse_phase, duration, period, m_star, r_star, r_planet, b):
    e= 0
    omega = 0
    chi_square = ((eclipse_phase[0]-eclipse_phase(omega, e))/eclipse_phase[1])**2
    chi_square += ((duration[0]-duration(e, period[0], omega[0], m_star[0], r_star[0], r_planet[0], b[0]))/duration[1])**2
    return chi_square

centers = [[282.333, 0.001],
    [470.048, 0.006],
    [477.978, 0.003],
    [496.484, 0.001],
    [501.776, 0.005]]

gj436b=[[22.61612, 0.00037],
[25.26002, 0.00038],
[25.26052, 0.00030],
[43.76657, 0.00026],
[46.40982, 0.00040],
[51.69956, 0.00030]]


def period(dates, initial_guess, n=10000, width=0.1):
    #Define epoch 0 as the date with the least error
    for i in range(0, len(dates)):
        if dates[i][1] < dates[i-1][1]:
            b = dates[i][0]
    #Assign other epochs
    epoch = np.zeros(len(dates))
    for i in range(0, len(dates)):
        epoch[i] = round((dates[i][0] - b)/initial_guess)
        #print(epoch[i])
    #Determine current goodness of fit
    chisq = 0
    for i in range(0, len(dates)):
        chisq += ((dates[i][0]-(initial_guess*epoch[i]+b))/dates[i][1])**2
    #Minimize chi-squared
    k = 0
    up = b+initial_guess
    down=b-initial_guess
    while k < n**0.5:
        guess = initial_guess
        left = guess-width
        right = guess+width
        j = 0
        while j < n**0.5:
            chisql = 0
            chisqr = 0
            for i in range(0, len(dates)):
                chisql += ((dates[i][0]-(left*epoch[i]+up))/dates[i][1])**2
                chisqr += ((dates[i][0]-(right*epoch[i]+up))/dates[i][1])**2
                #chisql += (dates[i][0]-(left*epoch[i]+b))**2
                #chisql += (dates[i][0]-(right*epoch[i]+b))**2
            if chisql < chisqr:
                right = (left+right)/2
            else:
                left = (left+right)/2
            j +=1
        chisqu=0 
        chisqd = 0
        for i in range(0, len(dates)):
            chisqu += ((dates[i][0]-(left*epoch[i]+up))/dates[i][1])**2
            chisqd += ((dates[i][0]-(right*epoch[i]+down))/dates[i][1])**2
        if chisqu < chisqd:
            down = (up+down)/2
        else:
            up = (up+down)/2
        k+=1
    #Compute error as standard error
    sigma = 0
    for i in range(0, len(dates)):
        sigma+=(dates[i][0]-(left*epoch[i]+up))**2
        #print(dates[i][0]-(left*epoch[i]+up))
    sigma /= len(dates)
    sigma=(sigma/len(dates))**0.5
    return left, sigma, down

data = [[0.12, 332, 0.11, 11], [0.16, 351, 0.019, 1.2]]

def phase_fit(phase, error_phase, data):
    test_statistic = 0    
    for i in range(0, 100):
        for j in range(0, len(data)):'''
            

#print(ephemeris(2.64, 2454875, 2454280.78148, 100, .585))
#print("Eclipse phase: \t", eclipse_phase(351, 0.15), "+/-",  error_eclipse(0.15, 0.012, 351, 1.2)
#print("ecosw:\t", ecosomega(0.5868),"+/-",  error_ecosomega(2454455.279241, 1.5e-4, 2.643904, 5e-6, 2454282.33311743, 0.0005*2.643904)[1]
