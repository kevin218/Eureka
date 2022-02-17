import numpy as np
import spiderman

def spiderman_spot(params, t, etc = []):
   """
  This function creates a model that fits a hotspot model 

  Parameters
  ----------
    	t0:		time of conjunction 
	per:		orbital period
	a_abs:		semi-major axis (AU)
	inc:		inclinations (deg)
	ecc:		eccentricity
	w:		arg of periastron (deg)
	rp:		planet radius (stellar radii)
	a:		semi-major axis (stellar radii)
	p_u1:		planet linear limb darkening parameter
	p_u2:		planet quadratic limb darkening
	T_s:		stellar Teff
	l1:		blue wavelength (m)
	l2:		red wavelength (m)
	la0:		latitude of hotspot		
	lo0:		longitude of hotspot
	spotsize:	 hot spot radius in degrees	
	spot_T:		the surface temperature of the hotspot as a fraction of temperature of the star
	p_T:		the temperature of the planet that is not in the hotspot

  Returns
  -------
	This function returns planet-to-star flux at each time t. 

  Revisions
  ---------
  2017-09-11 	Laura Kreidberg	
                laura.kreidberg@gmail.com 
                Original version
  2019-02-24	update interpolation, add to github version 
  TODO          add response function, nlayers to etc
   """
   p = spiderman.ModelParams(brightness_model =  'hotspot_t', stellar_model = 'blackbody')
   p.nlayers = 5

   p.t0    	    = params[0]
   p.per       	    = params[1]
   p.a_abs 	    = params[2]
   p.inc	    = np.arccos(params[3])*180./np.pi
   p.ecc	    = params[4]
   p.w	   	    = params[5]
   p.rp	    	    = params[6]
   p.a	   	    = params[7]
   p.p_u1	    = params[8]
   p.p_u2	    = params[9]
   p.T_s	    = params[10]
   p.l1	   	    = params[11]
   p.l2	    	    = params[12]
   p.la0	    = params[13]
   p.lo0	    = params[14]
   p.size	    = params[15]
   p.spot_T	    = params[16]
   p.p_T	    = params[17]
   npoints          = int(params[18])
 
   #p.filter = "/Users/lkreidberg/Desktop/Util/Throughput/spitzer_irac_ch2.txt"

   #calculate light curve over npoints phase bins
   phase = (t - p.t0)/p.per
   phase -= np.round(phase)

   phase_bin = np.linspace(phase.min(), phase.max(), npoints)
   t_bin = phase_bin*p.per + p.t0

   lc_bin = spiderman.web.lightcurve(t_bin, p)

   #interpolate the binned light curve to the original t array
   lc = np.interp(phase, phase_bin, lc_bin)

   return lc 


