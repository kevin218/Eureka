import numpy as np
import spiderman

def spiderman_zhang(params, t, etc = []):
   """
  This function creates a model that fits a physical motivated model based on Zhang et al. 2017, ApJ, 836, 73

  Parameters
  ----------
    	t0:		time of conjunction 
	per:		orbital period
	a_abs:		semi-major axis (AU)
	cos(i):	        cosine of the orbital inclination	
	ecc:		eccentricity
	w:		arg of periastron (deg)
	rp:		planet radius (stellar radii)
	a:		semi-major axis (stellar radii)
	p_u1:		planet linear limb darkening parameter
	p_u2:		planet quadratic limb darkening
	T_s:		stellar Teff
	l1:		short wavelength (m)
	l2:		long wavelength (m)
	xi:		radiative/advective timescale
	T_n:		nightside temperature
	delta_T:	day-night temperature contrast
        npoints:        number of phase bins for light curve interpolation
	

  Returns
  -------
	This function returns planet-to-star flux at each time t. 

  Revisions
  ---------
  2016-11-19 	Laura Kreidberg	
                laura.kreidberg@gmail.com 
                Original version
  2019-02-24	update interpolation, add to github version 
  TODO          add response function, nlayers to etc
   """
   p = spiderman.ModelParams(brightness_model =  'zhang', stellar_model = 'blackbody')
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
   p.xi	   	    = params[13]
   p.T_n	    = params[14]
   p.delta_T	    = params[15]
   npoints          = int(params[16])

   #TODO: add filter path to etc
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


