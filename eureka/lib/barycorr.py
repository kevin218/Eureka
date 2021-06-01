# Based on a perl script found on https://renenyffenegger.ch/notes/Wissenschaft/Astronomie/Ephemeriden/JPL-Horizons
# Retrieves vector data of Hubble from JPL's HORIZONS system on https://ssd.jpl.nasa.gov/horizons_batch.cgi (see Web interface on https://ssd.jpl.nasa.gov/horizons.cgi)
# Also helpful: https://github.com/kevin218/POET/blob/master/code/doc/spitzer_Horizons_README.txt

import sys
#sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka/eureka/lib')
import os
import numpy as np
from astropy.io import ascii
#from astropy.time import Time
import urllib.request 
from . import suntimecorr




import os
import numpy as np
from astropy.io import ascii
#from astropy.time import Time
import urllib.request 


# converts mjd into bjd
def to_bjdtdb(event, t_mjd, nvisit):
  
    t_bjd = np.zeros(len(t_mjd))

    horizons_file = horizons_downloader(event, t_mjd, nvisit)

    for i in range(1):
	    t_jd = t_mjd + 2400000.5  # converts time to BJD_TDB; see Eastman et al. 2010 equation 4
	    t_jd = t_jd + (32.184) / (24.0 * 60.0 * 60.0)
	    t_bjd = t_jd + (suntimecorr.suntimecorr(event.ra, event.dec, t_jd, horizons_file, verbose=False)) / (60.0 * 60.0 * 24.0)
    
    return t_bjd


#downloads the horizons file from JPL Horizons
def horizons_downloader(event, t_mjd, nvisit):
    settings = [
	    "COMMAND= -139479", #Gaia     #JWST is  [-170]
	    "CENTER= 500@0", #Solar System Barycenter (SSB) [500@0]
	    "MAKE_EPHEM= YES",
	    "TABLE_TYPE= VECTORS",
	    #"START_TIME= $ARGV[0]",
	    #"STOP_TIME= $ARGV[1]",
	    "STEP_SIZE= 3m", # 5 Minute interval 
	    "OUT_UNITS= KM-S",
	    "REF_PLANE= FRAME",
	    "REF_SYSTEM= J2000",
	    "VECT_CORR= NONE",
	    "VEC_LABELS= YES",
	    "VEC_DELTA_T= NO",
	    "CSV_FORMAT= NO",
	    "OBJ_DATA= YES",
	    "VEC_TABLE= 3"]

    #Replacing symbols for URL encoding
    for i, setting in enumerate(settings):
        settings[i] = settings[i].replace(" =", "=").replace("= ", "=")
        settings[i] = settings[i].replace(" ", "%20")
        settings[i] = settings[i].replace("&", "%26")
        settings[i] = settings[i].replace(";", "%3B")
        settings[i] = settings[i].replace("?", "%3F")

    settings = '&'.join(settings)
    settings = 'https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&' + settings
    #print(settings)


    dirname = event.workdir + '/horizons'
    if not os.path.exists(dirname): os.makedirs(dirname)

    for i in range(1): 
        print('Retrieving Horizons file for visit {0}/{1}'.format(i, 0))
        t_mjd_visit = t_mjd
        t_start = min(t_mjd_visit) + 2400000.5 - 10/(24*60) #Start of Horizons file one minute before first exposure in visit
        t_end = max(t_mjd_visit) + 2400000.5 + 10/(24*60) #End of Horizons file one minute after last exposure in visit

        set_start = "START_TIME=JD{0}".format(t_start)
        set_end = "STOP_TIME=JD{0}".format(t_end)

        settings_new = settings + '&' + set_start + '&' + set_end

        urllib.request.urlretrieve(settings_new, dirname + '/horizons_results_v{0}.txt'.format(i))

    return dirname + '/horizons_results_v{0}.txt'.format(i)
