
# MIRI specific rountines go here
import numpy as np
from astropy.io import fits
from eureka.S3_data_reduction import sigrej, optspex
from . import bright2flux as b2f
from . import nircam

# Read FITS file from JWST's NIRCam instrument
def read(filename, data):
    '''
    Reads single FITS file from JWST's MIRI instrument.

    Parameters
    ----------
    filename          : Single filename to read
    data              : data object in which the fits data will stored

    Returns
    -------
    data              : updated data object with the fits data stored inside

    History
    -------
    Written by Kevin Stevenson          November 2012
    Updated for NIRCam (KBS)            May 2021
    Updated docs for MIRI (TJB)         Jun 2021

    '''
    assert isinstance(filename, str)

    hdulist = fits.open(filename)

    # Load main and science headers
    data.mhdr    = hdulist[0].header
    data.shdr    = hdulist['SCI',1].header

    data.intstart    = data.mhdr['INTSTART']
    data.intend      = data.mhdr['INTEND']

    data.data    = hdulist['SCI',1].data
    data.err     = hdulist['ERR',1].data
    data.dq      = hdulist['DQ',1].data
    data.wave    = hdulist['WAVELENGTH',1].data
    data.v0      = hdulist['VAR_RNOISE',1].data
    data.int_times = hdulist['INT_TIMES',1].data[data.intstart-1:data.intend]
    
    # MIRI appears to be rotated by 90° compared to NIRCam, so rotating arrays to allow the re-use of NIRCam code
    if data.shdr['DISPAXIS']==2:
        data.data    = np.swapaxes(data.data, 1, 2)
        data.err     = np.swapaxes(data.err , 1, 2)
        data.dq      = np.swapaxes(data.dq  , 1, 2)
        data.wave    = np.swapaxes(data.wave, 0, 1)
        data.v0      = np.swapaxes(data.v0  , 1, 2)
    
    return data

def unit_convert(data, meta, log):
    '''
    Temporary function template that will later convert from MJy/sr to e-
    '''
    
    # currently: s3_reduce.py -> inst (like nircam.py) -> bright2flux.py -> bright2dn. But: data.mhdr['PUPIL'] doesn't
    # exist in MIRI. Todo: Make that instrument-specific and make it work for miri. (Not really doable right now because
    # the MIRI stage 2 calints files don't include wavelength information and you need that in order to correct for the
    # wavelength-dependent response function).
    
    return data, meta

def flag_bg(data, meta):
    '''
    Temporary function template that will later flag outliers in sky background along time axis
    '''

    # Code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)
    
    return nircam.flag_bg(data, meta)

def fit_bg(data, mask, y1, y2, bg_deg, p3thresh, n, isplots=False):
    '''
    Temporary function template that will later fit for non-uniform background
    '''
    
    # Code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)
    
    return nircam.fit_bg(data, mask, y1, y2, bg_deg, p3thresh, n, isplots)
