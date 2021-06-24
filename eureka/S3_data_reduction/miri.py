
# MIRI specific rountines go here
import numpy as np
from astropy.io import fits
from eureka.S3_data_reduction import sigrej, optspex
from . import bright2flux as b2f

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

    return data