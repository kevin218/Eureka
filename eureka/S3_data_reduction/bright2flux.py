
import numpy as np
import scipy.interpolate as spi
from scipy.constants import arcsec
from astropy.io import fits

def dn2electrons(data, err, v0, gainfile, mhdr, ywindow, xwindow):
    """
    This function converts the data, uncertainty, and variance arrays from
    raw units (DN) to electrons.

    Parameters:
    -----------
    data:       ndarray
                data array of shape ([nx, ny, nimpos, npos]) in units of MJy/sr.
    err:        ndarray
                uncertainties of data (same shape and units).
    v0:         ndarray
                Read noise variance array
    gainfile:   str
                Absolute file name of JWST gain reference file (R_GAIN)
    mhdr:       Record array
                JWST master header
    ywindow:    Tuple
                Start and end of subarray along Y axis
    xwindow:    Tuple
                Start and end of subarray along X axis

    Return:
    -------
    This procedure returns the input arrays in units of electrons.

    Notes:
    ------
    The gain files can be downloaded from CRDS (https://jwst-crds.stsci.edu/browse_db/)

    Modification History:
    ---------------------
    Written by Kevin Stevenson      Jun 2021
    """
    # Subarray parameters
    xstart  = mhdr['SUBSTRT1']
    ystart  = mhdr['SUBSTRT2']
    nx      = mhdr['SUBSIZE1']
    ny      = mhdr['SUBSIZE2']

    # Load gain array in units of e-/ADU
    gain    = fits.getdata(gainfile)[ystart:ystart+ny,xstart:xstart+nx]

    # Gain subarray
    subgain = gain[ywindow[0]:ywindow[1],xwindow[0]:xwindow[1]]

    # Convert to electrons
    data *= subgain
    err  *= subgain
    v0   *= (subgain)**2

    return data, err, v0

def bright2dn(data, err, v0, wave, photfile, mhdr, shdr):
    """
    This function converts the data, uncertainty, and variance arrays from
    brightness units (MJy/sr) to raw units (DN).

    Parameters:
    -----------
    data:       ndarray
                data array of shape ([nx, ny, nimpos, npos]) in units of MJy/sr.
    err:        ndarray
                uncertainties of data (same shape and units).
    v0:         ndarray
                Read noise variance array
    wave:       ndarray
                Pixel dependent wavelength values
    photfile:   str
                Absolute file name of JWST photometric reference file (R_PHOTOM)
    mhdr:       Record array
                JWST master header
    shdr:       Record array
                JWST science header

    Return:
    -------
    This procedure returns the input arrays in units of
    data numbers (DN)).

    Notes:
    ------
    The photometry files can be downloaded from CRDS (https://jwst-crds.stsci.edu/browse_db/)

    Modification History:
    ---------------------
    2021-05-28 kbs       Initial version
    """
    # Load response function and wavelength
    foo = fits.getdata(photfile)
    ind = np.where((foo['filter'] == mhdr['FILTER']) * (foo['pupil'] == mhdr['PUPIL']) * (foo['order'] == 1))[0][0]
    response_wave = foo['wavelength'][ind]
    response_vals = foo['relresponse'][ind]
    igood = np.where(response_wave > 0)[0]
    response_wave = response_wave[igood]
    response_vals = response_vals[igood]
    # Interpolate response at desired wavelengths
    f = spi.interp1d(response_wave, response_vals, 'cubic')
    response = f(wave)

    scalar = shdr['PHOTMJSR']
    # Convert to DN/sec
    data /= scalar * response
    err  /= scalar * response
    v0   /= (scalar * response)**2
    # From DN/sec to DN
    int_time = mhdr['EFFINTTM']
    data *= int_time
    err  *= int_time
    v0   *= int_time

    return data, err, v0

def bright2flux(data, err, v0, pixel_area):
    """
    This function converts the data and uncertainty arrays from
    brightness units (MJy/sr) to flux units (Jy/pix).

    Parameters:
    -----------
    data:    ndarray
             data array of shape ([nx, ny, nimpos, npos]) in units of MJy/sr.
    uncd:    ndarray
             uncertainties of data (same shape and units).
    pixel_area:  ndarray
             Pixel area (arcsec/pix)

    Return:
    -------
    This procedure returns the input arrays Data and Uncd into
    flux units (Jy/pix), if they are defined in the input.

    Notes:
    ------
    The input arrays Data and Uncd are changed in place.

    Modification History:
    ---------------------
    2005-06-20 statia    Written by  Statia Luszcz, Cornell.
                         shl35@cornell.edu
    2005-10-13 jh        Renamed, modified doc, removed posmed, fixed
    	         nimpos default bug (was float rather than int).
    2005-10-28 jh        Updated header to give units being converted
    	         from/to, made srperas value a calculation
    	         rather than a constant, added Allen reference.
    2005-11-24 jh        Eliminated NIMPOS.
    2008-06-28 jh        Allow npos=1 case.
    2010-01-29 patricio  Converted to python. pcubillos@fulbrightmail.org
    2010-11-01 patricio  Documented, and incorporated scipy.constants.
    2021-05-28 kbs       Updated for JWST
    """
    # steradians per square arcsecond
    srperas = arcsec**2.0

    data *= srperas * 1e6 * pixel_area
    err  *= srperas * 1e6 * pixel_area
    v0   *= srperas * 1e6 * pixel_area

    return data, err, v0
