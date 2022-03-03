import os
import numpy as np
from astropy.io import fits
from . import background, nircam
from . import bright2flux as b2f
from jwst import datamodels
from gwcs.wcstools import grid_from_bounding_box

def read(filename, data, meta):
    '''Reads single FITS file from JWST's MIRI instrument.

    Parameters
    ----------
    filename:   str
        Single filename to read
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaData
        The metadata object

    Returns
    -------
    data: DataClass
        The updated data object with the fits data stored inside

    Notes
    -----
    History:
    
    - Nov 2012 Kevin Stevenson
        Initial Version
    - May 2021  Kevin Stevenson
        Updated for NIRCam          
    - Jun 2021  Taylor Bell
        Updated docs for MIRI        
    - Jun 2021  Sebastian Zieba
        Updated for MIRI 
    '''
    assert isinstance(filename, str)

    hdulist = fits.open(filename)

    # Load main and science headers
    data.mhdr    = hdulist[0].header
    data.shdr    = hdulist['SCI',1].header

    data.intstart    = data.mhdr['INTSTART']
    data.intend      = data.mhdr['INTEND']
    data.data = hdulist['SCI', 1].data
    data.err = hdulist['ERR', 1].data
    data.dq = hdulist['DQ', 1].data

    print('WARNING: The wavelength for the simulated MIRI data are currently hardcoded '
          'because they are not in the .fits files themselves')

    data.wave = np.tile(wave_MIRI(filename),(data.data.shape[2],1))[:,::-1]    # hdulist['WAVELENGTH', 1].data
    data.v0 = hdulist['VAR_RNOISE', 1].data
    int_times = hdulist['INT_TIMES', 1].data[data.intstart - 1:data.intend]

    # Record integration mid-times in BJD_TDB
    # data.time = data.int_times['int_mid_BJD_TDB']
    print('WARNING: The timestamps for the simulated MIRI data are currently hardcoded '
          'because they are not in the .fits files themselves')
    if data.mhdr['EFFINTTM']==10.3376:
        # There is no time information in the old simulated MIRI data
        # As a placeholder, I am creating timestamps indentical to the ones in STSci-SimDataJWST/MIRI/Ancillary_files/times.dat.txt converted to days
        data.time = np.linspace(0, 17356.28742796742/3600/24, 1680, endpoint=True)[data.intstart - 1:data.intend]
    elif data.mhdr['EFFINTTM']==47.712:
        # A new manually created time array for the new MIRI simulations
        data.time = np.linspace(0, 47.712*(42*44-1)/3600/24, 42*44, endpoint=True)[data.intstart - 1:data.intend-1] # Need to subtract an extra 1 from intend for these data
    meta.time_units = 'BJD_TDB'

    # MIRI appears to be rotated by 90° compared to NIRCam, so rotating arrays to allow the re-use of NIRCam code
    # Having wavelengths increase from left to right on the rotated frame makes life easier
    if data.shdr['DISPAXIS']==2:
        data.data    = np.swapaxes(data.data, 1, 2)[:,:,::-1]
        data.err     = np.swapaxes(data.err , 1, 2)[:,:,::-1]
        data.dq      = np.swapaxes(data.dq  , 1, 2)[:,:,::-1]
        #data.wave    = np.swapaxes(data.wave, 0, 1)[:,:,::-1]
        data.v0      = np.swapaxes(data.v0  , 1, 2)[:,:,::-1]
        if meta.firstFile:
            # If not, we've already done this and don't want to switch it back
            temp         = np.copy(meta.ywindow)
            meta.ywindow = meta.xwindow
            meta.xwindow = data.data.shape[2] - temp[::-1]

    return data, meta


def wave_MIRI(filename):
    '''This code uses the jwst and gwcs packages to get the wavelength information out of the WCS for the MIRI data.

    Parameters
    ----------
    filename:   str
        The filename for the file being read-in.

    Returns
    -------
    lam_x_full: list
        A list of the wavelengths
    '''
    tso = datamodels.open(filename)
    x, y = grid_from_bounding_box(tso.meta.wcs.bounding_box)
    ra, dec, lam = tso.meta.wcs(x, y)

    # This array only contains wavelength information for the BB
    lam_x = [np.mean(lam[i]) for i in range(len(lam))]

    # Including nans for out of BB area (eg for reference pixels) so that length agrees with detector/subarray size
    lam_x_full = [np.float64(np.nan)] * int(y[0, 0]) + lam_x + [np.float64(np.nan)] * int(416 - y[-1, 0] - 1)

    return lam_x_full

def flag_bg(data, meta):
    '''Outlier rejection of sky background along time axis.

    Uses the code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)

    Parameters
    ----------
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaData
        The metadata object

    Returns
    -------
    data:   DataClass
        The updated data object with outlier background pixels flagged.
    '''
    return nircam.flag_bg(data, meta)


def fit_bg(data, meta, n, isplots=False):
    '''Fit for a non-uniform background.

    Uses the code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)
    '''
    return nircam.fit_bg(data, meta, n, isplots=isplots)
