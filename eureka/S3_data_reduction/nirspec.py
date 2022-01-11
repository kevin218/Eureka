# NIRSpec specific rountines go here
import os
import numpy as np
from astropy.io import fits
from . import sigrej, background, nircam
from . import bright2flux as b2f

def read(filename, data, meta):
    '''Reads single FITS file from JWST's NIRCam instrument.

    Parameters
    ----------
    filename:   str
        Single filename to read
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaClass
        The metadata object

    Returns
    -------
    data: DataClass
        The updated data object with the fits data stored inside

    Notes
    -----
    History:

    - November 2012 Kevin Stevenson
        Initial version
    - June 2021 Aarynn Carter/Eva-Maria Ahrer
        Updated for NIRSpec
    '''

    assert isinstance(filename, str)

    # Decide whether to perform the Stage 2 processing ourselves.
    # if stage2_processing:
    # 	# Run pipeline on a *_rateints.fits Stage 1 data product, but avoiding significant subarray trimming.
    # 	stage2_filename = process_to_stage2(filename, do_assignwcs=do_assignwcs, do_extract2d=do_extract2d, do_srctype=do_srctype, do_flatfield=do_flatfield, do_photom=do_photom, delete_files=delete_files)
    # else:
    # 	# Use the input file as is.
    # 	stage2_filename = filename


    # Now we can start working with the data.
    hdulist 		= fits.open(filename)
    data.mhdr 		= hdulist[0].header
    data.shdr 		= hdulist['SCI',1].header

    data.intstart 	= 1
    print('  WARNING: Manually setting INTSTART to 1 for NIRSpec CV3 data.')
    #data.intstart    = data.mhdr['INTSTART']
    data.intend 	= data.mhdr['NINTS']

    data.data 		= hdulist['SCI',1].data
    data.err 		= hdulist['ERR',1].data
    data.dq 		= hdulist['DQ',1].data
    data.wave 		= hdulist['WAVELENGTH',1].data
    data.v0 		= hdulist['VAR_RNOISE',1].data
    data.int_times	= hdulist['INT_TIMES',1].data[data.intstart-1:data.intend]

    # Record integration mid-times in BJD_TDB
    # data.bjdtdb = data.int_times['int_mid_BJD_TDB']
    # There is no time information in the simulated NIRSpec data
    print('  WARNING: The timestamps for the simulated NIRSpec data are currently '
          'hardcoded because they are not in the .fits files themselves')
    data.bjdtdb = np.linspace(data.mhdr['EXPSTART'], data.mhdr['EXPEND'], data.intend)

    # NIRSpec CV3 data has a lot of NaNs in the data and err arrays, which is making life difficult.
    print('  WARNING: Manually changing NaNs from DATA and ERR arrays to 0 for the CV3 data')
    data.err[np.where(np.isnan(data.err))] = np.inf
    data.data[np.where(np.isnan(data.data))] = 0

    return data, meta


def flag_bg(data, meta):
    '''Outlier rejection of sky background along time axis.

    Uses the code written for NIRCam and untested for NIRSpec, but likely to still work

    Parameters
    ----------
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaClass
        The metadata object

    Returns
    -------
    data:   DataClass
        The updated data object with outlier background pixels flagged.
    '''
    return nircam.flag_bg(data, meta)


def fit_bg(data, meta, mask, y1, y2, bg_deg, p3thresh, n, isplots=False):
    '''Fit for a non-uniform background.

    Uses the code written for NIRCam and untested for NIRSpec, but likely to still work
    '''
    return nircam.fit_bg(data, meta, mask, y1, y2, bg_deg, p3thresh, n, isplots=isplots)
