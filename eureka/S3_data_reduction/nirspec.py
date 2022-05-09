# NIRSpec specific rountines go here
import os
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import nircam, sigrej

def read(filename, data, meta):
    '''Reads single FITS file from JWST's NIRCam instrument.

    Parameters
    ----------
    filename:   str
        Single filename to read
    data:   Xarray Dataset
        The Dataset object in which the fits data will be stored
    meta:   MetaClass
        The metadata object

    Returns
    -------
    data: Xarray Dataset
        The updated Dataset object with the fits data stored inside
    meta:   MetaClass
        The metadata object

    Notes
    -----
    History:

    - November 2012 Kevin Stevenson
        Initial version
    - June 2021 Aarynn Carter/Eva-Maria Ahrer
        Updated for NIRSpec
    - Apr 22, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    '''

    assert isinstance(filename, str)

    hdulist = fits.open(filename)

    # Load master and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr']     = hdulist[0].header
    data.attrs['shdr']     = hdulist['SCI',1].header
    data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']
    data.attrs['intend']   = data.attrs['mhdr']['INTEND']
    try:
        data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']
        data.attrs['intend']   = data.attrs['mhdr']['INTEND']
    except:
        print('  WARNING: Manually setting INTSTART to 1 and INTEND to NINTS')
        data.attrs['intstart']  = 1
        data.attrs['intend']    = data.attrs['mhdr']['NINTS']

    sci     = hdulist['SCI',1].data
    err     = hdulist['ERR',1].data
    dq      = hdulist['DQ',1].data
    v0      = hdulist['VAR_RNOISE',1].data
    wave_2d = hdulist['WAVELENGTH',1].data
    int_times = hdulist['INT_TIMES',1].data[data.attrs['intstart']-1:data.attrs['intend']]

    # Record integration mid-times in BJD_TDB
    if len(int_times['int_mid_BJD_TDB']) == 0:
        # There is no time information in the simulated NIRSpec data
        print('  WARNING: The timestamps for the simulated NIRSpec data are currently '
              'hardcoded because they are not in the .fits files themselves')
        time = np.linspace(data.mhdr['EXPSTART'], data.mhdr['EXPEND'], data.intend)
    else:
        time = int_times['int_mid_BJD_TDB']

    # Record units
    flux_units  = data.attrs['shdr']['BUNIT']
    time_units = 'BJD_TDB'
    wave_units = 'microns'

    data['flux'] = xrio.makeFluxLikeDA( sci, time, flux_units, time_units, name='flux')
    data['err']  = xrio.makeFluxLikeDA( err, time, flux_units, time_units, name='err')
    data['dq']   = xrio.makeFluxLikeDA(  dq, time,     "None", time_units, name='dq')
    data['v0']   = xrio.makeFluxLikeDA(  v0, time, flux_units, time_units, name='v0')
    data['wave_2d'] = (['y','x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

    return data, meta

def flag_bg(data, meta):
    '''Outlier rejection of sky background along time axis.

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
    y1, y2, bg_thresh = meta.bg_y1, meta.bg_y2, meta.bg_thresh

    bgdata1 = data.flux[:,  :y1]
    bgmask1 = data.mask[:,  :y1]
    bgdata2 = data.flux[:,y2:  ]
    bgmask2 = data.mask[:,y2:  ]
    bgerr1  = np.ma.median(np.ma.masked_equal(data.err[:,  :y1],0))  # This might not be necessary for real data
    bgerr2  = np.ma.median(np.ma.masked_equal(data.err[:,y2:  ],0))

    estsig1 = [bgerr1 for j in range(len(bg_thresh))]
    estsig2 = [bgerr2 for j in range(len(bg_thresh))]
    # FINDME: KBS removed estsig from inputs to speed up outlier detection.
    # Need to test performance with and without estsig on real data.
    data['mask'][:,  :y1] = sigrej.sigrej(bgdata1, bg_thresh, bgmask1)#, estsig1)
    data['mask'][:,y2:  ] = sigrej.sigrej(bgdata2, bg_thresh, bgmask2)#, estsig2)

    return data

def fit_bg(dataim, datamask, n, meta, isplots=False):
    '''Fit for a non-uniform background.

    Uses the code written for NIRCam.
    '''
    return nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)
