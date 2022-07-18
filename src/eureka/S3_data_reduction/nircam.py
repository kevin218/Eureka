# NIRCam specific rountines go here
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import sigrej, background
from ..lib.util import read_time
from tqdm import tqdm


def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's NIRCam instrument.

    Parameters
    ----------
    filename : str
        Single filename to read.
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The current log.

    Notes
    -----
    History:

    - November 2012 Kevin Stevenson
        Initial version
    - May 2021 KBS
        Updated for NIRCam
    - July 2021
        Moved bjdtdb into here
    - Apr 20, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    '''
    hdulist = fits.open(filename)

    # Load master and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header
    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data
    data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']
    data.attrs['intend'] = data.attrs['mhdr']['INTEND']
    int_times = hdulist['INT_TIMES', 1].data[data.attrs['intstart'] - 1: data.attrs['intend']]
    if hdulist[0].header['CHANNEL'] == 'SHORT':  # Photometry will have "SHORT" as CHANNEL
        meta.photometry = True
        data.attrs['shdr']['DISPAXIS'] = 1 # This argument does not exist for photmetry data. Added it here so that code in other sections doesnt have to be changed
        #time = np.linspace(hdulist[0].header['EXPSTART'], hdulist[0].header['EXPEND'], (hdulist[0].header['INTEND']-hdulist[0].header['INTSTART']+1))
        if hdulist[0].header['FILTER'] == 'F210M': #TODO make this better for all filters
            wave_2d = np.ones_like(sci[0])*2.1

    elif hdulist[0].header['CHANNEL'] == 'LONG':  # Spectroscopy will have "LONG" as CHANNEL
        meta.photometry = False
        wave_2d = hdulist['WAVELENGTH', 1].data


    # Record integration mid-times in BJD_TDB
    if (hasattr(meta, 'time_file') and meta.time_file is not None):
        time = read_time(meta, data)
    else:
        time = int_times['int_mid_BJD_TDB']

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    time_units = 'BJD_TDB'
    wave_units = 'microns'

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq')
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0')
    data['wave_2d'] = (['y', 'x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

    return data, meta, log


def flag_bg(data, meta, log):
    '''Outlier rejection of sky background along time axis.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with outlier background pixels flagged.
    '''
    log.writelog('  Performing background outlier rejection...',
                 mute=(not meta.verbose))

    meta.bg_y2 = meta.src_ypos + meta.bg_hw
    meta.bg_y1 = meta.src_ypos - meta.bg_hw
    
    bgdata1 = data.flux[:, :meta.bg_y1]
    bgmask1 = data.mask[:, :meta.bg_y1]
    bgdata2 = data.flux[:, meta.bg_y2:]
    bgmask2 = data.mask[:, meta.bg_y2:]
    # FINDME: KBS removed estsig from inputs to speed up outlier detection.
    # Need to test performance with and without estsig on real data.
    if hasattr(meta, 'use_estsig') and meta.use_estsig:
        bgerr1 = np.median(data.err[:, :meta.bg_y1])
        bgerr2 = np.median(data.err[:, meta.bg_y2:])
        estsig1 = [bgerr1 for j in range(len(meta.bg_thresh))]
        estsig2 = [bgerr2 for j in range(len(meta.bg_thresh))]
    else:
        estsig1 = None
        estsig2 = None
    data['mask'][:, :meta.bg_y1] = sigrej.sigrej(bgdata1, meta.bg_thresh,
                                                 bgmask1, estsig1)
    data['mask'][:, meta.bg_y2:] = sigrej.sigrej(bgdata2, meta.bg_thresh,
                                                 bgmask2, estsig2)

    return data


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Parameters
    ----------
    dataim : ndarray (2D)
        The 2D image array.
    datamask : ndarray (2D)
        An array of which data should be masked.
    n : int
        The current integration.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
        The plotting verbosity, by default 0.

    Returns
    -------
    bg : ndarray (2D)
        The fitted background level.
    mask : ndarray (2D)
        The updated mask after background subtraction.
    n : int
        The current integration number.
    """
    bg, mask = background.fitbg(dataim, meta, datamask, meta.bg_y1,
                                meta.bg_y2, deg=meta.bg_deg,
                                threshold=meta.p3thresh, isrotate=2,
                                isplots=isplots)

    return bg, mask, n


def cut_aperture(data, meta, log):
    """Select the aperture region out of each trimmed image.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    apdata : ndarray
        The flux values over the aperture region.
    aperr : ndarray
        The noise values over the aperture region.
    apmask : ndarray
        The mask values over the aperture region.
    apbg : ndarray
        The background flux values over the aperture region.
    apv0 : ndarray
        The v0 values over the aperture region.

    Notes
    -----
    History:

    - 2022-06-17, Taylor J Bell
        Initial version based on the code in s3_reduce.py
    """
    log.writelog('  Extracting aperture region...',
                 mute=(not meta.verbose))

    ap_y1 = int(meta.src_ypos-meta.spec_hw)
    ap_y2 = int(meta.src_ypos+meta.spec_hw)
    apdata = data.flux[:, ap_y1:ap_y2].values
    aperr = data.err[:, ap_y1:ap_y2].values
    apmask = data.mask[:, ap_y1:ap_y2].values
    apbg = data.bg[:, ap_y1:ap_y2].values
    apv0 = data.v0[:, ap_y1:ap_y2].values

    return apdata, aperr, apmask, apbg, apv0


def flag_bg_phot(data, meta, log):
    '''Outlier rejection of segment along time axis for photometry pipeline.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with outlier background pixels flagged.
    '''
    log.writelog('  Performing outlier rejection...',
                 mute=(not meta.verbose))

    flux = data.flux.values
    mask = data.mask.values
    # FINDME: KBS removed estsig from inputs to speed up outlier detection.
    # Need to test performance with and without estsig on real data.
    if hasattr(meta, 'use_estsig') and meta.use_estsig:
        bgerr = np.median(data.err)
        estsig = [bgerr for j in range(len(meta.bg_thresh))]
    else:
        estsig = None

    nbadpix_total = 0
    for i in tqdm(range(flux.shape[1]), desc='Looping over Rows for outlier removal'):
        for j in range(flux.shape[2]):
            ngoodpix = np.sum(mask[:,i,j]==True)
            data['mask'][:,i,j] = sigrej.sigrej(flux[:,i,j], meta.bg_thresh, mask[:,i,j], estsig)
            if not all(data['mask'][:,i,j].values):
                #counting the amount of flagged bad pixels
                nbadpix = ngoodpix - np.sum(data['mask'][:,i,j].values)
                nbadpix_total += nbadpix
    print("{0}% of the pixels have been flagged as outliers\n".format(nbadpix_total/np.product(flux.shape)*100))

    return data