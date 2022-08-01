# NIRCam specific rountines go here
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import sigrej, background
from ..lib.util import read_time
from tqdm import tqdm
from ..lib import meanerr as me


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

    data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
    data.attrs['intend'] = data.attrs['mhdr']['INTEND']

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data

    int_times = hdulist['INT_TIMES', 1].data
    if hdulist[0].header['CHANNEL'] == 'LONG':  # Spectroscopy will have "LONG" as CHANNEL
        meta.photometry = False
        wave_2d = hdulist['WAVELENGTH', 1].data
    elif hdulist[0].header['CHANNEL'] == 'SHORT':  # Photometry will have "SHORT" as CHANNEL
        meta.photometry = True
        # The DISPAXIS argument does not exist in the header of the photometry data.
        # Added it here so that code in other sections doesn't have to be changed
        data.attrs['shdr']['DISPAXIS'] = 1
        if hdulist[0].header['FILTER'] == 'F210M': #TODO make this better for all filters
            wave_1d = np.ones_like(sci[0,0]) * 2.1
        elif hdulist[0].header['FILTER'] == 'F187N':
            wave_1d = np.ones_like(sci[0,0]) * 1.87

    int_times = hdulist['INT_TIMES', 1].data


    # Record integration mid-times in BJD_TDB
    if (hasattr(meta, 'time_file') and meta.time_file is not None):
        time = read_time(meta, data, log)
    else:
        time = int_times['int_mid_BJD_TDB']
        if len(time) > len(sci):
            # This line is needed to still handle the simulated data
            # which had the full time array for all segments
            time = time[data.attrs['intstart']:data.attrs['intend']]

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
    if not meta.photometry:
        data['wave_2d'] = (['y', 'x'], wave_2d)
        data['wave_2d'].attrs['wave_units'] = wave_units
    else:
        data['wave_1d'] = (['x'], wave_1d)
        data['wave_1d'].attrs['wave_units'] = wave_units
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
    '''Outlier rejection of segment along time axis adjusted for the photometry reduction routine.

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
    for i in tqdm(range(flux.shape[1]), desc='Looping over Rows for outlier removal'): #Loops over Rows
        for j in range(flux.shape[2]): # Loops over Columns
            ngoodpix = np.sum(mask[:,i,j]==True)
            data['mask'][:,i,j] *= sigrej.sigrej(flux[:,i,j], meta.bg_thresh, mask[:,i,j], estsig)
            if not all(data['mask'][:,i,j].values):
                #counting the amount of flagged bad pixels
                nbadpix = ngoodpix - np.sum(data['mask'][:,i,j].values)
                nbadpix_total += nbadpix
    flag_percent = nbadpix_total/np.product(flux.shape)*100
    log.writelog("  {0:.5f}% of the pixels have been flagged as outliers\n".format(flag_percent),
                 mute=(not meta.verbose))

    return data


def corr_oof(data, meta, i, star_pos_x):
    """
    Correcting for 1/f noise.
    """
    print('Correcting for 1/f noise...')

    # Let's first determine which amplifier regions are left in the frame.
    # For NIRCam: 4 amplifiers, 512 pixels in x dimension per amplifier
    pxl_idxs = np.arange(2048) # Every NIRCam subarray has 2048 pixels in the x dimension
    pxl_in_window_bool = np.zeros(2048, dtype=bool)
    # pxl_in_window_bool is True for pixels which weren't trimmed away by meta.xwindow
    for j in range(len(pxl_idxs)):
        if meta.xwindow[0] <= pxl_idxs[j] < meta.xwindow[1]:
            pxl_in_window_bool[j] = True
    ampl_used_bool = np.array(list(map(any, pxl_in_window_bool.reshape((4, 512)))))
    # Example: if only the middle two amplifier are left after trimming: ampl_used = [False, True, True, False]

    star_pos_x_untrim = int(star_pos_x) + meta.xwindow[0]  # position of star before trimming
    star_exclusion_area_untrim = np.array([star_pos_x_untrim - meta.oof_bg_dist, star_pos_x_untrim + meta.oof_bg_dist])

    use_cols = np.ones(2048, dtype=bool)
    for k in range(2048):
        if star_exclusion_area_untrim[0] <= k < star_exclusion_area_untrim[1]:
            use_cols[k] = False
    use_cols = use_cols[meta.xwindow[0]:meta.xwindow[1]]
    # Array with bools checking if column should be used for background subtraction

    # Let's go through each amplifier region
    if ampl_used_bool[0]:
        edges0 = np.array([0,512]) - meta.xwindow[0]
        edges0[np.where(edges0 < 0)] = 0
        use_cols0 = np.copy(use_cols)
        for kk in range(len(use_cols0)):
            if edges0[0] <= kk < edges0[1]:
                continue
            else:
                use_cols0[kk] = False
        flux0 = data.flux.values[i][:, use_cols0]
        err0 =  data.err.values[i] [:, use_cols0]
        mask0 = data.mask.values[i][:, use_cols0]
    if ampl_used_bool[1]:
        edges1 = np.array([512,1024]) - meta.xwindow[0]
        edges1[np.where(edges1 < 0)] = 0
        use_cols1 = np.copy(use_cols)
        for kk in range(len(use_cols1)):
            if edges1[0] <= kk < edges1[1]:
                continue
            else:
                use_cols1[kk] = False # False if columns are out of amplifier region
        flux1 = data.flux.values[i][:, use_cols1]
        err1 =  data.err.values[i] [:, use_cols1]
        mask1 = data.mask.values[i][:, use_cols1]
    if ampl_used_bool[2]:
        edges2 = np.array([1024,1536]) - meta.xwindow[0]
        edges2[np.where(edges2 < 0)] = 0
        use_cols2 = np.copy(use_cols)
        for kk in range(len(use_cols2)):
            if edges2[0] <= kk < edges2[1]:
                continue
            else:
                use_cols2[kk] = False
        flux2 = data.flux.values[i][:, use_cols2]
        err2 =  data.err.values[i] [:, use_cols2]
        mask2 = data.mask.values[i][:, use_cols2]
    if ampl_used_bool[3]:
        edges3 = np.array([1536,2048]) - meta.xwindow[0]
        edges3[np.where(edges3 < 0)] = 0
        use_cols3 = np.copy(use_cols)
        for kk in range(len(use_cols3)):
            if edges3[0] <= kk < edges3[1]:
                continue
            else:
                use_cols3[kk] = False
        flux3 = data.flux.values[i][:, use_cols3]
        err3 =  data.err.values[i] [:, use_cols3]
        mask3 = data.mask.values[i][:, use_cols3]

    if meta.oof_corr == 'meanerr':
        for j in range(128):
           if ampl_used_bool[0]:
               data.flux.values[i][j, edges0[0]:edges0[1]] -= me.meanerr(flux0[j], err0[j], mask=mask0[j], err=False)
           if ampl_used_bool[1]:
               data.flux.values[i][j, edges1[0]:edges1[1]] -= me.meanerr(flux1[j], err1[j], mask=mask1[j], err=False)
           if ampl_used_bool[2]:
               data.flux.values[i][j, edges2[0]:edges2[1]] -= me.meanerr(flux2[j], err2[j], mask=mask2[j], err=False)
           if ampl_used_bool[3]:
               data.flux.values[i][j, edges3[0]:edges3[1]] -= me.meanerr(flux3[j], err3[j], mask=mask3[j], err=False)
    elif meta.oof_corr == 'median':
        if ampl_used_bool[0]:
            data.flux.values[i][:, edges0[0]:edges0[1]] -= np.median(flux0, axis=1)[:,None]
        if ampl_used_bool[1]:
            data.flux.values[i][:, edges1[0]:edges1[1]] -= np.median(flux1, axis=1)[:,None]
        if ampl_used_bool[2]:
            data.flux.values[i][:, edges2[0]:edges2[1]] -= np.median(flux2, axis=1)[:,None]
        if ampl_used_bool[3]:
            data.flux.values[i][:, edges3[0]:edges3[1]] -= np.median(flux3, axis=1)[:,None]
    else:
        print('This oof correction method is not supported. Please choose between meanerr or median.')

    return data
