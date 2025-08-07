# NIRISS specific rountines go here
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import nircam, sigrej, optspex, plots_s3
from ..lib.util import read_time, supersample
from pastasoss import get_soss_traces
from .straighten import roll_columns
from .background import fitbg

__all__ = ['read', 'get_wave', 'straighten_trace', 'flag_ff', 'flag_bg',
           'clean_median_flux', 'fit_bg', 'cut_aperture', 'standard_spectrum',
           'residualBackground', 'lc_nodriftcorr']

'''
TODO:
    Implement niriss.calibrated_spectra()
    0th-order masking using F277W filter
    Get 2D MAD calculation working
'''


def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's NIRISS instrument.

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
        The metadata object
    log : logedit.Logedit
        The current log.

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

    # Increase pixel resolution along cross-dispersion direction
    if meta.expand > 1:
        log.writelog(f'    Super-sampling y axis from {sci.shape[1]} ' +
                     f'to {sci.shape[1]*meta.expand} pixels...',
                     mute=(not meta.verbose))
        sci = supersample(sci, meta.expand, 'flux', axis=1)
        err = supersample(err, meta.expand, 'err', axis=1)
        dq = supersample(dq, meta.expand, 'cal', axis=1)
        v0 = supersample(v0, meta.expand, 'flux', axis=1)

    # Record integration mid-times in BMJD_TDB
    if meta.time_file is not None:
        time = read_time(meta, data, log)
    else:
        time = int_times['int_mid_BJD_TDB']

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    time_units = 'BMJD_TDB'

    # Duplicate science arrays for each order to be analyzed
    if isinstance(meta.orders, int):
        meta.orders = [meta.orders]
    norders = len(meta.all_orders)
    sci = np.repeat(sci[:, :, :, np.newaxis], norders, axis=3)
    err = np.repeat(err[:, :, :, np.newaxis], norders, axis=3)
    dq = np.repeat(dq[:, :, :, np.newaxis], norders, axis=3)
    v0 = np.repeat(v0[:, :, :, np.newaxis], norders, axis=3)

    if (meta.firstFile and meta.spec_hw == meta.spec_hw_range[0] and
            meta.bg_hw == meta.bg_hw_range[0]):
        # Only apply super-sampling expansion once
        meta.ywindow[0] *= meta.expand
        meta.ywindow[1] *= meta.expand

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux', order=meta.all_orders)
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err', order=meta.all_orders)
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq', order=meta.all_orders)
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0', order=meta.all_orders)

    # Initialize bad pixel mask (False = good, True = bad)
    data['mask'] = (['time', 'y', 'x', 'order'], np.zeros(data.flux.shape,
                                                          dtype=bool))

    return data, meta, log


def get_wave(data, meta, log):
    '''Use NIRISS pupil position to determine location
    of traces and corresponding wavelength solutions.

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
        The updated Dataset object with...
    '''
    # Report pupil position
    pwcpos = data.attrs['mhdr']['PWCPOS']
    log.writelog(f"  The NIRISS pupil position is {pwcpos:3f} degrees",
                 mute=(not meta.verbose))

    norders = len(meta.all_orders)
    data['trace'] = (['x', 'order'],
                     np.zeros((data.x.shape[0], norders)) +
                     np.array(meta.src_ypos)[np.newaxis])
    data['wave_1d'] = (['x', 'order'],
                       np.zeros((data.x.shape[0], norders))*np.nan)
    data['wave_1d'].attrs['wave_units'] = 'microns'

    for order in meta.all_orders:
        # Get trace for the given order and pupil position
        trace = get_soss_traces(pwcpos=pwcpos, order=str(order), interp=True)
        if data.attrs['mhdr']['SUBARRAY'] == 'SUBSTRIP96' and \
                meta.trace_offset is None:
            # PASTASOSS doesn't account for different substrip starting rows;
            # therefore, set trace offset to best guess (-12 pixels).
            meta.trace_offset = -12
        if meta.trace_offset is not None:
            trace.y += meta.trace_offset
            subarray = data.attrs['mhdr']['SUBARRAY']
            log.writelog(f"  Shifting trace by {meta.trace_offset} pixels "
                         f"for {subarray} and Order {order}.",
                         mute=(not meta.verbose))
        # Assign trace and wavelength for given order
        ind1 = np.nonzero(np.in1d(trace.x, data.x.values))[0]
        ind2 = np.nonzero(np.in1d(data.x.values, trace.x))[0]
        data['trace'].sel(order=order)[ind2] = trace.y[ind1]
        data['wave_1d'].sel(order=order)[ind2] = trace.wavelength[ind1]

    return data


def mask_other_orders(data, meta):
    '''Mask trace regions from other orders.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with regions masked.
    '''
    for order in meta.all_orders:
        trace = np.round(data.trace.sel(order=order).values).astype(int)
        wave = data.wave_1d.sel(order=order).values
        other_orders = list.copy(meta.all_orders)
        other_orders.remove(order)
        for other_order in other_orders:
            if other_order in meta.orders:
                other_trace = np.round(
                    data.trace.sel(order=other_order).values).astype(int)
                # Loop over valid wavelengths in current order
                for j in np.where(~np.isnan(wave))[0]:
                    ymin = np.max((0,
                                   trace[j] - meta.spec_hw,
                                   other_trace[j] + meta.bg_hw + 1))
                    ymax = np.min((len(data.y) + 1,
                                   trace[j] + meta.spec_hw + 1))
                    # Mask extraction region for 'order' in 'other_order'
                    data['mask'].sel(order=other_order)[:, ymin:ymax, j] = True
    return data


def straighten_trace(data, meta, log, m):
    '''Takes a set of integrations with a curved trace and shifts the
    columns to bring the center of mass to the middle of the detector
    (and straighten the trace)

    The correction is made by whole pixels (i.e. no fractional pixel shifts)
    The shifts to be applied are computed once from the median frame and then
    applied to each integration in the timeseries

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    m : int
        The file number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    '''
    # Mask trace regions from other orders
    data = mask_other_orders(data, meta)

    for k, order in enumerate(meta.orders):
        log.writelog(f'  Correcting curvature for order {order} and ' +
                     f'bringing the trace to row {meta.src_ypos[k]}... ',
                     mute=(not meta.verbose))
        shifts = np.round(data.trace.sel(order=order).values).astype(int)
        new_center = meta.src_ypos[k]
        new_shifts = new_center - shifts

        # Keep shifts from exceeding the height of the detector
        # This only happens with SUBSTRIP96 and Order 2,
        # which is not recommended.
        ymax = data.flux.shape[1]
        new_shifts[new_shifts > ymax] = ymax
        new_shifts[new_shifts < -ymax] = -ymax

        # broadcast the shifts to the number of integrations
        new_shifts = np.reshape(np.repeat(new_shifts,
                                data.flux.shape[0]),
                                (data.flux.shape[0],
                                data.flux.shape[2]),
                                order='F')

        # Apply the shifts to the data
        data['flux'].sel(order=order)[:] = roll_columns(
            data.flux.sel(order=order).values, new_shifts)
        data['mask'].sel(order=order)[:] = roll_columns(
            data.mask.sel(order=order).values, new_shifts)
        data['err'].sel(order=order)[:] = roll_columns(
            data.err.sel(order=order).values, new_shifts)
        data['dq'].sel(order=order)[:] = roll_columns(
            data.dq.sel(order=order).values, new_shifts)
        data['v0'].sel(order=order)[:] = roll_columns(
            data.v0.sel(order=order).values, new_shifts)
        data['medflux'].sel(order=order)[:] = roll_columns(
            np.expand_dims(data.medflux.sel(order=order).values, axis=0),
            new_shifts).squeeze()

    return data, meta


def flag_ff(data, meta, log):
    '''Outlier rejection of full frame along time axis.
    For data with deep transits, there is a risk of masking good transit data.
    Proceed with caution.

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
        The updated Dataset object with outlier pixels flagged.
    '''
    return nircam.flag_ff(data, meta, log)


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

    # Look for outliers above and below the curvature-corrected trace
    for k, order in enumerate(meta.orders):
        bgdata1 = data.flux.sel(order=order)[:, :meta.bg_y1[k]]
        bgmask1 = data.mask.sel(order=order)[:, :meta.bg_y1[k]]
        bgdata2 = data.flux.sel(order=order)[:, meta.bg_y2[k]:]
        bgmask2 = data.mask.sel(order=order)[:, meta.bg_y2[k]:]

        data['mask'].sel(order=order)[:, :meta.bg_y1[k]] = sigrej.sigrej(
            bgdata1, meta.bg_thresh, bgmask1, None)
        data['mask'].sel(order=order)[:, meta.bg_y2[k]:] = sigrej.sigrej(
            bgdata2, meta.bg_thresh, bgmask2, None)

    return data


def clean_median_flux(data, meta, log, m):
    """Computes a median flux frame that is free of bad pixels.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.
    order : int; optional
        Spectral order. Default is None

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object.
    """
    log.writelog('  Computing clean median frame...', mute=(not meta.verbose))

    data['medflux'] = (['y', 'x', 'order'], np.zeros_like(data.flux[0]))
    data['medflux'].attrs['flux_units'] = data.flux.attrs['flux_units']

    # Currently, the median frame is identical for each order,
    # so looping over the orders feels unnecesary; however, when only
    # a single order is specified in the ECF, the following code will
    # work on the correct order.
    for order in meta.orders:
        # Compute median flux using masked arrays
        flux_ma = np.ma.masked_where(data.mask.sel(order=order).values,
                                     data.flux.sel(order=order).values)
        medflux = np.ma.median(flux_ma, axis=0)
        # Compute median error array
        err_ma = np.ma.masked_where(data.mask.sel(order=order).values,
                                    data.err.sel(order=order).values)
        mederr = np.ma.median(err_ma, axis=0)

        # Call subroutine
        clean_flux = optspex.get_clean(data, meta, log, medflux, mederr)

        # Assign (un)cleaned median frame to data object
        data['medflux'].sel(order=order)[:] = clean_flux

        if meta.isplots_S3 >= 3:
            plots_s3.median_frame(data, meta, m, clean_flux, order)

    return data


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Instrument wrapper for fitting the background.

    Parameters
    ----------
    dataim : ndarray (3D)
        The 3D image array (y, x, order).
    datamask : ndarray (3D)
        A boolean array of which data (set to True) should be masked.
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
        The updated boolean mask after background subtraction, where True
        values should be masked.
    n : int
        The current integration number.
    """
    norders = len(meta.orders)
    bg = np.zeros_like(dataim)
    mask = np.zeros_like(dataim, dtype=bool)
    for k in range(norders):
        bg[:, :, k], mask[:, :, k] = fitbg(
            dataim[:, :, k], meta, datamask[:, :, k],
            meta.bg_y1[k], meta.bg_y2[k], deg=meta.bg_deg,
            threshold=meta.p3thresh, isrotate=2, isplots=isplots)

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
        The mask values over the aperture region. True values should be masked.
    apbg : ndarray
        The background flux values over the aperture region.
    apv0 : ndarray
        The v0 values over the aperture region.
    apmedflux : ndarray
        The median flux over the aperture region.
    """
    log.writelog('  Extracting aperture region...',
                 mute=(not meta.verbose))

    apdata = np.zeros((len(data.time), 2*meta.spec_hw+1,
                       len(data.x), len(meta.orders)))
    aperr = np.zeros_like(apdata)
    apmask = np.zeros_like(apdata, dtype=bool)
    apbg = np.zeros_like(apdata)
    apv0 = np.zeros_like(apdata)
    apmedflux = np.zeros_like(apdata[0])
    for k in range(len(meta.orders)):
        ap_y1 = int(meta.src_ypos[k] - meta.spec_hw)
        ap_y2 = int(meta.src_ypos[k] + meta.spec_hw + 1)
        apdata[:, :, :, k] = data.flux.values[:, ap_y1:ap_y2, :, k]
        aperr[:, :, :, k] = data.err.values[:, ap_y1:ap_y2, :, k]
        apmask[:, :, :, k] = data.mask.values[:, ap_y1:ap_y2, :, k]
        apbg[:, :, :, k] = data.bg.values[:, ap_y1:ap_y2, :, k]
        apv0[:, :, :, k] = data.v0.values[:, ap_y1:ap_y2, :, k]
        apmedflux[:, :, k] = data.medflux.values[ap_y1:ap_y2, :, k]
        # Mask invalid regions
        inan = np.isnan(data.wave_1d[:, k])
        apmask[:, :, inan, k] = True

    return apdata, aperr, apmask, apbg, apv0, apmedflux


def standard_spectrum(data, meta, apdata, apmask, aperr):
    """Instrument wrapper for computing the standard box spectrum.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    apdata : ndarray
        The pixel values in the aperture region.
    apmask : ndarray
        The outlier mask in the aperture region. True where pixels should be
        masked.
    aperr : ndarray
        The noise values in the aperture region.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object in which the spectrum data will stored.
    """

    return nircam.standard_spectrum(data, meta, apdata, apmask, aperr)


def residualBackground(data, meta, m, vmin=None, vmax=None):
    """Plot the median, BG-subtracted frame to study the residual BG region and
    aperture/BG sizes. (Fig 3304)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    vmin : int; optional
        Minimum value of colormap. Default is None.
    vmax : int; optional
        Maximum value of colormap. Default is None.
    """
    for k, order in enumerate(meta.orders):
        # Specify aperture region for given order
        ap_y = [meta.src_ypos[k] - meta.spec_hw,
                meta.src_ypos[k] + meta.spec_hw + 1]
        # Specify bg region for given order
        bg_y = [meta.bg_y1[k], meta.bg_y2[k]]
        # Median flux of segment
        flux = data.medflux.sel(order=order).values
        plots_s3.residualBackground(data, meta, m, flux=flux, order=order,
                                    ap_y=ap_y, bg_y=bg_y,
                                    vmin=None, vmax=None)


def lc_nodriftcorr(spec, meta):
    '''Plot a 2D light curve without drift correction. (Fig 3101+3102)

    Fig 3101 uses a linear wavelength x-axis, while Fig 3102 uses a linear
    detector pixel x-axis.

    Parameters
    ----------
    spec : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    for k, order in enumerate(meta.orders):
        wave_1d = spec.wave_1d.sel(order=order)
        optspec = spec.optspec.sel(order=order)
        optmask = spec.optmask.sel(order=order)
        mad = meta.mad_s3[k]
        plots_s3.lc_nodriftcorr(meta, wave_1d, optspec, optmask=optmask,
                                mad=mad, order=order)
