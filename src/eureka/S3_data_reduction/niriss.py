import numpy as np
import astraeus.xarrayIO as xrio
from astropy.io import fits

from .background import fitbg3
from .niriss_extraction import dirty_mask
import .tracing_niriss as tn


__all__ = ['read', 'define_traces', 'fit_bg']


def read(filename, data, meta):
    """Reads a single FITS file from JWST's NIRISS instrument.

    This takes in the Stage 2 processed files.

    Parameters
    ----------
    filename : str
       Single filename to read. Should be a `.fits` file.
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
       Metadata stored in the FITS file.
    """
    with fits.open(filename) as hdulist:
        # Load master and science headers
        data.attrs['filename'] = filename
        data.attrs['mhdr'] = hdulist[0].header
        data.attrs['shdr'] = hdulist['SCI', 1].header
        data.attrs['NINTS'] = data.attrs['mhdr']['NINTS']

        # need some placeholder right now for testing
        data.attrs['intstart'] = 0
        data.attrs['intend'] = 3

        if hasattr(meta, 'f277_filename') and meta.f277_filename is not None:
            with fits.open(meta.f277_filename) as f277:
                data.attrs['f277'] = f277[1].data

        # Load data
        sci = hdulist['SCI', 1].data
        err = hdulist['ERR', 1].data
        dq = hdulist['DQ', 1].data
        v0 = hdulist['VAR_RNOISE', 1].data
        # var  = hdulist['VAR_POISSON',1].data
        wave_2d = hdulist['WAVELENGTH', 1].data
        int_times = hdulist['INT_TIMES', 1].data

        # Record integration mid-times in BJD_TDB
        if len(int_times['int_mid_BJD_TDB']) != 0:
            time = int_times['int_mid_BJD_TDB']
            time_units = 'BJD_TDB'
        else:
            # This exception is (hopefully) only for simulated data
            print("  WARNING: INT_TIMES not found. "
                  "Using EXPSTART and EXPEND in UTC.")
            time = np.linspace(data.attrs['mhdr']['EXPSTART'],
                               data.attrs['mhdr']['EXPEND'],
                               int(data.attrs['NINTS']))

            time_units = 'UTC'
            # Check that number of SCI integrations matches NINTS from header
            if data.attrs['NINTS'] != sci.shape[0]:
                print("  WARNING: Number of SCI integrations doesn't match "
                      " NINTS from header. Updating NINTS.")
                data.attrs['NINTS'] = sci.shape[0]
                time = time[:data.attrs['NINTS']]

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    wave_units = 'microns'

    # removes NaNs from the data & error arrays
    sci[np.isnan(sci)] = 0
    err[np.isnan(sci)] = 0
    # median = np.nanmedian(sci, axis=0)

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

    return data, meta


def fit_bg(data, meta, readnoise=11, sigclip=[4, 4, 4],
           box=(5, 2), filter_size=(2, 2),
           bkg_estimator=['median', ],
           testing=False, isplots=0):
    """Subtracts background from non-spectral regions.

    Uses photutils.background.Background2D to estimate background noise.
    More documentation can be found at:
    https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    readnoise : float, optional
       An estimation of the readnoise of the detector.
       Default is 11.
    sigclip : list, array; optional
       A list or array of len(n_iiters) corresponding to the
       sigma-level which should be clipped in the cosmic
       ray removal routine. Default is [4,4,4].
    box : list, array; optional
       The box size along each axis. Box has two elements: (ny, nx). For best
       results, the box shape should be chosen such that the data are covered
       by an integer number of boxes in both dimensions. Default is (5, 2).
    filter_size : list, array; optional
       The window size of the 2D median filter to apply to the low-resolution
       background map. Filter_size has two elements: (ny, nx). A filter size of
       1 (or (1,1)) means no filtering. Default is (2, 2).
    bkg_estimator : list, array; optional
       The value which to approximate the background values as. Options are
       "mean", "median", or "MMMBackground". Default is ['median', ].
    testing : bool, optional
       Evaluates the background across fewer integrations to test and
       save computational time. Default is False.
    isplots : int, optional
       The level of output plots to display. Default is 0
       (no plots).

    Returns
    -------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    """
    if meta.trace_method == 'ears':
        box_mask = dirty_mask(data.medflux.values,
                              meta.trace_ear, booltype=True,
                              return_together=True)
    elif meta.trace_method == 'edges':
        box_mask = dirty_mask(data.medflux.values,
                              meta.trace_edge, booltype=True,
                              return_together=True)

    bkg, bkg_var, cr_mask = fitbg3(data, np.array(box_mask-1, dtype=bool),
                                   readnoise, sigclip,
                                   bkg_estimator=bkg_estimator,
                                   box=box, filter_size=filter_size,
                                   testing=testing, isplots=isplots)

    data['bg'] = xrio.makeFluxLikeDA(bkg, meta.time,
                                     data['flux'].attrs['flux_units'],
                                     data['flux'].attrs['time_units'],
                                     name='bg')
    data['bg_var'] = xrio.makeFluxLikeDA(bkg_var, meta.time,
                                         data['flux'].attrs['flux_units'],
                                         data['flux'].attrs['time_units'],
                                         name='bg_var')
    data['bg_removed'] = xrio.makeFluxLikeDA(data.flux - data.bg, meta.time,
                                             data['flux'].attrs['flux_units'],
                                             data['flux'].attrs['time_units'],
                                             name='bg_removed')

    return data


def define_traces(meta, log):
    """A cute little routine that defines the NIRISS traces.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    """
    # Get summed frame for tracing
    with fits.open(meta.segment_list[-1]) as hdulist:
        # Figure out which instrument we are using
        meta.median = np.nansum(hdulist[1].data, axis=0)

    # identifies the trace for all orders
    if meta.trace_method == 'ears':
        traces = tn.mask_method_ears(meta,
                                     degree=meta.poly_order,
                                     save=meta.save_table,
                                     outdir=meta.outputdir,
                                     isplots=meta.isplots_S3)
        meta.trace_ear = traces
    elif meta.trace_method == 'edges':
        traces = tn.mask_method_edges(meta,
                                      radius=meta.radius,
                                      gf=meta.filter,
                                      save=meta.save_table,
                                      outdir=meta.outputdir,
                                      isplots=meta.isplots_S3)
        meta.trace_edge = traces
    else:
        # This will break if traces cannot be extracted
        log.writelog('Method for identifying NIRISS trace'
                     'not implemented. Please select between "ears"'
                     'and "edges".\n')
        raise AssertionError('Method for identifying NIRISS trace'
                             'not implemented.')
    return meta
