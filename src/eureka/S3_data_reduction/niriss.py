import numpy as np
import astraeus.xarrayIO as xrio
from astropy.io import fits

from .niriss_extraction import dirty_mask
from .tracing_niriss import mask_method_edges, ref_file

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


def fit_bg(data, meta, which_bkg='simple', testing=False, isplots=0):
    """Subtracts background from non-spectral regions.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    """
    box_mask = dirty_mask(data.medflux.values,
                          meta.trace, booltype=True,
                          return_together=True)

    ## NEED TO DEINE NEW BACKGROUND ROUTINE HERE

    data['bg'] = xrio.makeFluxLikeDA(bkg, data.time,
                                     data['flux'].attrs['flux_units'],
                                     data['flux'].attrs['time_units'],
                                     name='bg')

    data['bg_removed'] = xrio.makeFluxLikeDA(data.flux - data.bg, data.time,
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
    if meta.trace_method == 'edges':
        traces = mask_method_edges(meta,
                                   radius=meta.radius,
                                   gf=meta.filter,
                                   save=meta.save_table,
                                   outdir=meta.outputdir,
                                   isplots=meta.isplots_S3)
        meta.trace = traces
    elif meta.trace_method == 'ref':
        traces = ref_file(trace_filename)

    else:
        # This will break if traces cannot be extracted
        log.writelog('Method for identifying NIRISS trace'
                     'not implemented. Please select between "edges"'
                     'and "ref" (for reference file).\n')
        raise AssertionError('Method for identifying NIRISS trace'
                             'not implemented.')
    return meta
