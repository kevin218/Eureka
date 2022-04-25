import numpy as np
from . import sort_nicely as sn
import os, time
import re

def readfiles(meta):
    """Reads in the files saved in topdir + inputdir and saves them into a list

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.

    Returns
    -------
    meta:   MetaClass
        The metadata object with added segment_list containing the sorted data fits files.
    """
    meta.segment_list = []
    for fname in os.listdir(meta.inputdir):
        if fname.endswith(meta.suffix + '.fits'):
            meta.segment_list.append(meta.inputdir + fname)
    meta.segment_list = np.array(sn.sort_nicely(meta.segment_list))
    return meta

def trim(data, meta):
    """Removes the edges of the data arrays

    Parameters
    ----------
    data:   DataClass
        The data object.
    meta:   MetaClass
        The metadata object.

    Returns
    -------
    data:   DataClass
        The data object with added subdata arrays with trimmed edges depending on xwindow and ywindow which have been set in the S3 ecf.
    meta:   MetaClass
        The metadata object.
    """
    data.subdata = data.data[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.suberr  = data.err[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subdq   = data.dq[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subwave = data.wave[meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subv0   = data.v0[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    meta.subny = meta.ywindow[1] - meta.ywindow[0]
    meta.subnx = meta.xwindow[1] - meta.xwindow[0]
    if hasattr(meta, 'diffmask'):
        # Need to crop diffmask and variance from WFC3 as well
        meta.subdiffmask.append(meta.diffmask[-1][:,meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]])
        data.subvariance = np.copy(data.variance[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]])
        delattr(data, 'variance')

    return data, meta

def check_nans(data, mask, log, name=''):
    """Checks where a data array has NaNs

    Parameters
    ----------
    data:   ndarray
        a data array (e.g. data, err, dq, ...)
    mask:   ndarray
        input mask
    log:    logedit.Logedit
        The open log in which NaNs will be mentioned if existent.
    name:   str, optional
        The name of the data array passed in (e.g. SUBDATA, SUBERR, SUBV0)

    Returns
    -------
    mask:   ndarray
        output mask where 0 will be written where the input data array has NaNs
    """
    num_nans = np.sum(np.isnan(data))
    if num_nans > 0:
        log.writelog(f"  WARNING: {name} has {num_nans} NaNs.  Your subregion may be off the edge of the detector subarray.\n"+
                     "Masking NaN region and continuing, but you should really stop and reconsider your choices.")
        inan = np.where(np.isnan(data))
        #subdata[inan]  = 0
        mask[inan]  = 0
    return mask

def makedirectory(meta, stage, **kwargs):
    """Creates a directory for the current stage

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    stage:  str
        'S#' string denoting stage number (i.e. 'S3', 'S4')
    **kwargs

    Returns
    -------
    run:    int
        The run number
    """
    if not hasattr(meta, 'datetime') or meta.datetime is None:
        meta.datetime = time.strftime('%Y-%m-%d')
    datetime = meta.datetime

    # This code allows the input and output files to be stored outside of the Eureka! folder
    rootdir = os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))
    if rootdir[-1]!='/':
      rootdir += '/'

    outputdir = rootdir + stage + '_' + datetime + '_' + meta.eventlabel +'_'

    for key, value in kwargs.items():

        outputdir += key+str(value)+'_'

    outputdir += 'run'

    counter=1

    while os.path.exists(outputdir+str(counter)):
        counter += 1

    meta.outputdir = outputdir+str(counter)+'/'
    if not os.path.exists(meta.outputdir):
        try:
            os.makedirs(meta.outputdir)
        except (PermissionError, OSError) as e:
            # Raise a more helpful error message so that users know to update topdir in their ecf file
            raise PermissionError(f'You do not have the permissions to make the folder {meta.outputdir}\n'+
                                  f'Your topdir is currently set to {meta.topdir}, but your user account is called {os.getenv("USER")}.\n'+
                                  f'You likely need to update the topdir setting in your {stage} .ecf file.') from e
    if not os.path.exists(meta.outputdir + "figs"):
        os.makedirs(meta.outputdir + "figs")

    return counter

def pathdirectory(meta, stage, run, old_datetime=None, **kwargs):
    """Finds the directory for the requested stage, run, and datetime (or old_datetime)

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    stage:  str
        'S#' string denoting stage number (i.e. 'S3', 'S4')
    run:    int
        run #, output from makedirectory function
    old_datetime:   str
        The date that a previous run was made (for looking up old data)
    **kwargs

    Returns
    -------
    path:   str
        Directory path for given parameters
    """
    if old_datetime is not None:
        datetime = old_datetime
    else:
        if not hasattr(meta, 'datetime') or meta.datetime is None:
            meta.datetime = time.strftime('%Y-%m-%d')
        datetime = meta.datetime

    # This code allows the input and output files to be stored outside of the Eureka! folder
    rootdir = os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))
    if rootdir[-1]!='/':
      rootdir += '/'

    outputdir = rootdir + stage + '_' + datetime + '_' + meta.eventlabel +'_'

    for key, value in kwargs.items():

        outputdir += key+str(value)+'_'

    outputdir += 'run'

    path = outputdir+str(run)+'/'

    return path

def get_mad(meta, wave_1d, optspec, wave_min=None, wave_max=None):
    """Computes variation on median absolute deviation (MAD) using ediff1d for 2D data.

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    wave_1d:    ndarray
        Wavelength array (nx) with trimmed edges depending on xwindow and ywindow which have been set in the S3 ecf
    optspec:    ndarray
        Optimally extracted spectra, 2D array (time, nx)
    wave_min:   float
        Minimum wavelength for binned lightcurves, as given in the S4 .ecf file
    wave_max:   float
        Maximum wavelength for binned lightcurves, as given in the S4 .ecf file

    Returns:
        Single MAD value in ppm
    """
    optspec = np.ma.masked_invalid(optspec)
    n_int, nx = optspec.shape
    if wave_min is not None:
        iwmin = np.argmin(np.abs(wave_1d-wave_min))
    else:
        iwmin = 0
    if wave_max is not None:
        iwmax = np.argmin(np.abs(wave_1d-wave_max))
    else:
        iwmax = None
    normspec = optspec / np.ma.mean(optspec, axis=0)
    ediff = np.ma.zeros(n_int)
    for m in range(n_int):
        ediff[m] = get_mad_1d(normspec[m],iwmin,iwmax)
    mad = np.ma.mean(ediff)
    return mad

def get_mad_1d(data, ind_min=0, ind_max=-1):
    """Computes variation on median absolute deviation (MAD) using ediff1d for 1D data.

    Parameters
    ----------
    data : ndarray
        The array from which to calculate MAD.
    int_min : int
        Minimum index to consider.
    ind_max : int
        Maximum index to consider (excluding ind_max).

    Returns:
        Single MAD value in ppm
    """
    return 1e6 * np.ma.median(np.ma.abs(np.ma.ediff1d(data[ind_min:ind_max])))
