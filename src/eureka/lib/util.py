import numpy as np
import os
import glob
from astropy.io import fits
from . import sort_nicely as sn


def readfiles(meta, log):
    """Reads in the files saved in topdir + inputdir and saves them into a list.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object with added segment_list containing the sorted
        data fits files.
    """
    meta.segment_list = []

    # Look for files in the input directory
    for fname in glob.glob(meta.inputdir+'*'+meta.suffix+'.fits'):
        meta.segment_list.append(fname)

    # Need to allow for separated sci and cal directories for WFC3
    if len(meta.segment_list) == 0:
        # Add files from the sci directory if present
        if not hasattr(meta, 'sci_dir') or meta.sci_dir is None:
            meta.sci_dir = 'sci'
        sci_path = os.path.join(meta.inputdir, meta.sci_dir)+os.sep
        for fname in glob.glob(sci_path+'*'+meta.suffix+'.fits'):
            meta.segment_list.append(fname)
        # Add files from the cal directory if present
        if not hasattr(meta, 'cal_dir') or meta.cal_dir is None:
            meta.cal_dir = 'cal'
        cal_path = os.path.join(meta.inputdir, meta.cal_dir)+os.sep
        for fname in glob.glob(cal_path+'*'+meta.suffix+'.fits'):
            meta.segment_list.append(fname)

    meta.segment_list = np.array(sn.sort_nicely(meta.segment_list))

    meta.num_data_files = len(meta.segment_list)
    if meta.num_data_files == 0:
        raise AssertionError(f'Unable to find any "{meta.suffix}.fits" files '
                             f'in the inputdir: \n"{meta.inputdir}"!\n'
                             f'You likely need to change the inputdir in '
                             f'{meta.filename} to point to the folder '
                             f'containing the "{meta.suffix}.fits" files.')
    else:
        mute = hasattr(meta, 'verbose') and not meta.verbose
        log.writelog(f'\nFound {meta.num_data_files} data file(s) '
                     f'ending in {meta.suffix}.fits',
                     mute=mute)

        with fits.open(meta.segment_list[-1]) as hdulist:
            # Figure out which instrument we are using
            meta.inst = hdulist[0].header['INSTRUME'].lower()

    return meta


def trim(data, meta):
    """Removes the edges of the data arrays.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    subdata : Xarray Dataset
        A new Dataset object with arrays that have been trimmed, depending on
        xwindow and ywindow as set in the S3 ecf.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    """
    subdata = data.isel(y=np.arange(meta.ywindow[0], meta.ywindow[1]),
                        x=np.arange(meta.xwindow[0], meta.xwindow[1]))
    meta.subny = meta.ywindow[1] - meta.ywindow[0]
    meta.subnx = meta.xwindow[1] - meta.xwindow[0]
    if meta.inst == 'wfc3':
        subdata['guess'] = subdata.guess - meta.ywindow[0]

    return subdata, meta


def check_nans(data, mask, log, name=''):
    """Checks where a data array has NaNs or infs.

    Parameters
    ----------
    data : ndarray
        a data array (e.g. data, err, dq, ...).
    mask : ndarray
        Input mask.
    log : logedit.Logedit
        The open log in which NaNs will be mentioned if existent.
    name : str; optional
        The name of the data array passed in (e.g. SUBDATA, SUBERR, SUBV0).
        Defaults to ''.

    Returns
    -------
    mask : ndarray
        Output mask where 0 will be written where the input data array has NaNs
        or infs.
    """
    data = np.ma.masked_where(mask == 0, np.copy(data))
    num_nans = np.sum(np.ma.masked_invalid(data).mask)
    if num_nans > 0:
        log.writelog(f"  WARNING: {name} has {num_nans} NaNs/infs. Your "
                     "subregion may be off the edge of the detector "
                     "subarray.\n    Masking NaN region and continuing, "
                     "but you should really stop and reconsider your"
                     "choices.")
        inan = np.where(np.ma.masked_invalid(data).mask)
        # subdata[inan]  = 0
        mask[inan] = 0
    return mask


def makedirectory(meta, stage, counter=None, **kwargs):
    """Creates a directory for the current stage.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    stage : str
        'S#' string denoting stage number (i.e. 'S3', 'S4').
    counter : int; optional
        The run number if you want to force a particular run number.
        Defaults to None which automatically finds the run number.
    **kwargs : dict
        Additional key,value pairs to add to the folder name
        (e.g. {'ap': 4, 'bg': 10}).

    Returns
    -------
    run : int
        The run number
    """
    # This code allows the input and output files to be stored outside
    # of the Eureka! folder
    rootdir = os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))
    if rootdir[-1] != os.sep:
        rootdir += os.sep

    outputdir = rootdir+stage+'_'+meta.datetime+'_'+meta.eventlabel+'_run'

    if counter is None:
        counter = 1
        while os.path.exists(outputdir+str(counter)):
            counter += 1
        outputdir += str(counter)+os.sep
    else:
        outputdir += str(counter)+os.sep

    # Nest the different folders underneath one main folder for this run
    for key, value in kwargs.items():
        outputdir += key+str(value)+'_'

    # Remove trailing _ if present
    if outputdir[-1] == '_':
        outputdir = outputdir[:-1]

    # Add trailing slash
    if outputdir[-1] != os.sep:
        outputdir += os.sep

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except (PermissionError, OSError) as e:
            # Raise a more helpful error message so that users know to update
            # topdir in their ecf file
            message = (f'You do not have the permissions to make the folder '
                       f'{outputdir}\nYour topdir is currently set to'
                       f'{meta.topdir}, but your user account is called '
                       f'{os.getenv("USER")}.\nYou likely need to update the '
                       f'topdir setting in your {stage} .ecf file.')
            raise PermissionError(message) from e
    if not os.path.exists(os.path.join(outputdir, "figs")):
        os.makedirs(os.path.join(outputdir, "figs"))

    return counter


def pathdirectory(meta, stage, run, old_datetime=None, **kwargs):
    """Finds the directory for the requested stage, run, and datetime
    (or old_datetime).

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    stage : str
        'S#' string denoting stage number (i.e. 'S3', 'S4')
    run : int
        run #, output from makedirectory function
    old_datetime : str; optional
        The date that a previous run was made (for looking up old data).
        Defaults to None in which case meta.datetime is used instead.
    **kwargs : dict
        Additional key,value pairs to add to the folder name
        (e.g. {'ap': 4, 'bg': 10}).

    Returns
    -------
    path : str
        Directory path for given parameters
    """
    if old_datetime is not None:
        datetime = old_datetime
    else:
        datetime = meta.datetime

    # This code allows the input and output files to be stored outside
    # of the Eureka! folder
    rootdir = os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))
    if rootdir[-1] != os.sep:
        rootdir += os.sep

    outputdir = (rootdir+stage+'_'+datetime+'_'+meta.eventlabel+'_run' +
                 str(run)+os.sep)

    for key, value in kwargs.items():
        outputdir += key+str(value)+'_'

    # Remove trailing _ if present
    if outputdir[-1] == '_':
        outputdir = outputdir[:-1]

    # Add trailing slash
    if outputdir[-1] != os.sep:
        outputdir += os.sep

    return outputdir


def find_fits(meta):
    '''Locates S1 or S2 output FITS files if unable to find an metadata file.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The new meta object for the current stage processing.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The meta object with the updated inputdir pointing to the location of
        the input files to use.

    Notes
    -----
    History:

    - April 25, 2022 Taylor Bell
        Initial version.
    '''
    fnames = glob.glob(meta.inputdir+'*'+meta.suffix + '.fits')
    if len(fnames) == 0:
        # There were no rateints files in that folder, so let's see if
        # there are in children folders
        fnames = glob.glob(meta.inputdir+'**'+os.sep+'*'+meta.suffix+'.fits',
                           recursive=True)
        fnames = sn.sort_nicely(fnames)

    if len(fnames) == 0:
        # If the code can't find any of the reqested files, raise an error
        # and give a helpful message
        message = (f'Unable to find any "{meta.suffix}.fits" files in the '
                   f'inputdir: \n"{meta.inputdir}"!\nYou likely need to change'
                   f' the inputdir in {meta.filename} to point to the folder '
                   f'containing the "{meta.suffix}.fits" files.')
        raise AssertionError(message)

    folders = np.unique([os.sep.join(fname.split(os.sep)[:-1])
                         for fname in fnames])
    if len(folders) >= 1:
        # get the file with the latest modified time
        folder = max(folders, key=os.path.getmtime)

    if len(folders) > 1:
        # There may be multiple runs - use the most recent but warn the user
        print(f'WARNING: There are multiple folders containing '
              f'"{meta.suffix}.fits" files in your inputdir:\n'
              f'"{meta.inputdir}"\n'
              f'Using the files in: \n{folder}\n'
              f'and will consider aperture ranges listed there. If this '
              f'metadata file is not a part\nof the run you intended, please '
              f'provide a more precise folder for the metadata file.')

    meta.inputdir = folder
    meta.inputdir_raw = folder[len(meta.topdir):]

    # Make sure there's a trailing slash at the end of the paths
    if meta.inputdir[-1] != os.sep:
        meta.inputdir += os.sep

    return meta


def normalize_spectrum(meta, optspec, opterr=None, optmask=None):
    """Normalize a spectrum by its temporal mean.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The new meta object for the current stage processing.
    optspec : ndarray
        The spectrum to normalize.
    opterr : ndarray, optional
        The noise array to normalize using optspec, by default None.
    optmask : ndarray (1D), optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.

    Returns
    -------
    normspec
        The normalized spectrum.
    normerr : ndarray, optional
        The normalized error. Only returned if opterr is not none.
    """
    normspec = np.ma.masked_invalid(np.ma.copy(optspec))
    normspec = np.ma.masked_where(optmask, normspec)

    if opterr is not None:
        normerr = np.ma.masked_invalid(np.ma.copy(opterr))
        normerr = np.ma.masked_where(np.ma.getmaskarray(normspec), normerr)

    # Normalize the spectrum
    if meta.inst == 'wfc3':
        scandir = np.repeat(meta.scandir, meta.nreads)
        
        for p in range(2):
            iscans = np.where(scandir == p)[0]
            if len(iscans) > 0:
                for r in range(meta.nreads):
                    if opterr is not None:
                        normerr[iscans[r::meta.nreads]] /= np.ma.mean(
                            normspec[iscans[r::meta.nreads]], axis=0)
                    normspec[iscans[r::meta.nreads]] /= np.ma.mean(
                        normspec[iscans[r::meta.nreads]], axis=0)
    else:
        if opterr is not None:
            normerr = normerr/np.ma.mean(normspec, axis=0)
        normspec = normspec/np.ma.mean(normspec, axis=0)

    if opterr is not None:
        return normspec, normerr
    else:
        return normspec


def get_mad(meta, log, wave_1d, optspec, optmask=None,
            wave_min=None, wave_max=None):
    """Computes variation on median absolute deviation (MAD) using ediff1d
    for 2D data.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        Unused. The metadata object.
    log : logedit.Logedit
        The current log.
    wave_1d : ndarray
        Wavelength array (nx) with trimmed edges depending on xwindow and
        ywindow which have been set in the S3 ecf
    optspec : ndarray
        Optimally extracted spectra, 2D array (time, nx)
    optmask : ndarray (1D), optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.
    wave_min : float; optional
        Minimum wavelength for binned lightcurves, as given in the S4 .ecf
        file. Defaults to None which does not impose a lower limit.
    wave_maxf : float; optional
        Maximum wavelength for binned lightcurves, as given in the S4 .ecf
        file. Defaults to None which does not impose an upper limit.

    Returns
    -------
    mad : float
        Single MAD value in ppm
    """
    optspec = np.ma.masked_invalid(optspec)
    optspec = np.ma.masked_where(optmask, optspec)

    if wave_min is not None:
        iwmin = np.argmin(np.abs(wave_1d-wave_min))
    else:
        iwmin = 0
    if wave_max is not None:
        iwmax = np.argmin(np.abs(wave_1d-wave_max))
    else:
        iwmax = None

    # Normalize the spectrum
    normspec = normalize_spectrum(meta, optspec[:, iwmin:iwmax])

    # Compute the MAD
    n_int = normspec.shape[0]
    ediff = np.ma.zeros(n_int)
    for m in range(n_int):
        ediff[m] = get_mad_1d(normspec[m])

    if meta.inst == 'wfc3':
        scandir = np.repeat(meta.scandir, meta.nreads)

        # Compute the MAD for each scan direction
        for p in range(2):
            iscans = np.where(scandir == p)[0]
            if len(iscans) > 0:
                mad = np.ma.mean(ediff[iscans])
                log.writelog(f"Scandir {p} MAD = {int(np.round(mad))} ppm")
                setattr(meta, f'mad_scandir{p}', mad)   

    return np.ma.mean(ediff)


def get_mad_1d(data, ind_min=0, ind_max=-1):
    """Computes variation on median absolute deviation (MAD) using ediff1d
    for 1D data.

    Parameters
    ----------
    data : ndarray
        The array from which to calculate MAD.
    int_min : int
        Minimum index to consider.
    ind_max : int
        Maximum index to consider (excluding ind_max).

    Returns
    -------
    mad : float
        Single MAD value in ppm
    """
    return 1e6 * np.ma.median(np.ma.abs(np.ma.ediff1d(data[ind_min:ind_max])))


def read_time(meta, data, log):
    """Read in a time CSV file instead of using the FITS time array.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    data : Xarray Dataset
        The Dataset object with the fits data stored inside.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    time : ndarray
        The time array stored in the meta.time_file CSV file.
    """
    fname = os.path.join(meta.topdir,
                         os.sep.join(meta.time_file.split(os.sep)))
    if meta.firstFile:
        log.writelog('  Note: Using the time stamps from:\n    '+fname)
    time = np.loadtxt(fname).flatten()[data.attrs['intstart']:
                                       data.attrs['intend']-1]

    return time


def manmask(data, meta, log):
    '''Manually mask input bad pixels.

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
        The updated Dataset object with requested pixels masked.
    '''
    log.writelog("  Masking manually identified bad pixels...",
                 mute=(not meta.verbose))
    for i in range(len(meta.manmask)):
        colstart, colend, rowstart, rowend = meta.manmask[i]
        data['mask'][rowstart:rowend, colstart:colend] = 0

    return data
