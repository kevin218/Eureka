import numpy as np
import os
import glob
from astropy.io import fits
from . import sort_nicely as sn
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from scipy.stats import binned_statistic
from .naninterp1d import naninterp1d

from .citations import CITATIONS

# populate common imports for current stage
COMMON_IMPORTS = np.array([
    ["astropy", "eureka", "h5py", "jwst", "numpy", ],
    ["astropy", "eureka", "h5py", "jwst", "matplotlib", ],
    ["astraeus", "astropy", "crds", "eureka", "h5py", "jwst",
     "matplotlib", "numpy", "scipy", "xarray", ],
    ["astraeus", "astropy", "eureka", "h5py", "matplotlib", "numpy",
     "pandas", "scipy", "xarray", ],
    ["astraeus", "astropy", "eureka", "h5py", "matplotlib", "numpy",
     "pandas", "scipy", "xarray", ],
    ["astraeus", "astropy", "eureka", "h5py", "matplotlib", "numpy",
     "pandas", "xarray", ],
], dtype=object)


def readfiles(meta, log):
    """Read in the files saved in topdir + inputdir and save them to a list.

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


def manual_clip(lc, meta, log):
    """Manually clip integrations along time axis.

    Parameters
    ----------
    lc : Xarray Dataset
        The Dataset object containing light curve and time data.
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    lc : Xarray Dataset
        The updated Dataset object containing light curve and time data
        with the requested integrations removed.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    log : logedit.Logedit
        The updated log.
    """
    log.writelog('Manually removing data points from meta.manual_clip...',
                 mute=(not meta.verbose))

    meta.manual_clip = np.array(meta.manual_clip)
    if len(meta.manual_clip.shape) == 1:
        # The user didn't quite enter things right, so reshape
        meta.manual_clip = meta.manual_clip[np.newaxis]

    # Figure out which indices are being clipped
    time_bool = np.ones(len(lc.data.time), dtype=bool)
    for inds in meta.manual_clip:
        time_bool[inds[0]:inds[1]] = False
    time_inds = np.arange(len(lc.data.time))[time_bool]

    # Remove the requested integrations
    lc = lc.isel(time=time_inds)

    return meta, lc, log


def check_nans(data, mask, log, name=''):
    """Checks where a data-like array is invalid (contains NaNs or infs).

    Parameters
    ----------
    data : ndarray
        a data-like array (e.g. data, err, dq, ...).
    mask : ndarray
        Input mask.
    log : logedit.Logedit
        The open log in which NaNs/Infs will be mentioned, if existent.
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
    masked = np.ma.masked_invalid(data).mask
    inan = np.where(masked)
    num_nans = np.sum(masked)
    num_pixels = np.size(data)
    perc_nans = 100*num_nans/num_pixels
    if num_nans > 0 and name == 'wavelength':
        log.writelog(f"  WARNING: Your {name} array has {num_nans} NaNs, which"
                     f" are outside of the wavelength solution. You should "
                     f"consider removing indices {inan} as their data quality "
                     f"may be poor.")
    elif num_nans > 0:
        log.writelog(f"  {name} has {num_nans} NaNs/infs, which is "
                     f"{perc_nans:.2f}% of all pixels.")
        mask[inan] = 0
    if perc_nans > 10:
        log.writelog("  WARNING: Your region of interest may be off the edge "
                     "of the detector subarray.  Masking NaN/inf regions and "
                     "continuing, but you should really stop and reconsider "
                     "your choices.")
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


def binData(data, nbin=100, err=False):
    """Temporally bin data for easier visualization.

    Parameters
    ----------
    data : ndarray (1D)
        The data to temporally bin.
    nbin : int, optional
        The number of bins there should be. By default 100.
    err : bool, optional
        If True, divide the binned data by sqrt(N) to get the error on the
        mean. By default False.

    Returns
    -------
    binned : ndarray
        The binned data.
    """
    # Make a copy for good measure
    data = np.ma.copy(data)
    data = np.ma.masked_invalid(data)
    # Make sure there's a whole number of bins
    data = data[:nbin*int(len(data)/nbin)]
    # Bin data
    binned = np.ma.mean(data.reshape(nbin, -1), axis=1)
    if err:
        binned /= np.sqrt(int(len(data)/nbin))
    return binned


def binData_time(data, time, nbin=100, err=False):
    """Temporally bin data for easier visualization.

    Parameters
    ----------
    data : ndarray (1D)
        The data to temporally bin.
    time : ndarray (1D)
        The time axis along which to bin
    nbin : int, optional
        The number of bins there should be. By default 100.
    err : bool, optional
        If True, divide the binned data by sqrt(N) to get the error on the
        mean. By default False.

    Returns
    -------
    binned : ndarray
        The binned data.
    """
    # Make a copy for good measure
    data = np.ma.copy(data)
    data = np.ma.masked_invalid(data)

    binned, _, _ = binned_statistic(time, data,
                                    statistic=np.ma.mean,
                                    bins=nbin)
    if err:
        binned_count, _, _ = binned_statistic(time, data,
                                              statistic='count',
                                              bins=nbin)
        binned /= np.sqrt(binned_count)

    # Need to mask invalid data in case there is an empty bin
    # (leading to divide by zero)
    return np.ma.masked_invalid(binned)


def normalize_spectrum(meta, optspec, opterr=None, optmask=None, scandir=None):
    """Normalize a spectrum by its temporal mean.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The new meta object for the current stage processing.
    optspec : ndarray
        The spectrum to normalize.
    opterr : ndarray; optional
        The noise array to normalize using optspec, by default None.
    optmask : ndarray (1D); optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.
    scandir : ndarray; optional
        For HST spatial scanning mode, 0=forward scan and 1=reverse scan.
        Defaults to None which is fine for JWST data, but must be provided
        for HST data (can be all zero values if not spatial scanning mode).

    Returns
    -------
    normspec
        The normalized spectrum.
    normerr : ndarray; optional
        The normalized error. Only returned if opterr is not none.
    """
    normspec = np.ma.masked_invalid(np.ma.copy(optspec))
    normspec = np.ma.masked_where(optmask, normspec)

    if opterr is not None:
        normerr = np.ma.masked_invalid(np.ma.copy(opterr))
        normerr = np.ma.masked_where(np.ma.getmaskarray(normspec), normerr)

    # Normalize the spectrum
    if meta.inst == 'wfc3':
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
            wave_min=None, wave_max=None, scandir=None):
    """Computes variation on median absolute deviation (MAD) using ediff1d
    for 2D data.

    The computed MAD is the average MAD along the time axis. In
    otherwords, the MAD is computed in the time direction for each
    wavelength, and then the returned value is the average of those MAD
    values.

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
    optmask : ndarray (1D); optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.
    wave_min : float; optional
        Minimum wavelength for binned lightcurves, as given in the S4 .ecf
        file. Defaults to None which does not impose a lower limit.
    wave_maxf : float; optional
        Maximum wavelength for binned lightcurves, as given in the S4 .ecf
        file. Defaults to None which does not impose an upper limit.
    scandir : ndarray; optional
        For HST spatial scanning mode, 0=forward scan and 1=reverse scan.
        Defaults to None which is fine for JWST data, but must be provided
        for HST data (can be all zero values if not spatial scanning mode).

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
    normspec = normalize_spectrum(meta, optspec[:, iwmin:iwmax],
                                  optmask=optmask[:, iwmin:iwmax],
                                  scandir=scandir)

    if meta.inst == 'wfc3':
        # Setup 1D MAD arrays
        n_wav = normspec.shape[1]
        ediff = np.ma.zeros((2, n_wav))

        # Compute the MAD for each scan direction
        for p in range(2):
            iscans = np.where(scandir == p)[0]
            if len(iscans) > 0:
                # Compute the MAD
                for m in range(n_wav):
                    ediff[p, m] = get_mad_1d(normspec[iscans, m])

                mad = np.ma.mean(ediff[p])
                log.writelog(f"Scandir {p} MAD = {int(np.round(mad))} ppm")
                setattr(meta, f'mad_scandir{p}', mad)

        if np.all(scandir == scandir[0]):
            # Only scanned in one direction, so get rid of the other
            ediff = ediff[scandir[0]]
        else:
            # Collapse the MAD along the scan direction
            ediff = np.mean(ediff, axis=0)
    else:
        # Setup 1D MAD array
        n_wav = normspec.shape[1]
        ediff = np.ma.zeros(n_wav)

        # Compute the MAD
        for m in range(n_wav):
            ediff[m] = get_mad_1d(normspec[:, m])

    return np.ma.mean(ediff)


def get_mad_1d(data, ind_min=0, ind_max=None):
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
        data['mask'][:, rowstart:rowend, colstart:colend] = 0

    return data


# PHOTOMETRY
def interp_masked(data, meta, i, log):
    """
    Interpolates masked pixels.
    Based on the example here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The current integration.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with requested pixels masked.
    """
    if i == 0:
        log.writelog('  Interpolating masked values...',
                     mute=(not meta.verbose))
    flux = data.flux.values[i]
    mask = data.mask.values[i]
    nx = flux.shape[1]
    ny = flux.shape[0]
    grid_x, grid_y = np.mgrid[0:ny-1:complex(0, ny), 0:nx-1:complex(0, nx)]
    points = np.where(mask == 1)
    # x,y positions of not masked pixels
    points_t = np.array(points).transpose()
    values = flux[np.where(mask == 1)]  # flux values of not masked pixels

    # Use scipy.interpolate.griddata to interpolate
    if meta.interp_method == 'nearest':
        grid_z = griddata(points_t, values, (grid_x, grid_y), method='nearest')
    elif meta.interp_method == 'linear':
        grid_z = griddata(points_t, values, (grid_x, grid_y), method='linear')
    elif meta.interp_method == 'cubic':
        grid_z = griddata(points_t, values, (grid_x, grid_y), method='cubic')
    else:
        log.writelog('Your method for interpolation is not supported!'
                     'Please choose between None, nearest, linear or cubic.',
                     mute=(not meta.verbose))

    data.flux.values[i] = grid_z

    return data


def phot_arrays(data):
    """Setting up arrays for the photometry routine.

    These arrays will be populated by the returns coming from centerdriver.py
    and apphot.py

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with new arrays where the
        outputs from the photometry routine will be saved in.
    """
    keys = ['centroid_x', 'centroid_y', 'centroid_sx', 'centroid_sy',
            'aplev', 'aperr', 'nappix', 'skylev', 'skyerr', 'nskypix',
            'nskyideal', 'status', 'betaper']

    for key in keys:
        data[key] = (['time'], np.zeros_like(data.time))

    data['aplev'].attrs['flux_units'] = data.flux.attrs['flux_units']
    data['aplev'].attrs['time_units'] = data.flux.attrs['time_units']
    data['aperr'].attrs['flux_units'] = data.flux.attrs['flux_units']
    data['aperr'].attrs['time_units'] = data.flux.attrs['time_units']

    return data


def make_citations(meta, stage=None):
    """Store relevant citation information in the current meta file.

        Searches through imported libraries and current ECF parameters for
        terms that match BibTeX entries in citations.py. Every entry that
        matches gets added to a bibliography field in the meta file.

        Parameters
        ----------
        meta : eureka.lib.readECF.MetaClass
            The current metadata object.
        mods : array-like
            Array of strings containing the currently installed modules.
        stage: integer
            The integer number of the current stage (1,2,3,4,5,6)
    """

    # get common modules for the current stage
    module_cites = COMMON_IMPORTS[stage-1]

    # in S5, extract fitting methods/myfuncs to grab citations
    other_cites = []

    # check for nircam photometry in S3
    if stage == 3:
        if hasattr(meta, 'inst') and hasattr(meta, "photometry"):
            if meta.photometry and meta.inst == "nircam":
                other_cites = other_cites + ["nircam_photometry"]

    if stage == 5:
        # concat non-lsq fit methods (emcee/dynesty) to the citation list
        if "emcee" in meta.fit_method:
            other_cites = other_cites + ["emcee"]
        if "dynesty" in meta.fit_method:
            other_cites = other_cites + ["dynesty"]
        if "nuts" in meta.fit_method:
            other_cites = other_cites + ["pymc3"]
        if "exoplanet" in meta.fit_method:
            other_cites = other_cites + ["exoplanet"]

        # check if batman or GP is being used for transit/eclipse modeling
        if "batman_tr" in meta.run_myfuncs or "batman_ecl" in meta.run_myfuncs:
            other_cites.append("batman")
        if "starry" in meta.run_myfuncs:
            other_cites.append("starry")
        if "GP" in meta.run_myfuncs:
            if hasattr(meta, "GP_package"):
                other_cites.append(meta.GP_package)

    # I set the instrument in the relevant bits of S1/2, so I don't think this
    # should really be necessary. Taylor's boilerplate for later
    if not hasattr(meta, 'inst'):
        valid = False
        insts = ['miri', 'nirspec', 'nircam', 'niriss', 'wfc3']
        while not valid:
            inst = input('Which JWST/HST instrument are you using? \
                        (leave blank for none): ').lower()
            if inst != '':
                if inst in insts:
                    # The entered instrument was valid, so continue
                    meta.inst = inst
                    valid = True
            else:
                # No instrument, so just continue
                valid = True
            if not valid:
                # The entered instrument was not valid, so explain, ask again
                print(f'The instrument {inst} is not a valid instrument. \
                        Please choose from {insts}.')

    # make sure instrument is in citation list
    other_cites.append(meta.inst)

    # get all new citations together
    current_cites = np.union1d(module_cites, other_cites)

    # check if meta has existing list of citations/bibitems, if it does, make
    # sure we include imports from previous stages in our citations
    prev_cites = []
    if hasattr(meta, 'citations'):
        prev_cites = meta.citations

    all_cites = np.union1d(current_cites, prev_cites).tolist()

    # make sure everything in meta citation list can be added to bibliography
    for entry in all_cites:
        if entry not in CITATIONS.keys():
            all_cites.remove(entry)

    # store everything in the meta object
    meta.citations = all_cites
    meta.bibliography = [CITATIONS[entry] for entry in meta.citations]


def supersample(data, expand, type, axis=1):
    """Apply subpixel interpolation to the given arrays in the cross-disperion
    direction.

    Parameters
    ----------
    data : ND array
        Array of values to be super-sampled.
    expand : int
        Super-sampling factor along the given axis.
    type : str
        Options are: data, err, dq, or wave.
    axis : int, Optional
        Axis along which interpolation is performed (default is 1).

    Returns
    -------
    zdata : ND array
        The updated array at higher resolution
    """
    # Build array of expansion factors
    # e.g., [1, 5, 1] for axis=1 and expand=5
    ndim = np.ndim(data)
    expand_seq = np.ones(ndim, dtype=int)
    expand_seq[axis] = expand

    # SciPy's zoom can't handle NaNs, so let's replace them via interpolation
    if ndim == 3:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = naninterp1d(data[i, j])
    if ndim == 2:
        for i in range(data.shape[0]):
            data[i] = naninterp1d(data[i])

    # Apply linear interpolation along axis
    if type == 'flux':
        # Divide by 'expand' to conserve flux/variance
        zdata = zoom(data, expand_seq, order=1, mode='nearest')/expand
    elif type == 'err':
        # Divide by 'sqrt(expand)'' to conserve uncertainty
        zdata = zoom(data, expand_seq, order=1, mode='nearest')/np.sqrt(expand)
    elif type == 'cal':
        # Apply same dq flag, gain values, etc to all super-sampled pixels
        zdata = np.repeat(data, expand, axis=axis)
    elif type == 'wave':
        zdata = zoom(data, expand_seq, order=1, mode='nearest')
    else:
        print(f"Type {type} not supported.  Must be one of flux, err, cal, " +
              "or wave. No super-sampling applied.")
        zdata = data
    return zdata
