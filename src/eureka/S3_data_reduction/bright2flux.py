import numpy as np
import scipy.interpolate as spi
from scipy.constants import arcsec
from astropy.io import fits
from ..lib.util import supersample
import crds


def rate2count(data):
    """This function converts the data, uncertainty, and variance arrays from
    rate units (#/s) to counts (#).

    Parameters
    ----------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        rate units (#/s).

    Returns
    -------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        count units (#).

    Notes
    -----
    History:

    - Mar 7, 2022 Taylor J Bell
        Initial version
    - Apr 20, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    """
    if "EFFINTTM" in data.attrs['mhdr'].keys():
        int_time = data.attrs['mhdr']['EFFINTTM']
    elif "EXPTIME" in data.attrs['mhdr'].keys():
        int_time = data.attrs['mhdr']['EXPTIME']
    else:
        raise ValueError('No FITS header keys found to permit conversion from '
                         'rate units (#/s) to counts (#)')

    data['flux'] *= int_time
    data['err'] *= int_time
    data['v0'] *= int_time

    return data


def dn2electrons(data, meta, log):
    """This function converts the data, uncertainty, and variance arrays from
    raw units (DN) to electrons.

    Parameters
    ----------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of DN.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of electrons.

    Notes
    -----
    The gain files can be downloaded from CRDS
    (https://jwst-crds.stsci.edu/browse_db/)

    History:

    - Jun 2021 Kevin Stevenson
        Initial version
    - Jul 2021
        Added gainfile rotation
    - Apr 20, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    """
    # Subarray parameters
    xstart = data.attrs['mhdr']['SUBSTRT1']
    ystart = data.attrs['mhdr']['SUBSTRT2']
    nx = data.attrs['mhdr']['SUBSIZE1']
    ny = data.attrs['mhdr']['SUBSIZE2']

    if hasattr(meta, 'gain') and meta.gain is not None:
        # Load gain array or value
        gain = np.array(meta.gain)
    else:
        # Find the required gainfile
        if hasattr(meta, 'gainfile') and meta.gainfile is not None:
            log.writelog(f'  Using provided gainfile={meta.gainfile} to '
                         'convert units to DN...',
                         mute=(not meta.verbose))
        else:
            # Retrieve the required reference files if not manually specified
            log.writelog('  Automatically getting reference files to reverse'
                         ' the PHOTOM step...', mute=(not meta.verbose))
            if data.attrs['mhdr']['TELESCOP'] != 'JWST':
                message = ('Error: Currently unable to automatically download '
                           'reference files for non-JWST observations!')
                log.writelog(message, mute=True)
                raise ValueError(message)
            meta.gainfile = retrieve_ancil(data.attrs['filename'], 'gain')

        # Load gain array in units of e-/ADU
        gain_header = fits.getheader(meta.gainfile)
        xstart_gain = gain_header['SUBSTRT1']
        ystart_gain = gain_header['SUBSTRT2']

        ystart_trim = ystart-ystart_gain + 1  # 1 indexed, NOT zero
        xstart_trim = xstart-xstart_gain + 1

        gain = fits.getdata(meta.gainfile)[ystart_trim:ystart_trim+ny,
                                           xstart_trim:xstart_trim+nx]

    if data.attrs['shdr']['DISPAXIS'] == 2 and gain.size > 1:
        # In the case of MIRI data, the gain file data has to be
        # rotated by 90 degrees and mirrored along that new x-axis
        # so that wavelength increases to the right
        gain = np.swapaxes(gain, 0, 1)[:, ::-1]

    if gain.size > 1:
        # Super-sample gain file
        # Apply same gain values to all super-sampled pixels
        gain = supersample(gain, meta.expand, 'cal', axis=0)
        # Get the gain subarray
        gain = gain[meta.ywindow[0]:meta.ywindow[1],
                    meta.xwindow[0]:meta.xwindow[1]]

    # Convert to electrons
    data['flux'] *= gain
    data['err'] *= gain
    data['v0'] *= (gain)**2  # FINDME: should this really be squared

    return data


def bright2dn(data, meta, log, mjy=False):
    """This function converts the data, uncertainty, and variance arrays from
    brightness units (MJy/sr) or (MJy) to raw units (DN).

    Parameters
    ----------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of MJy/sr.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of DN.

    Notes
    -----
    The photometry files can be downloaded from CRDS
    (https://jwst-crds.stsci.edu/browse_db/)

    History:

    - 2021-05-28 kbs
        Initial version
    - 2021-07-21 sz
        Added functionality for MIRI
    - Apr 20, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    """
    # Find the required photfile
    if hasattr(meta, 'photfile') and meta.photfile is not None:
        log.writelog(f'  Using provided photfile={meta.photfile} to '
                     'convert units to DN...',
                     mute=(not meta.verbose))
    else:
        # Retrieve the required reference files if not manually specified
        log.writelog('  Automatically getting reference files to reverse the '
                     'PHOTOM step...', mute=(not meta.verbose))
        if data.attrs['mhdr']['TELESCOP'] != 'JWST':
            message = ('Error: Currently unable to automatically download '
                       'reference files for non-JWST observations!')
            log.writelog(message, mute=True)
            raise ValueError(message)
        meta.photfile = retrieve_ancil(data.attrs['filename'], 'photom')

    # Load response function and wavelength
    phot = fits.getdata(meta.photfile)
    if meta.inst == 'nircam':
        if meta.photometry:
            ind = np.where((phot['filter'] == data.attrs['mhdr']['FILTER']) *
                           (phot['pupil'] == data.attrs['mhdr']['PUPIL'])
                           )[0][0]
        else:
            ind = np.where((phot['filter'] == data.attrs['mhdr']['FILTER']) *
                           (phot['pupil'] == data.attrs['mhdr']['PUPIL']) *
                           (phot['order'] == 1))[0][0]
    elif meta.inst == 'miri':
        ind = np.where((phot['filter'] == data.attrs['mhdr']['FILTER']) *
                       (phot['subarray'] == data.attrs['mhdr']['SUBARRAY'])
                       )[0][0]
    elif meta.inst == 'nirspec':
        ind = np.where((phot['filter'] == data.attrs['mhdr']['FILTER']) *
                       (phot['grating'] == data.attrs['mhdr']['GRATING']) *
                       (phot['slit'] == data.attrs['shdr']['SLTNAME']))[0][0]
    elif meta.inst == 'niriss':
        ind = np.where((phot['filter'] == data.attrs['mhdr']['FILTER']) *
                       (phot['pupil'] == data.attrs['mhdr']['PUPIL']) *
                       (phot['order'] == 1))[0][0]
    else:
        raise ValueError(f'The bright2dn function has not been edited to '
                         f'handle the instrument {meta.inst}.\nIt can '
                         f'currently only handle JWST niriss, nirspec, nircam,'
                         f' and miri observations.')

    if meta.photometry:
        response = 1.0
    else:
        response_wave = phot['wavelength'][ind]
        response_vals = phot['relresponse'][ind]
        igood = np.where(response_wave > 0)[0]
        response_wave = response_wave[igood]
        response_vals = response_vals[igood]
        # Interpolate response at desired wavelengths
        f = spi.interp1d(response_wave, response_vals, kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
        response = f(data.wave_1d)

    scalar = data.attrs['shdr']['PHOTMJSR']
    if mjy:
        scalar *= data.attrs['shdr']['PIXAR_SR']
    # Convert to DN/sec
    data['flux'] /= scalar * response
    data['err'] /= scalar * response
    # FINDME: should this really be squared
    data['v0'] /= (scalar * response)**2

    return data


def bright2flux(data, pixel_area):
    """This function converts the data and uncertainty arrays from brightness
    units (MJy/sr) to flux units (Jy/pix).

    Parameters
    ----------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of MJy/sr.
    pixel_area : ndarray
            Pixel area (arcsec/pix)

    Returns
    -------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of Jy/pix.

    Notes
    -----
    The input arrays Data and Uncd are changed in place.

    History:

    - 2005-06-20 Statia Luszcz, Cornell (shl35@cornell.edu).
    - 2005-10-13 jh
        Renamed, modified doc, removed posmed, fixed
        nimpos default bug (was float rather than int).
    - 2005-10-28 jh
        Updated header to give units being converted
        from/to, made srperas value a calculation
        rather than a constant, added Allen reference.
    - 2005-11-24 jh
        Eliminated NIMPOS.
    - 2008-06-28 jh
        Allow npos=1 case.
    - 2010-01-29 patricio (pcubillos@fulbrightmail.org)
        Converted to python.
    - 2010-11-01 patricio
        Documented, and incorporated scipy.constants.
    - 2021-05-28 kbs
        Updated for JWST
    - 2021-12-09 TJB
        Updated to account for the new DataClass object
    - Apr 20, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    """
    # steradians per square arcsecond
    srperas = arcsec**2.0

    data['flux'] *= srperas * 1e6 * pixel_area
    data['err'] *= srperas * 1e6 * pixel_area
    data['v0'] *= srperas * 1e6 * pixel_area

    return data


def convert_to_e(data, meta, log):
    """This function converts the data object to electrons from MJy/sr or DN/s.

    Parameters
    ----------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of MJy/sr or DN/s.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    data : Xarray Dataset
        Dataset object containing data, uncertainty, and variance arrays in
        units of electrons.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    """
    if data.attrs['shdr']['BUNIT'] == 'ELECTRONS':
        # HST/WFC3 spectra are in ELECTRONS already, so do nothing
        return data, meta

    if data.attrs['shdr']['BUNIT'] == 'MJy/sr':
        # Convert from brightness units (MJy/sr) to DN/s
        log.writelog('  Converting from brightness units (MJy/sr) to '
                     'electrons...')
        data = bright2dn(data, meta, log)
        data = dn2electrons(data, meta, log)
    elif data.attrs['shdr']['BUNIT'] == 'MJy':
        # Convert from brightness units (MJy) to DN/s
        log.writelog('  Converting from brightness units MJy to electrons...')
        data = bright2dn(data, meta, log, mjy=True)
        data = dn2electrons(data, meta, log)
    elif data.attrs['shdr']['BUNIT'] == 'DN/s':
        # Convert from DN/s to e/s
        log.writelog('  Converting from data numbers per second (DN/s) to '
                     'electrons...', mute=(not meta.verbose))
        data = dn2electrons(data, meta, log)
    elif data.attrs['shdr']['BUNIT'] != 'ELECTRONS/S':
        message = (f'Currently unable to convert from input units '
                   f'{data.attrs["shdr"]["BUNIT"]} to electrons.'
                   f'\nTry running Stage 2 again without the photom step.')
        log.writelog(message, mute=True)
        raise ValueError(message)
    else:
        log.writelog('  Converting from electrons per second (e/s) to '
                     'electrons...', mute=(not meta.verbose))

    # Convert from e/s to e
    data = rate2count(data)
    data['flux'].attrs['flux_units'] = 'ELECTRONS'
    data['err'].attrs['flux_units'] = 'ELECTRONS'
    data['v0'].attrs['flux_units'] = 'ELECTRONS'

    return data, meta


def retrieve_ancil(fitsname, reftype='gain'):
    '''Use crds package to find/download the needed ancilliary files.

    This code requires that the CRDS_PATH and CRDS_SERVER_URL environment
    variables be set in your .bashrc file (or equivalent, e.g.
    .bash_profile or .zshrc)

    Parameters
    ----------
    fitsname : str
        The filename of the file currently being analyzed.
    reftype : str
        The ancillary reference file to retrieve (e.g., "gain", or "photom").

    Returns
    -------
    phot_filename : str
        The full path to the photom calibration file.
    gain_filename : str
        The full path to the gain calibration file.

    Notes
    -----
    History:

    - 2022-03-04 Taylor J Bell
        Initial code version.
    - 2022-03-28 Taylor J Bell
        Removed jwst dependency, using crds package now instead.
    '''
    with fits.open(fitsname) as file:
        # Automatically get the best reference files using the information
        # contained in the FITS header and the crds package. The parameters
        # below are easily obtained from model.get_crds_parameters(), but
        # datamodels is a jwst sub-package. Instead, I've resorted to manually
        # populating the required lines for finding gain and photom
        # reference files.
        parameters = {
            "meta.ref_file.crds.context_used": file[0].header["CRDS_CTX"],
            "meta.ref_file.crds.sw_version": file[0].header["CRDS_VER"],
            "meta.instrument.name": file[0].header["INSTRUME"],
            "meta.instrument.detector": file[0].header["DETECTOR"],
            "meta.observation.date": file[0].header["DATE-OBS"],
            "meta.observation.time": file[0].header["TIME-OBS"],
            "meta.exposure.type": file[0].header["EXP_TYPE"],
        }
        observatory = file[0].header['TELESCOP'].lower()
        refiles = crds.getreferences(parameters, [reftype,],
                                     observatory=observatory)

    return refiles[reftype]
