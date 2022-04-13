import numpy as np
import scipy.interpolate as spi
from scipy.constants import arcsec
from astropy.io import fits
import crds

def rate2count(data):
    """This function converts the data, uncertainty, and variance arrays from rate units (#/s) to counts (#).

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in rate units (#/s).

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in count units (#).

    Notes
    -----
    History:
    
    - Mar 7, 2022 Taylor J Bell
        Initial version
    """
    if "EFFINTTM" in data.mhdr.keys():
        int_time = data.mhdr['EFFINTTM']
    elif "EXPTIME" in data.mhdr.keys():
        int_time = data.mhdr['EXPTIME']
    else:
        raise ValueError('No FITS header keys found to permit conversion from rate units (#/s) to counts (#)')
    
    data.subdata *= int_time
    data.suberr  *= int_time
    data.subv0   *= int_time

    return data

def dn2electrons(data, meta):
    """This function converts the data, uncertainty, and variance arrays from raw units (DN) to electrons.

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of DN.
    meta:   MetaClass
        The metadata object.

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of electrons.

    Notes
    -----
    The gain files can be downloaded from CRDS (https://jwst-crds.stsci.edu/browse_db/)

    History:
    
    - Jun 2021 Kevin Stevenson
        Initial version
    - Jul 2021
        Added gainfile rotation
    """
    # Subarray parameters
    xstart  = data.mhdr['SUBSTRT1']
    ystart  = data.mhdr['SUBSTRT2']
    nx      = data.mhdr['SUBSIZE1']
    ny      = data.mhdr['SUBSIZE2']

    # Load gain array in units of e-/ADU
    gain    = fits.getdata(meta.gainfile)[ystart:ystart+ny,xstart:xstart+nx]

    # Like in the case of MIRI data, the gain file data has to be rotated by 90 degrees
    if data.shdr['DISPAXIS']==2:
        gain = np.swapaxes(gain, 0, 1)

    # Gain subarray
    subgain = gain[meta.ywindow[0]:meta.ywindow[1],meta.xwindow[0]:meta.xwindow[1]]

    # Convert to electrons
    data.subdata *= subgain
    data.suberr  *= subgain
    data.subv0   *= (subgain)**2

    return data

def bright2dn(data, meta, mjy=False):
    """This function converts the data, uncertainty, and variance arrays from brightness units (MJy/sr) or (MJy) to raw units (DN).

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of MJy/sr.
    meta:   MetaClass
        The metadata object.

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of DN.

    Notes
    -----
    The photometry files can be downloaded from CRDS (https://jwst-crds.stsci.edu/browse_db/)

    History:

    - 2021-05-28 kbs
        Initial version
    - 2021-07-21 sz
        Added functionality for MIRI
    """
    # Load response function and wavelength
    foo = fits.getdata(meta.photfile)
    if meta.inst == 'nircam':
        ind = np.where((foo['filter'] == data.mhdr['FILTER']) * (foo['pupil'] == data.mhdr['PUPIL']) * (foo['order'] == 1))[0][0]
    elif meta.inst == 'miri':
        ind = np.where((foo['filter'] == data.mhdr['FILTER']) * (foo['subarray'] == data.mhdr['SUBARRAY']))[0][0]
    elif meta.inst == 'nirspec':
        ind = np.where((foo['filter'] == data.mhdr['FILTER']) * (foo['grating'] == data.mhdr['GRATING']) * (foo['slit'] == data.shdr['SLTNAME']))[0][0] 
    elif meta.inst == 'niriss':
        ind = np.where((foo['filter'] == data.mhdr['FILTER']) * (foo['pupil'] == data.mhdr['PUPIL']) * (foo['order'] == 1))[0][0]
    else:
        raise ValueError(f'The bright2dn function has not been edited to handle the instrument {meta.inst},and can currently only handle JWST niriss, nirspec, nircam, and miri observations.')

    response_wave = foo['wavelength'][ind]
    response_vals = foo['relresponse'][ind]
    igood = np.where(response_wave > 0)[0]
    response_wave = response_wave[igood]
    response_vals = response_vals[igood]
    # Interpolate response at desired wavelengths
    f = spi.interp1d(response_wave, response_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')
    response = f(data.subwave)

    scalar = data.shdr['PHOTMJSR']
    if mjy == True:
        scalar *= data.shdr['PIXAR_SR']
    # Convert to DN/sec
    data.subdata /= scalar * response
    data.suberr  /= scalar * response
    data.subv0   /= (scalar * response)**2

    return data


def bright2flux(data, pixel_area):
    """This function converts the data and uncertainty arrays from brightness units (MJy/sr) to flux units (Jy/pix).

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of MJy/sr.
    pixel_area:  ndarray
            Pixel area (arcsec/pix)

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of Jy/pix.
    
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
    """
    # steradians per square arcsecond
    srperas = arcsec**2.0

    data.subdata *= srperas * 1e6 * pixel_area
    data.suberr  *= srperas * 1e6 * pixel_area
    data.subv0   *= srperas * 1e6 * pixel_area

    return data

def convert_to_e(data, meta, log):
    """This function converts the data object to electrons from MJy/sr or DN/s.

    Parameters
    ----------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of MJy/sr or DN/s.
    meta:   MetaClass
        The metadata object.
    log:    logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    data:   DataClass
        Data object containing data, uncertainty, and variance arrays in units of electrons.
    meta:   MetaClass
        The metadata object.
    """
    if data.shdr['BUNIT']=='ELECTRONS':
        # HST/WFC3 spectra are in ELECTRONS already, so do nothing
        return data, meta

    if data.shdr['BUNIT'] != 'ELECTRONS/S':
        log.writelog('  Automatically getting reference files to convert units to electrons', mute=(not meta.verbose))
        if data.mhdr['TELESCOP'] != 'JWST':
            log.writelog('Error: Currently unable to automatically download reference files for non-jwst observations!', mute=True)
            raise ValueError('Error: Currently unable to automatically download reference files for non-jwst observations!')
        meta.photfile, meta.gainfile = retrieve_ancil(data.filename)
    else:
        log.writelog('  Converting from electrons per second (e/s) to electrons', mute=(not meta.verbose))

    if data.shdr['BUNIT'] == 'MJy/sr':
        # Convert from brightness units (MJy/sr) to DN/s
        log.writelog('  Converting from brightness units (MJy/sr) to electrons')
        data = bright2dn(data, meta)
        data = dn2electrons(data, meta)
    elif data.shdr['BUNIT'] == 'MJy':
        # Convert from brightness units (MJy) to DN/s
        log.writelog('  Converting from brightness units MJy to electrons')
        data = bright2dn(data, meta, mjy=True)
        data = dn2electrons(data, meta)
    elif data.shdr['BUNIT'] == 'DN/s':
        # Convert from DN/s to e/s
        log.writelog('  Converting from data numbers per second (DN/s) to electrons', mute=(not meta.verbose))
        data = dn2electrons(data, meta)
    elif data.shdr['BUNIT'] != 'ELECTRONS/S':
        log.writelog(f'Currently unable to convert from input units {data.shdr["BUNIT"]} to electrons - try running Stage 2 again without the photom step.', mute=True)
        raise ValueError(f'Currently unable to convert from input units {data.shdr["BUNIT"]} to electrons - try running Stage 2 again without the photom step.')
    
    # Convert from e/s to e
    data = rate2count(data)

    return data, meta

def retrieve_ancil(fitsname):
    '''Use crds package to find/download the needed ancilliary files.

    This code requires that the CRDS_PATH and CRDS_SERVER_URL environment variables be set
    in your .bashrc file (or equivalent, e.g. .bash_profile or .zshrc)

    Parameters
    ----------
    fitsname:   
        The filename of the file currently being analyzed.

    Returns
    -------
    phot_filename:  str
        The full path to the photom calibration file.
    gain_filename:  str
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
        # Automatically get the best reference files using the information contained in the FITS header and the crds package.
        # The parameters below are easily obtained from model.get_crds_parameters(), but datamodels is a jwst sub-package.
        # Instead, I've resorted to manually populating the required lines for finding gain and photom reference files.
        parameters = {
            "meta.ref_file.crds.context_used": file[0].header["CRDS_CTX"],
            "meta.ref_file.crds.sw_version": file[0].header["CRDS_VER"],
            "meta.instrument.name": file[0].header["INSTRUME"],
            "meta.instrument.detector": file[0].header["DETECTOR"],
            "meta.observation.date": file[0].header["DATE-OBS"],
            "meta.observation.time": file[0].header["TIME-OBS"],
            "meta.exposure.type": file[0].header["EXP_TYPE"],
            }
        refiles = crds.getreferences(parameters, ["gain", "photom"], observatory=file[0].header['TELESCOP'].lower())
        gain_filename = refiles["gain"]
        phot_filename = refiles["photom"]

    return phot_filename, gain_filename 
