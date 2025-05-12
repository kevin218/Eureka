
import numpy as np
import re
import shutil
import os
from astroquery.mast import Observations
import astropy.io.fits as pf
from astropy.io import ascii


def columnNames():
    """Print column names from MAST Observation table.

    Notes
    -----
    History:

    - June 2022 Kevin Stevenson
        Initial version
    """
    meta_table = Observations.get_metadata("observations")
    for val in meta_table['Column Name']:
        print(val)
    return


def writeTable_JWST(proposal_id, observation, visit, filename, format='csv'):
    """Write all products from specified visit to an ASCII file

    Parameters
    ----------
    proposal_id : str or int
        JWST proposal/program ID (e.g., 1366).
    observation : str or int
        JWST observation number listed on the Visit Status Report (e.g., 2).
        See www.stsci.edu/cgi-bin/get-visit-status?id=XXXXX&observatory=JWST,
        where XXXXX is the proposal/program ID.
    visit : str or int
        JWST visit number listed on the Visit Status Report (e.g., 1).
        See www.stsci.edu/cgi-bin/get-visit-status?id=XXXXX&observatory=JWST,
        where XXXXX is the proposal/program ID.
    filename : str; optional
        The file format to use. Defaults to 'csv'.

    Notes
    -----
    History:

    - July 2022 Kevin Stevenson
        Initial version
    """
    # Convert to string
    if type(proposal_id) is not str:
        proposal_id = str(proposal_id).zfill(5)
    if type(observation) is not str:
        observation = str(observation).zfill(3)
    if type(visit) is not str:
        visit = str(visit).zfill(3)
    # Specify obsid using wildcards
    obsid = 'jw'+proposal_id+observation+visit+'*'

    # Query MAST for requested visit
    sci_table = Observations.query_criteria(proposal_id=proposal_id,
                                            obs_id=obsid)
    ascii.write(sci_table, filename, format=format)
    return


def login(mast_token=None):
    """Log into the MAST portal.

    Parameters
    ----------
    mast_token : str; optional
        The token to authenticate the user. Default is None.
        This can be generated at https://auth.mast.stsci.edu/token.
        If not supplied, it will be prompted for if not in the keyring
        or set via $MAST_API_TOKEN.

    Notes
    -----
    History:

    - July 2022 Kevin Stevenson
        Initial version
    """
    Observations.login(mast_token)
    return


def logout():
    """Log out of current MAST session.

    Notes
    -----
    History:

    - July 2022 Kevin Stevenson
        Initial version
    """
    Observations.logout()
    return


def downloadHST(proposal_id, visit, inst='WFC3', download_dir='.',
                subgroup='IMA'):
    """Download observation visit number from specified proposal ID.

    Parameters
    ----------
    proposal_id : str or int
        HST proposal/program ID (e.g., 13467).
    visit : str or int
        HST visit number listed on the Visit Status Report (e.g., 60).
        See https://www.stsci.edu/cgi-bin/get-visit-status?id=XXXXX,
        where XXXXX is the proposal/program ID.
    inst : str; optional
        HST instrument name, can be upper or lower case.
        Supported options include: WFC3, STIS, COS, or FGS.
        Defaults to 'WFC3'.
    download_dir : str; optional
        Temporary download directory will be 'download_dir'/mastDownload/...
        Defaults to '.'.
    subgroup : str; optional
        FITS file type (usually IMA, sometimes FLT). Defaults to 'IMA'.

    Returns
    -------
    result : AstroPy Table
        The manifest of files downloaded.

    Notes
    -----
    History:

    - June 2022 Kevin Stevenson
        Initial version
    """
    # Convert to string
    if type(proposal_id) is not str:
        proposal_id = str(proposal_id).zfill(5)
    if type(visit) is not str:
        visit = str(visit).zfill(2)
    # Determine instrument ID, as indicated by a single letter
    # L=COS; I=WFC3; J=ACS; N=NICMOS; O=STIS; U=WFPC2; W=WFPC;
    # X=FOC; Y=FOS; Z=GHRS; F=FGS; V=HSP;
    if inst.casefold() == 'wfc3':
        iid = 'i'
    elif inst.casefold() == 'stis':
        iid = 'o'
    elif inst.casefold() == 'cos':
        iid = 'l'
    elif inst.casefold() == 'fgs':
        iid = 'f'
    else:
        print(f"Unknown instrument: {inst}")
        print("Supported options include: WFC3, STIS, COS, or FGS")
        return None
    # Specify obsid using wildcards
    obsid = iid+'*'+visit+'*'

    # Query MAST for requested visit
    sci_table = Observations.query_criteria(proposal_id=proposal_id,
                                            obs_id=obsid)

    # AstroQuery doesn't support single character wildcards,
    # so sometimes it identifies extra, unwanted files.
    # Using regex to remove these unwanted files
    counter = 0
    for ii, val in enumerate(sci_table['obs_id']):
        if not re.search('i...'+visit, val):
            sci_table.remove_row(ii-counter)
            # Increment counter to get index right
            # when multiple files need to be removed
            counter += 1

    # Get product list
    data_products_by_id = Observations.get_product_list(sci_table)

    # Filter for IMA files
    filtered = Observations.filter_products(
        data_products_by_id, productSubGroupDescription=subgroup)
    nimage = np.sum(filtered['dataproduct_type'] == 'image')
    nspec = np.sum(filtered['dataproduct_type'] == 'spectrum')
    print("Number of image products:", nimage)
    print("Number of spectrum products:", nspec)

    # Download data products
    result = Observations.download_products(filtered, curl_flag=False,
                                            download_dir=download_dir)
    return result


def filterJWST(proposal_id, observation, visit, calib_level, subgroup):
    """Find JWST data products by applying standard filters.

    Parameters
    ----------
    proposal_id : str or int
        JWST proposal/program ID (e.g., 1366).
    observation : str or int
        JWST observation number listed on the Visit Status Report (e.g., 2).
        See www.stsci.edu/cgi-bin/get-visit-status?id=XXXXX&observatory=JWST,
        where XXXXX is the proposal/program ID.
    visit : str or int
        JWST visit number listed on the Visit Status Report (e.g., 2).
        See www.stsci.edu/cgi-bin/get-visit-status?id=XXXXX&observatory=JWST,
        where XXXXX is the proposal/program ID.
    calib_level : list or int
        Product Calibration Level (0 = raw, 1 = uncalibrated, 2 = calibrated,
        3 = science product, 4 = contributed science product)
    subgroup : str
        FITS file type, varies by calib_level.
        1: UNCAL, GS-ACQ1, GS-ACQ2, GS-FG, GS-ID, GS-TRACK
        2: CAL, CALINTS, RATE, RATEINTS, X1DINTS, ANNNN_CRFINTS,
        GS-ACQ1, GS-ACQ2, GS-FG, GS-ID, GS-TRACK, RAMP
        3: X1DINTS, WHTLT

    Returns
    -------
    table : AstroPy Table
        The filtered table of data products.

    Notes
    -----
    History:

    - July 2022 Kevin Stevenson
        Initial version
    """
    # Convert to string
    if type(proposal_id) is not str:
        proposal_id = str(proposal_id).zfill(5)
    if type(observation) is not str:
        observation = str(observation).zfill(3)
    if type(visit) is not str:
        visit = str(visit).zfill(3)
    if type(calib_level) is int:
        calib_level = [calib_level]
    # Specify obsid using wildcards, obs_id can come in two flavours
    obsid = f'jw{proposal_id}{observation}{visit}_04*'
    obsid2 = f'jw{proposal_id}-o{observation}_t*'

    # Query MAST for requested visit
    sci_table = Observations.query_criteria(proposal_id=proposal_id,
                                            obs_id=obsid)
    sci_table2 = Observations.query_criteria(proposal_id=proposal_id,
                                             obs_id=obsid2)

    table = []
    if len(sci_table) > 0:
        # Get product list
        data_products_by_id = Observations.get_product_list(sci_table)
        # Filter for desired files
        table = Observations.filter_products(
            data_products_by_id, productSubGroupDescription=subgroup,
            calib_level=calib_level)
    if len(sci_table2) > 0:
        # Get product list
        data_products_by_id2 = Observations.get_product_list(sci_table2)
        # Filter for desired files
        table2 = Observations.filter_products(
            data_products_by_id2, productSubGroupDescription=subgroup,
            calib_level=calib_level)

    if len(sci_table) > 0 and len(sci_table2) > 0:
        # Concatenate tables one row at a time, checking for uniqueness
        for row in range(len(table2)):
            if ((table['obsID'] != table2['obsID'][row])).all():
                table.add_row(table2[row])
    elif len(sci_table2) > 0:
        table = table2

    print("Total number of data products:", len(table))
    print("Number of data products with exclusive access:",
          np.sum(table['dataRights'] == 'EXCLUSIVE_ACCESS'))

    return table


def consolidate(result, final_dir):
    """Consolidate downloaded files into a single directory

    Parameters
    ----------
    result : AstroPy Table
        The manifest of files downloaded, returned from
        mastDownload.download().
    final_dir : str
        Final destination of files.

    Notes
    -----
    History:

    - June 2022 Kevin Stevenson
        Initial version
    """
    # Create directory
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    # Move files
    for path in result['Local Path']:
        filename = path.split('/')[-1]
        try:
            shutil.move(path, os.path.join(final_dir, filename))
        except:
            print(f"File not found: {path}")
    return


def sortJWST(source_dir, target_dir, filetype):
    """

    """
    # Create directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move files
    for filename in os.listdir(source_dir):
        if filename.endswith(filetype):
            shutil.move(os.path.join(source_dir, filename),
                        os.path.join(target_dir, filename))

    return


def sortHST(final_dir, sci_dir='sci', cal_dir='cal'):
    """Sort files into science and calibration subdirectories.

    Parameters
    ----------
    final_dir : str
        Final destination of files.
    sci_dir : str; optional
        Name of science subdirectory within 'final_dir'.
        Defaults to 'sci'.
    cal_dir : str; optional
        Name of calibration subdirectory within 'final_dir'.
        Defaults to 'cal'.

    Notes
    -----
    History:

    - June 2022 Kevin Stevenson
        Initial version
    """
    # Create directories
    if not os.path.exists(os.path.join(final_dir, sci_dir)):
        os.makedirs(os.path.join(final_dir, sci_dir))
    if not os.path.exists(os.path.join(final_dir, cal_dir)):
        os.makedirs(os.path.join(final_dir, cal_dir))

    # Move files
    for filename in os.listdir(final_dir):
        if filename.endswith('.fits'):
            hdr = pf.getheader(os.path.join(final_dir, filename))
            # If spectrum, move to science directory
            if hdr['filter'].startswith('G'):
                shutil.move(os.path.join(final_dir, filename),
                            os.path.join(final_dir, sci_dir, filename))
            # If image, move to calibration directory
            elif hdr['filter'].startswith('F'):
                shutil.move(os.path.join(final_dir, filename),
                            os.path.join(final_dir, cal_dir, filename))
            # Otherwise, leave file in current location
    return


def cleanup(download_dir='.'):
    """Remove empty folders from download directory.

    Parameters
    ----------
    download_dir : str; optional
        Temporary download directory specified for mastDownload.download().
        Defaults to '.'.

    Notes
    -----
    History:

    - June 2022 Kevin Stevenson
        Initial version
    """
    src_dir = os.path.join(download_dir, 'mastDownload')
    for dirpath, _, _ in os.walk(src_dir, topdown=False):
        try:
            # Remove empty folder
            os.rmdir(dirpath)
        except OSError as ex:
            # Report any non-empty folders
            print(ex)
    return
