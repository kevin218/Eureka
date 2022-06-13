# !/usr/bin/python
# -*- coding: latin-1 -*-
"""
A module for utility funtions
"""

import glob
import itertools
import os
import re
import requests
import shutil
import urllib

from astropy.io import fits
import bokeh.palettes as bpal
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from svo_filters import svo

# Supported profiles
PROFILES = ['uniform', 'linear', 'quadratic',
            'square-root', 'logarithmic', 'exponential',
            '3-parameter', '4-parameter']

# Supported filters
FILTERS = svo.filters()

# Get the location of EXOCTK_DATA environvment variable
# and check that it is valid
EXOCTK_DATA = os.environ.get('EXOCTK_DATA')

# If the variable is blank or doesn't exist
HOME_DIR = os.path.expanduser('~')
ON_TRAVIS_OR_RTD = (HOME_DIR == '/home/travis' or HOME_DIR == '/Users/travis'
                    or HOME_DIR == '/home/docs')
if not ON_TRAVIS_OR_RTD:
    # if not EXOCTK_DATA:
    #     print(
    #         'WARNING (only important for Stage 5): The $EXOCTK_DATA '
    #         'environment variable is not set.  Please set the '
    #         'value of this variable to point to the location of the
    #         'exoctk_data download folder.  Users may retreive this folder by'
    #         ' clicking the "ExoCTK Data Download" button on the ExoCTK '
    #         'website, or by using the exoctk.utils.download_exoctk_data() '
    #         'function.')
    if EXOCTK_DATA:
        # If the variable exists but doesn't point to a real location
        if not os.path.exists(EXOCTK_DATA):
            print('WARNING (only important for Stage 5): The $EXOCTK_DATA '
                  'environment variable is set to a location that '
                  'cannot be accessed.')

        # If the variable exists, points to a real location, but
        # is missing contents
        for item in ['exoctk_contam', 'exoctk_log', 'fortney', 'generic',
                     'groups_integrations', 'modelgrid']:
            if item not in [os.path.basename(item)
                            for item in
                            glob.glob(os.path.join(EXOCTK_DATA, '*'))]:
                print('WARNING (only important for Stage 5): Missing {}/ '
                      'directory from {}. Please ensure that the ExoCTK data '
                      'package has been downloaded. Users may retrieve this '
                      'package by clicking the "ExoCTK Data Download" button '
                      'on the ExoCTK website, or by using the '
                      'exoctk.utils.download_exoctk_data() '
                      'function'.format(item, EXOCTK_DATA))

        EXOCTK_CONTAM_DIR = os.path.join(EXOCTK_DATA, 'exoctk_contam/')
        EXOCTKLOG_DIR = os.path.join(EXOCTK_DATA, 'exoctk_log/')
        FORTGRID_DIR = os.path.join(EXOCTK_DATA, 'fortney/')
        GENERICGRID_DIR = os.path.join(EXOCTK_DATA, 'generic/')
        GROUPS_INTEGRATIONS_DIR = os.path.join(EXOCTK_DATA,
                                               'groups_integrations/')
        MODELGRID_DIR = os.path.join(EXOCTK_DATA, 'modelgrid/')


def download_exoctk_data(download_location=os.path.expanduser('~')):
    """Retrieves the ``exoctk_data`` materials from Box, downloads them
    to the user's local machine, uncompresses the files, and arranges
    them into an ``exoctk_data`` directory.

    Parameters
    ----------
    download_location : string; optional
        The path to where the ExoCTK data package will be downloaded.
        The default setting is the user's $HOME directory.
    """
    print('\nDownloading ExoCTK data package. This may take a few minutes.')
    print(f'Materials will be downloaded to {download_location}/exoctk_data/')
    print()

    # Ensure the exoctk_data/ directory exists in user's home directory
    exoctk_data_dir = os.path.join(download_location, 'exoctk_data')
    try:
        if not os.path.exists(exoctk_data_dir):
            os.makedirs(exoctk_data_dir)
    except PermissionError:
        print(f'Data download failed. Unable to create {exoctk_data_dir}. '
              f'Please check permissions.')

    # URLs to download contents
    url_base = 'https://data.science.stsci.edu/redirect/JWST/ExoCTK/compressed'
    urls = [url_base+'/exoctk_contam.tar.gz',
            url_base+'/exoctk_log.tar.gz',
            url_base+'/groups_integrations.tar.gz',
            url_base+'/fortney.tar.gz',
            url_base+'/generic.tar.gz',
            url_base+'/modelgrid_ATLAS9.tar.gz',
            url_base+'/modelgrid_ACES_1.tar.gz',
            url_base+'/modelgrid_ACES_2.tar.gz']

    # Build landing paths for downloads
    download_paths = [os.path.join(exoctk_data_dir, os.path.basename(url))
                      for url in urls]

    # Perform the downloads
    for i, url in enumerate(urls):
        landing_path = os.path.join(exoctk_data_dir, os.path.basename(url))
        print(f'({i+1}/{len(urls)}) Downloading data to {landing_path} from '
              f'{url}')
        with requests.get(url, stream=True) as response:
            with open(landing_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=2048):
                    if chunk:
                        f.write(chunk)
    print('\nDownload complete\n')

    # Uncompress data
    print('Uncompressing data:\n')
    for path in download_paths:
        # Uncompress data
        print('\t{}'.format(path))
        shutil.unpack_archive(path, exoctk_data_dir)
        # Remove original .tar.gz files
        os.remove(path)

    # Combine modelgrid directories
    print('\nOrganizing files into exoctk_data/ directory')
    try:
        os.makedirs(os.path.join(exoctk_data_dir, 'modelgrid', 'ATLAS9'))
        os.makedirs(os.path.join(exoctk_data_dir, 'modelgrid', 'ACES'))
    except FileExistsError:
        pass
    modelgrid_files = glob.glob(os.path.join(exoctk_data_dir, 'modelgrid.*',
                                             '*'))
    for src in modelgrid_files:
        if 'ATLAS9' in src:
            dst = os.path.join(exoctk_data_dir, 'modelgrid', 'ATLAS9')
        elif 'ACES_' in src:
            dst = os.path.join(exoctk_data_dir, 'modelgrid', 'ACES')
        try:
            shutil.move(src, dst)
        except shutil.Error:
            print('Unable to organize modelgrid/ directory')
    shutil.rmtree(os.path.join(exoctk_data_dir, 'modelgrid.ATLAS9'))
    shutil.rmtree(os.path.join(exoctk_data_dir, 'modelgrid.ACES_1'))
    shutil.rmtree(os.path.join(exoctk_data_dir, 'modelgrid.ACES_2'))

    print('Completed!')


def color_gen(colormap='viridis', key=None, n=10):
    """Color generator for Bokeh plots.

    Parameters
    ----------
    colormap : str, sequence
        The name of the color map.
    key : str; optional
        The palette key. Defaults to None which uses the first palette key.
    n : int; optional
        If palette is callable, the argument to that function. Defaults to 10.

    Returns
    -------
    generator
        A generator for the color palette.
    """
    if colormap in dir(bpal):
        palette = getattr(bpal, colormap)

        if isinstance(palette, dict):
            if key is None:
                key = list(palette.keys())[0]
            palette = palette[key]

        elif callable(palette):
            palette = palette(n)

        else:
            raise TypeError("pallette must be a bokeh palette name or a "
                            "sequence of color hex values.")

    elif isinstance(colormap, (list, tuple)):
        palette = colormap

    else:
        raise TypeError("pallette must be a bokeh palette name or a sequence "
                        "of color hex values.")

    yield from itertools.cycle(palette)


COLORS = color_gen('Category10', 10)


def interp_flux(mu, flux, params, values):
    """Interpolate a cube of synthetic spectra for a
    given index of mu.

    Parameters
    ----------
    mu : int
        The index of the (Teff, logg, FeH, *mu*, wavelength)
        data cube to interpolate.
    flux : np.ndarray
        The 5D data array.
    params : list
        A list of each free parameter range.
    values : list
        A list of each free parameter values.

    Returns
    -------
    tuple
        The array of new flux values, and the generators.
    """
    # Iterate over each wavelength (-1 index of flux array)
    shp = flux.shape[-1]
    flx = np.zeros(shp)
    generators = []
    for lam in range(shp):
        interp_f = RegularGridInterpolator(params, flux[:, :, :, mu, lam])
        f, = interp_f(values)

        flx[lam] = f
        generators.append(interp_f)

    return flx, generators


def calc_zoom(R_f, arr):
    """Calculate the zoom factor required to make the given
    array into the given resolution.

    Parameters
    ----------
    R_f : int
        The desired final resolution of the wavelength array.
    arr : array-like
        The array to zoom.

    Returns
    -------
    z : float
        The zoom factor
    """
    # Get initial resolution
    lam = arr[-1]-arr[0]
    d_lam_i = np.nanmean(np.diff(arr))
    # R_i = lam/d_lam_i

    # Calculate zoom
    d_lam_f = lam/R_f
    z = d_lam_i/d_lam_f

    return z


def rebin_spec(spec, wavnew, oversamp=100, plot=False):
    """Rebin a spectrum to a new wavelength array while preserving
    the total flux.

    Parameters
    ----------
    spec : array-like
        The wavelength and flux to be binned.
    wavenew : array-like
        The new wavelength array.
    oversamp : int; optional
        The oversampling factor. Defaults to 100.
    plot : bool; optional
        Unused. Defaults to False.

    Returns
    -------
    np.ndarray
        The rebinned flux
    """
    wave, flux = spec
    nlam = len(wave)
    x0 = np.arange(nlam, dtype=float)
    x0int = np.arange((nlam-1.) * oversamp + 1., dtype=float)/oversamp
    w0int = np.interp(x0int, x0, wave)
    spec0int = np.interp(w0int, wave, flux)/oversamp

    # Set up the bin edges for down-binning
    maxdiffw1 = np.diff(wavnew).max()
    w1bins = np.concatenate(([wavnew[0]-maxdiffw1],
                             .5*(wavnew[1::]+wavnew[0: -1]),
                             [wavnew[-1]+maxdiffw1]))

    # Bin down the interpolated spectrum:
    w1bins = np.sort(w1bins)
    nbins = len(w1bins)-1
    specnew = np.zeros(nbins)
    inds2 = [[w0int.searchsorted(w1bins[ii], side='left'),
              w0int.searchsorted(w1bins[ii+1], side='left')]
             for ii in range(nbins)]

    for ii in range(nbins):
        specnew[ii] = np.sum(spec0int[inds2[ii][0]: inds2[ii][1]])

    return specnew


def writeFITS(filename, extensions, headers=()):
    """Write some data to a new FITS file.

    Parameters
    ----------
    filename : str
        The filename of the output FITS file.
    extensions : dict
        The extension name and associated data to include.
        in the file
    headers : array-like; optional
        The (keyword, value, comment) groups for the PRIMARY
        header extension. Defaults to ().
    """
    # Write the arrays to a FITS file
    prihdu = fits.PrimaryHDU()
    prihdu.name = 'PRIMARY'
    hdulist = fits.HDUList([prihdu])

    # Write the header to the PRIMARY HDU
    hdulist['PRIMARY'].header.extend(headers, end=True)

    # Write the data to the HDU
    for k, v in extensions.items():
        hdulist.append(fits.ImageHDU(data=v, name=k))

    # Write the file
    hdulist.writeto(filename, clobber=True)
    hdulist.close()

    # Insert END card to prevent header error
    # hdulist[0].header.tofile(filename, endcard=True, clobber=True)


def filter_table(table, **kwargs):
    """Retrieve the filtered rows.

    Parameters
    ----------
    table : astropy.table.Table, pandas.DataFrame
        The table to filter.
    **kwargs : dict
        param : str
            The parameter to filter by, e.g. 'Teff'.
        value : str, float, int, sequence
            The criteria to filter by,
            which can be single valued like 1400
            or a range with operators [<,<=,>,>=],
            e.g. ('>1200','<=1400').

    Returns
    -------
    astropy.table.Table, pandas.DataFrame
        The filtered table.
    """
    for param, value in kwargs.items():

        # Check it is a valid column
        if param not in table.colnames:
            raise KeyError("No column named {}".format(param))

        # Wildcard case
        if isinstance(value, (str, bytes)) and '*' in value:

            # Get column data
            data = list(map(str, table[param]))

            if not value.startswith('*'):
                value = '^'+value
            if not value.endswith('*'):
                value = value+'$'

            # Strip double quotes and decod
            value = value.replace("'", '')\
                         .replace('"', '')\
                         .replace('*', '(.*)')

            # Regex
            reg = re.compile(value, re.IGNORECASE)
            keep = list(filter(reg.findall, data))

            # Get indexes
            idx = np.where([i in keep for i in data])

            # Filter table
            table = table[idx]

        else:

            # Make single value string into conditions
            if isinstance(value, str):

                # Check for operator
                if any([value.startswith(o) for o in ['<', '>', '=']]):
                    value = [value]

                # Assume eqality if no operator
                else:
                    value = ['=='+value]

            # Turn numbers into strings
            if isinstance(value, (int, float, np.float16)):
                value = ["=={}".format(value)]

            # Iterate through multiple conditions
            for cond in value:

                # Equality
                if cond.startswith('='):
                    v = cond.replace('=', '')
                    if v.replace('.', '', 1).isdigit():
                        table = table[table[param] == eval(v)]
                    else:
                        table = table[table[param] == v]

                # Less than or equal
                elif cond.startswith('<='):
                    v = cond.replace('<=', '')
                    table = table[table[param] <= eval(v)]

                # Less than
                elif cond.startswith('<'):
                    v = cond.replace('<', '')
                    table = table[table[param] < eval(v)]

                # Greater than or equal
                elif cond.startswith('>='):
                    v = cond.replace('>=', '')
                    table = table[table[param] >= eval(v)]

                # Greater than
                elif cond.startswith('>'):
                    v = cond.replace('>', '')
                    table = table[table[param] > eval(v)]

                else:
                    raise ValueError("'{}' operator not valid.".format(cond))

    return table


def find_closest(axes, points, n=1, values=False):
    """Find the n-neighboring elements of a given value in an array.

    Parameters
    ----------
    axes : list, np.array
        The array(s) to search.
    points : array-like, float
        The point(s) to search for.
    n : int; optional
        The number of values to the left and right of the points.
        Defaults to 1.
    values : bool; optional
        Defaults to False.

    Returns
    -------
    np.ndarray
        The n-values to the left and right of 'points' in 'axes'.
    """
    results = []
    if not isinstance(axes, list):
        axes = [axes]
        points = [points]

    for i, (axis, point) in enumerate(zip(axes, points)):
        if point >= min(axis) and point <= max(axis):
            axis = np.asarray(axis)
            idx = np.clip(axis.searchsorted(point), 1, len(axis)-1)
            slc = slice(max(0, idx-n), min(idx+n, len(axis)))

            if values:
                result = axis[slc]
            else:

                result = np.arange(0, len(axis))[slc].astype(int)

            results.append(result)
        else:
            print('Point {} outside grid.'.format(point))
            return

    return results


def build_target_url(target_name):
    '''Build restful api url based on target name.

    Parameters
    ----------
    target_name : string
        The name of the target transit.

    Returns
    -------
    target_url : string
    '''
    # Encode the target name string.
    encode_target_name = urllib.parse.quote(target_name, encoding='utf-8')
    target_url = (f"https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
                  f"{encode_target_name}/properties/")

    return target_url


def get_canonical_name(target_name):
    '''Get ExoMAST prefered name for exoplanet.

    Parameters
    ----------
    target_name : string
        The name of the target transit.

    Returns
    -------
    canonical_name : string
    '''
    target_url = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/identifiers/"

    # Create params dict for url parsing. Easier than trying
    # to format yourself.
    params = {"name": target_name}

    r = requests.get(target_url, params=params)
    planetnames = r.json()
    canonical_name = planetnames['canonicalName']

    return canonical_name


def get_env_variables():
    """Returns a dictionary containing various environment variable
    information.

    Returns
    -------
    env_variables : dict
        A dictionary containing various environment variable data
    """
    env_variables = {}

    # Get the location of EXOCTK_DATA environvment variable and check
    # that it is valid
    env_variables['exoctk_data'] = os.environ.get('EXOCTK_DATA')

    # If the variable is blank or doesn't exist
    ON_TRAVIS = (os.path.expanduser('~') == '/home/travis' or
                 os.path.expanduser('~') == '/Users/travis')
    if not ON_TRAVIS:
        if not env_variables['exoctk_data']:
            raise ValueError(
                'The $EXOCTK_DATA environment variable is not set. Please set '
                'the value of this variable to point to the location of the '
                'ExoCTK data download folder.  Users may retreive this folder '
                'by clicking the "ExoCTK Data Download" button on the ExoCTK '
                'website.'
            )

        # If the variable exists but doesn't point to a real location
        if not os.path.exists(env_variables['exoctk_data']):
            raise FileNotFoundError('The $EXOCTK_DATA environment variable is '
                                    'set to a location that cannot be '
                                    'accessed.')

        # If the variable exists, points to a real location,
        # but is missing contents
        for item in ['modelgrid', 'fortney', 'exoctk_log', 'generic']:
            dirs = [os.path.basename(item)
                    for item in
                    glob.glob(os.path.join(env_variables['exoctk_data'], '*'))]
            if item not in dirs:
                raise KeyError(f'Missing {item}/ directory from '
                               f'{env_variables["exoctk_data"]}')

    env_variables['modelgrid_dir'] = \
        os.path.join(env_variables['exoctk_data'], 'modelgrid/')
    env_variables['fortgrid_dir'] = \
        os.path.join(env_variables['exoctk_data'], 'fortney/')
    env_variables['exoctklog_dir'] = \
        os.path.join(env_variables['exoctk_data'], 'exoctk_log/')
    env_variables['genericgrid_dir'] = \
        os.path.join(env_variables['exoctk_data'], 'generic/')

    return env_variables


def get_target_data(target_name):
    """Send request to exomast restful api for target information.

    Parameters
    ----------
    target_name : string
        The name of the target transit

    Returns
    -------
    target_data : json
        json object with target data.
    """
    canonical_name = get_canonical_name(target_name)

    target_url = build_target_url(canonical_name)

    r = requests.get(target_url)

    if r.status_code == 200:
        target_data = r.json()
    else:
        print('Whoops, no data for this target!')

    # Some targets have multiple catalogs
    # nexsci is the first choice.
    if len(target_data) > 1:
        # Get catalog names from exomast and make then the keys of a dictionary
        # and the values are its position in the json object.
        catalog_dict = {data['catalog_name']: index
                        for index, data in enumerate(target_data)}

        # Parse based on catalog accuracy.
        if 'nexsci' in list(catalog_dict.keys()):
            target_data = target_data[catalog_dict['nexsci']]
        elif 'exoplanets.org' in list(catalog_dict.keys()):
            target_data = target_data[catalog_dict['exoplanets.org']]
        else:
            target_data = target_data[0]
    else:
        target_data = target_data[0]

    # Strip spaces and non numeric or alphabetic characters and combine.
    name = re.sub(r'\W+', '', canonical_name)
    url = f'https://exo.mast.stsci.edu/exomast_planet.html?planet={name}'

    return target_data, url
