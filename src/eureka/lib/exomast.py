import itertools
import os
import re
import requests
import urllib


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
    target_url = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/{}/properties/".format(encode_target_name)

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

    # Create params dict for url parsing. Easier than trying to format yourself.
    params = {"name":target_name}

    r = requests.get(target_url, params=params)
    planetnames = r.json()
    canonical_name = planetnames['canonicalName']

    return canonical_name

def get_target_data(target_name):
    """
    Send request to exomast restful api for target information.
    Parameters
    ----------
    target_name : string
        The name of the target transit
    Returns
    -------
    target_data: json:
        json object with target data.
    """

    canonical_name = get_canonical_name(target_name)

    target_url = build_target_url(canonical_name)

    r = requests.get(target_url)

    if r.status_code == 200:
        target_data = r.json()
    else:
        raise Exception('Whoops, no data for this target!')

    # Some targets have multiple catalogs
    # nexsci is the first choice.
    if len(target_data) > 1:
        # Get catalog names from exomast and make then the keys of a dictionary
        # and the values are its position in the json object.
        catalog_dict = {data['catalog_name']: index for index, data in enumerate(target_data)}

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
    url = 'https://exo.mast.stsci.edu/exomast_planet.html?planet={}'.format(re.sub(r'\W+', '', canonical_name))

    return target_data, url
