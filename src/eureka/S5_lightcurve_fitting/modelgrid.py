# !/usr/bin/python
# -*- coding: latin-1 -*-
"""
A module for creating and managing grids of model spectra
"""
from functools import partial
from glob import glob
import multiprocessing
import os
import pickle
from pkg_resources import resource_filename
import time
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import astropy.table as at
import astropy.units as q
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

from . import utils

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)


class ModelGrid(object):
    """
    Creates a ModelGrid object which contains a multi-parameter
    grid of model spectra and its references

    Parameters
    ----------
    path : str
        The path to the directory of FITS files used to create the ModelGrid
    refs : list, str
        The references for the data contained in the ModelGrid
    teff_rng : tuple
        The range of effective temperatures [K]
    logg_rng : tuple
        The range of surface gravities [dex]
    FeH_rng : tuple
        The range of metalicities [dex]
    wave_rng : array-like
        The wavelength range of the models [um]
    n_bins : int
        The number of bins for the ModelGrid wavelength array
    data : astropy.table.Table
        The table of parameters for the ModelGrid
    inv_file : str
        An inventory file to more quickly load the database
    """

    def __init__(self, model_directory, bibcode='2013A & A...553A...6H',
                 names={'Teff': 'PHXTEFF', 'logg': 'PHXLOGG',
                        'FeH': 'PHXM_H', 'mass': 'PHXMASS', 'Lbol': 'PHXLUM'},
                 resolution=None, wave_units=q.um, **kwargs):
        """
        Initializes the model grid by creating a table with a column
        for each parameter and ingests the spectra

        Parameters
        ----------
        model_directory : str
            The path to the directory of FITS files of spectra,
            which may include a filename with a wildcard caharacter
        bibcode : str, array-like; optional
            The bibcode or list of bibcodes for this data set.
            Defaults to '2013A & A...553A...6H'.
        names : dict; optional
            A dictionary to rename the table columns. The Phoenix
            model keywords are given as an example. Defaults to
            {'Teff': 'PHXTEFF', 'logg': 'PHXLOGG', 'FeH': 'PHXM_H',
             'mass': 'PHXMASS', 'Lbol': 'PHXLUM'}.
        resolution : int; optional
            The desired wavelength resolution (lambda/d_lambda)
            of the grid spectra. Defaults to None.
        wave_units : astropy.units.quantity; optional
            The wavelength units. Defaults to astropy.units.um.
        **kwargs : dict
            Additional arguments to pass to self.customize().
        """
        # Make sure we can use glob if a directory
        # is given without a wildcard
        if '*' not in model_directory:
            model_directory = os.path.join(model_directory, '*')

        # Check for a precomputed pickle of this ModelGrid
        model_grid = None
        if model_directory.endswith(os.sep+'*'):
            # Location of model_grid pickle
            file = model_directory.replace('*', 'model_grid.p')

            if os.path.isfile(file):
                model_grid = pickle.load(open(file, 'rb'))

        # Instantiate the precomputed model grid
        if model_grid is not None:

            for k, v in model_grid.items():
                setattr(self, k, v)

            self.flux_file = os.path.join(self.path, 'model_grid_flux.hdf5')
            self.flux = None
            self.wavelength = None
            self.mu = None

            del model_grid

        # Or compute it from scratch
        else:

            # Print update...
            if model_directory.endswith(os.sep+'*'):

                print("Indexing models...")

            # Create some attributes
            self.path = os.path.dirname(model_directory)+os.sep
            self.refs = None
            self.wave_rng = (0*q.um, 40*q.um)
            self.flux_file = os.path.join(self.path, 'model_grid_flux.hdf5')
            self.flux = None
            self.wavelength = None
            self.mu = None

            # Save the refs to a References() object
            if bibcode:
                if isinstance(bibcode, (list, tuple)):
                    pass
                elif bibcode and isinstance(bibcode, str):
                    bibcode = [bibcode]
                else:
                    pass

                self.refs = bibcode
                # _check_for_ref_object()

            # Get list of spectral intensity files
            files = glob(model_directory)
            filenames = []
            if not files:
                print('No files match', model_directory, '.')
                return

            # Parse the FITS headers
            vals, dtypes = [], []
            for f in files:
                if f.endswith('.fits'):
                    try:
                        header = fits.getheader(f)
                        keys = np.array(header.cards).T[0]
                        dtypes = [type(i[1]) for i in header.cards]
                        vals.append([header.get(k) for k in keys])
                        filenames.append(f.split(os.sep)[-1])
                    except:
                        # FINDME: Need to only catch the expected exception.
                        print(f, 'could not be read into the model grid.')

            # Fix data types, trim extraneous values, and make the table
            dtypes = [str if d == bool else d for d in dtypes]
            vals = [v[: len(dtypes)] for v in vals]
            table = at.Table(np.array(vals), names=keys, dtype=dtypes)

            # Add the filenames as a column
            table['filename'] = filenames

            # Rename any columns
            for new, old in names.items():
                try:
                    table.rename_column(old, new)
                except:
                    # FINDME: Need to only catch the expected exception.
                    print('No column named', old)

            # Remove columns where the values are all the same
            # and store value as attribute instead
            for n in table.colnames:
                val = table[n][0]
                exc = n not in ['Teff', 'logg', 'FeH']
                if list(table[n]).count(val) == len(table[n]) and exc:
                    setattr(self, n, val)
                    table.remove_column(n)

            # Store the table in the data attribute
            self.data = table

            # Store the parameter ranges
            self.Teff_vals = np.asarray(np.unique(table['Teff']))
            self.logg_vals = np.asarray(np.unique(table['logg']))
            self.FeH_vals = np.asarray(np.unique(table['FeH']))

            # Write an inventory file to this directory for future table loads
            if model_directory.endswith(os.sep+'*'):
                self.file = file
                try:
                    pickle.dump(self.__dict__, open(self.file, 'wb'))
                except IOError:
                    print('Could not write model grid to', self.file)

        # Print something
        print(len(self.data), 'models loaded from', self.path)

        # In case no filter is used
        self.n_bins = 1

        # Set the wavelength_units
        self.wave_units = q.AA
        if wave_units:
            self.set_units(wave_units)
        else:
            self.const = 1

        # Save the desired resolution
        self.resolution = resolution

        # Customize from the get-go
        if kwargs:
            self.customize(**kwargs)

    def export(self, filepath, **kwargs):
        """Export the model with the given parameters to a FITS file
        at the given filepath.

        Parameters
        ----------
        filepath : str
            The path to the target FITS file.
        **kwargs : dict
            Additional parameters to pass to self.get().
        """
        if not filepath.endswith('.fits'):
            raise IOError("Target file must have a .fits extension.")

        # Get the model
        model = self.get(**kwargs)

        # Get a dummy FITS file
        ffile = resource_filename('ExoCTK',
                                  f'data{os.sep}core{os.sep}' +
                                  'ModelGrid_tmp.fits')
        hdu = fits.open(ffile)

        # Replace the data
        hdu[0].data = model['flux']
        hdu[1].data = model['mu']
        hdu[0].header['PHXTEFF'] = model['Teff']
        hdu[0].header['PHXLOGG'] = model['logg']
        hdu[0].header['PHXM_H'] = model['FeH']

        # Update the wavelength
        wave = model['wave']
        hdu[0].header['CRVAL1'] = min(wave)
        hdu[0].header['CDELT1'] = np.mean(np.diff(wave))
        hdu[0].header['CUNIT1'] = 'Micron'

        # Write the file
        hdu.writeto(filepath)

    def get(self, Teff, logg, FeH, resolution=None, interp=True):
        """Retrieve the wavelength, flux, and effective radius
        for the spectrum of the given parameters

        Parameters
        ----------
        Teff : int
            The effective temperature (K)
        logg : float
            The logarithm of the surface gravity (dex)
        FeH : float
            The logarithm of the ratio of the metallicity
            and solar metallicity (dex)
        resolution : int; optional
            The desired wavelength resolution (lambda/d_lambda).
            Defaults to None.
        interp : bool; optional
            Interpolate the model if possible. Defaults to True.

        Returns
        -------
        spec_dict : dict
            A dictionary of arrays of the wavelength, flux, and
            mu values and the effective radius for the given model
        """
        # See if the model with the desired parameters is witin the grid
        in_grid = all([(Teff >= min(self.Teff_vals)) &
                       (Teff <= max(self.Teff_vals)) &
                       (logg >= min(self.logg_vals)) &
                       (logg <= max(self.logg_vals)) &
                       (FeH >= min(self.FeH_vals)) &
                       (FeH <= max(self.FeH_vals))])

        if in_grid:

            # See if the model with the desired parameters is a true grid point
            on_grid = (self.data[[(self.data['Teff'] == Teff) &
                                  (self.data['logg'] == logg) &
                                  (self.data['FeH'] == FeH)]]
                       in self.data)

            # Grab the data if the point is on the grid
            if on_grid:

                # Get the row index and filepath
                row, = np.where((self.data['Teff'] == Teff) &
                                (self.data['logg'] == logg) &
                                (self.data['FeH'] == FeH))[0]

                filepath = self.path+str(self.data[row]['filename'])

                # Get the flux, mu, and abundance arrays
                raw_flux = fits.getdata(filepath, 0)
                mu = fits.getdata(filepath, 1)
                # abund = fits.getdata(filepath, 2)

                # Construct full wavelength scale and convert to microns
                if self.CRVAL1 == '-':
                    # Try to get data from WAVELENGTH extension...
                    dat = fits.getdata(filepath, ext=-1)
                    raw_wave = np.array(dat).squeeze()
                else:
                    # ...or try to generate it
                    b = self.CDELT1*np.arange(len(raw_flux[0]))
                    raw_wave = np.array(self.CRVAL1+b).squeeze()

                # Convert from A to desired units
                raw_wave *= self.const

                # Trim the wavelength and flux arrays
                idx, = np.where(np.logical_and(raw_wave*self.wave_units >=
                                               self.wave_rng[0],
                                               raw_wave*self.wave_units <=
                                               self.wave_rng[1]))
                flux = raw_flux[:, idx]
                wave = raw_wave[idx]

                # Bin the spectrum if necessary
                if resolution is not None or self.resolution is not None:

                    # Calculate zoom
                    z = utils.calc_zoom(resolution or self.resolution, wave)
                    wave = zoom(wave, z)
                    flux = zoom(flux, (1, z))

                # Make a dictionary of parameters
                # This should really be a core.Spectrum() object!
                row_data = self.data[row].as_void()
                spec_dict = dict(zip(self.data.colnames, row_data))
                spec_dict['wave'] = wave
                spec_dict['flux'] = flux
                spec_dict['mu'] = mu

            # If not on the grid, interpolate to it
            else:
                # Call grid_interp method
                if interp:
                    spec_dict = self.grid_interp(Teff, logg, FeH)
                else:
                    return

            return spec_dict

        else:
            print('Teff: ', Teff, ' logg: ', logg, ' FeH: ', FeH,
                  ' model not in grid.')
            return

    def grid_interp(self, Teff, logg, FeH, plot=False):
        """Interpolate the grid to the desired parameters

        Parameters
        ----------
        Teff : int
            The effective temperature (K)
        logg : float
            The logarithm of the surface gravity (dex)
        FeH : float
            The logarithm of the ratio of the metallicity
            and solar metallicity (dex)
        plot : bool; optional
            Plot the interpolated spectrum along
            with the 8 neighboring grid spectra. Defaults to False.

        Returns
        -------
        grid_point : dict
            A dictionary of arrays of the wavelength, flux, and
            mu values and the effective radius for the given model
        """
        # Load the fluxes
        if self.flux is None:
            self.load_flux()

        # Get the flux array
        flux = self.flux.copy()

        # Get the interpolable parameters
        params, values = [], []
        for p, v in zip([self.Teff_vals, self.logg_vals, self.FeH_vals],
                        [Teff, logg, FeH]):
            if len(p) > 1:
                params.append(p)
                values.append(v)
        values = np.asarray(values)
        label = '{}/{}/{}'.format(Teff, logg, FeH)

        try:
            # Interpolate flux values at each wavelength
            # using a pool for multiple processes
            print('Interpolating grid point [{}]...'.format(label))
            processes = 8
            mu_index = range(flux.shape[-2])
            start = time.time()
            pool = multiprocessing.Pool(processes)
            func = partial(utils.interp_flux, flux=flux, params=params,
                           values=values)
            new_flux, generators = zip(*pool.map(func, mu_index))
            pool.close()
            pool.join()

            # Clean up and time of execution
            new_flux = np.asarray(new_flux)
            generators = np.asarray(generators)
            print('Run time in seconds: ', time.time()-start)

            # Interpolate mu value
            interp_mu = RegularGridInterpolator(params, self.mu)
            mu = interp_mu(np.array(values)).squeeze()

            # Make a dictionary to return
            grid_point = {'Teff': Teff, 'logg': logg, 'FeH': FeH,
                          'mu': mu, 'flux': new_flux, 'wave': self.wavelength,
                          'generators': generators}

            return grid_point

        except IOError:
            print('Grid too sparse. Could not interpolate.')
            return

    def load_flux(self, reset=False):
        """Retrieve the flux arrays for all models
        and load into the ModelGrid.array attribute
        with shape (Teff, logg, FeH, mu, wavelength)

        Parameters
        ----------
        reset : bool; optional
            Delete the old file and clear the flux attribute.
            Defaults to False.
        """
        if reset:
            # Delete the old file and clear the flux attribute
            if os.path.isfile(self.flux_file):
                os.remove(self.flux_file)
            self.flux = None

        if self.flux is None:
            print('Loading flux into table...')

            if os.path.isfile(self.flux_file):

                # Load the flux from the HDF5 file
                f = h5py.File(self.flux_file, "r")
                self.flux = f['flux'][:]
                self.mu = f['mu'][:]
                self.wavelength = f['wave'][:]
                f.close()

            else:

                # Get array dimensions
                T, G, M = self.Teff_vals, self.logg_vals, self.FeH_vals
                shp = [len(T), len(G), len(M)]
                n, N = 1, np.prod(shp)

                # Iterate through rows
                for nt, teff in enumerate(T):
                    for ng, logg in enumerate(G):
                        for nm, feh in enumerate(M):

                            try:

                                # Retrieve flux using the `get()` method
                                d = self.get(teff, logg, feh, interp=False)

                                if d:

                                    # Make sure arrays exist
                                    if self.flux is None:
                                        new_shp = shp+list(d['flux'].shape)
                                        self.flux = np.zeros(new_shp)
                                    if self.mu is None:
                                        new_shp = shp+list(d['mu'].shape)
                                        self.mu = np.zeros(new_shp)

                                    # Add data to respective arrays
                                    self.flux[nt, ng, nm] = d['flux']
                                    self.mu[nt, ng, nm] = d['mu'].squeeze()

                                    # Get the wavelength array
                                    if self.wavelength is None:
                                        self.wavelength = d['wave']

                                    # Garbage collection
                                    del d

                                    # Print update
                                    n += 1
                                    msg = "{: .2f}% complete.".format(n*100./N)
                                    print(msg, end='\r')

                            except IOError:
                                # No model computed so reduce total
                                N -= 1

                # Load the flux into an HDF5 file
                f = h5py.File(self.flux_file, "w")
                f.create_dataset('flux', data=self.flux)
                f.create_dataset('mu', data=self.mu)
                f.create_dataset('wave', data=self.wavelength)
                f.close()
                # del dset
                print("100.00 percent complete!", end='\n')

        else:
            print('Data already loaded.')

    def customize(self, Teff_rng=(2300, 8000), logg_rng=(0, 6),
                  FeH_rng=(-2, 1), wave_rng=(0*q.um, 40*q.um), n_bins=''):
        """Trims the model grid by the given ranges in effective temperature,
        surface gravity, and metallicity. Also sets the wavelength range
        and number of bins for retrieved model spectra.

        Parameters
        ----------
        Teff_rng : array-like; optional
            The lower and upper inclusive bounds for the effective
            temperature (K). Defaults to (2300, 8000).
        logg_rng : array-like; optional
            The lower and upper inclusive bounds for the logarithm of the
            surface gravity (dex). Defaults to (0, 6).
        FeH_rng : array-like; optional
            The lower and upper inclusive bounds for the logarithm of the
            ratio of the metallicity and solar metallicity (dex). Defaults
            to (-2, 1).
        wave_rng : array-like; optional
            The lower and upper inclusive bounds for the wavelength (microns).
            Defaults to (0*q.um, 40*q.um).
        n_bins : int; optional
            The number of bins for the wavelength axis. Defaults to ''.
        """
        # Make a copy of the grid
        grid = self.data.copy()
        self.wave_rng = wave_rng
        self.n_bins = n_bins or self.n_bins

        # Filter grid by given parameters
        self.data = grid[[(grid['Teff'] >= Teff_rng[0])
                         & (grid['Teff'] <= Teff_rng[1])
                         & (grid['logg'] >= logg_rng[0])
                         & (grid['logg'] <= logg_rng[1])
                         & (grid['FeH'] >= FeH_rng[0])
                         & (grid['FeH'] <= FeH_rng[1])]]

        # Print a summary of the returned grid
        print('{}/{}'.format(len(self.data), len(grid)),
              'spectra in parameter range',
              'Teff: ', Teff_rng, ', logg: ', logg_rng,
              ', FeH: ', FeH_rng, ', wavelength: ', wave_rng)

        # Do nothing if he cut leaves the grid empty
        if len(self.data) == 0:
            self.data = grid
            print('The given param ranges would leave 0 models in the grid.')
            print('The model grid has not been updated. Please try again.')
            return

        # Update the wavelength and flux attributes
        if isinstance(self.wavelength, np.ndarray):
            w = self.wavelength
            W_idx, = np.where((w >= wave_rng[0]) & (w <= wave_rng[1]))
            T_idx, = np.where((self.Teff_vals >= Teff_rng[0]) &
                              (self.Teff_vals <= Teff_rng[1]))
            G_idx, = np.where((self.logg_vals >= logg_rng[0]) &
                              (self.logg_vals <= logg_rng[1]))
            M_idx, = np.where((self.FeH_vals >= FeH_rng[0]) &
                              (self.FeH_vals <= FeH_rng[1]))

            # Trim arrays
            self.wavelength = w[W_idx]
            self.flux = self.flux[T_idx[0]: T_idx[-1]+1,
                                  G_idx[0]: G_idx[-1]+1,
                                  M_idx[0]: M_idx[-1]+1,
                                  :,
                                  W_idx[0]: W_idx[-1]+1]
            self.mu = self.mu[T_idx[0]: T_idx[-1]+1,
                              G_idx[0]: G_idx[-1]+1,
                              M_idx[0]: M_idx[-1]+1]

        # Update the parameter attributes
        self.Teff_vals = np.unique(self.data['Teff'])
        self.logg_vals = np.unique(self.data['logg'])
        self.FeH_vals = np.unique(self.data['FeH'])

        # Reload the flux array with the new grid parameters
        self.load_flux(reset=True)

        # Clear the grid copy from memory
        del grid

    def info(self):
        """Print a table of info about the current ModelGrid"""
        # Get the info from the class
        tp = (int, bytes, bool, str, float, tuple, list, np.ndarray)
        info = [[k, str(v)] for k, v in vars(self).items()
                if isinstance(v, tp)]

        # Make the table
        table = at.Table(np.asarray(info).reshape(len(info), 2),
                         names=['Attributes', 'Values'])

        # Sort and print
        table.sort('Attributes')
        table.pprint(max_width=-1, align=['>', '<'])

    def reset(self):
        """Reset the current grid to the original state"""
        file = os.path.join(self.path+'model_grid_flux.hdf5')

        if os.path.isfile(file):
            os.remove(file)

        self.__init__(self.path)

    def set_units(self, wave_units=q.um):
        """Set the wavelength and flux units

        Parameters
        ----------
        wave_units : str, astropy.units.core.PrefixUnit/CompositeUnit; optional
            The wavelength units. Defaults to astropy.units.um.
        """
        # Set wavelength units
        old_unit = self.wave_units
        self.wave_units = q.Unit(wave_units)

        # Update the wavelength
        self.const = (old_unit/self.wave_units).decompose()._scale
