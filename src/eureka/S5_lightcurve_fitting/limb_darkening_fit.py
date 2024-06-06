#!/usr/bin/python
# -*- coding: latin-1 -*-
"""
A module to calculate limb darkening coefficients from a grid of model spectra
"""
import inspect
import warnings

import astropy.table as at
from astropy.utils.exceptions import AstropyWarning
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from svo_filters import svo
import bokeh.plotting as bkp
from bokeh.models import Range1d
from bokeh.models.widgets import Panel, Tabs

from . import utils
# from . import modelgrid

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)


def ld_profile(name='quadratic', latex=False, use_gen_ld='batman'):
    """Define the function to fit the limb darkening profile.

    Parameters
    ----------
    name : str; optional
        The name of the limb darkening profile function to use,
        including 'uniform', 'linear', 'quadratic', 'kipping2013',
        'squareroot', 'logarithmic', 'exponential', '3-parameter',
        and '4-parameter'. Defaults to 'quadratic'.
    latex : bool; optional
        Return the function as a LaTeX formatted string. Defaults to False.
    use_gen_ld : str; optional
        Which tool will be doing the limb darkening modelling from
        ['batman', 'exotic-ld']. Defaults to 'batman'.

    Returns
    -------
    function
        The corresponding function for the given profile.
        Returned if latex==False.
    OR str
        A string representation of the corresponding function
        for the given profile. Returned if latex==True.

    References
    ----------
    https://www.cfa.harvard.edu/~lkreidberg/batman/tutorial.html#limb-darkening-options
    https://exotic-ld.readthedocs.io/en/latest/views/quick_start.html
    """
    # Supported profiles a la BATMAN and/or POET
    names = ['uniform', 'linear', 'quadratic', 'kipping2013', 'squareroot',
             'logarithmic', 'exponential', '3-parameter', '4-parameter']

    # Generated profiles by exotic-ld
    exotic_ld_names = ['linear', 'quadratic', '3-parameter', '4-parameter']
    if use_gen_ld == 'exotic-ld':
        names = exotic_ld_names

    # Check that the profile is supported
    if name in names:

        if name == 'uniform':
            profile = uniform
        elif name == 'linear':
            profile = linear
        elif name == 'quadratic':
            profile = quadratic
        elif name == 'kipping2013':
            profile = kipping2013
        elif name == 'squareroot':
            profile = square_root
        elif name == 'logarithmic':
            profile = logarithmic
        elif name == 'exponential':
            profile = exponential
        elif name == '3-parameter':
            profile = three_parameter
        elif name == '4-parameter':
            profile = four_parameter

        if latex:
            profile = inspect.getsource(profile).replace('\n', '')
            profile = profile.replace('\\', '').split('return ')[1]

            for i, j in [('**', '^'), ('m', r'\mu'), (' ', ''), ('np.', '\\'),
                         ('0.5', '{0.5}'), ('1.5', '{1.5}')]:
                profile = profile.replace(i, j)

        return profile
    else:
        raise Exception(f"'{name}' is not a supported profile by "
                        f"'{use_gen_ld}'. Try {names}")


def uniform(m):
    """Uniform limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1.


def linear(m, c1):
    """Linear limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1. - c1*(1.-m)


def quadratic(m, c1, c2):
    """Quadratic limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1. - c1*(1.-m) - c2*(1.-m)**2


def kipping2013(m, c1, c2):
    """Reparameterized Quadratic (Kipping 2013) limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    u1 = 2*np.sqrt(c1)*c2
    u2 = np.sqrt(c1)*(1-2*c2)
    return 1. - u1*(1.-m) - u2*(1.-m)**2


def square_root(m, c1, c2):
    """Squareroot limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1. - c1*(1.-m) - c2*(1.-np.sqrt(m))


def logarithmic(m, c1, c2):
    """Logarithmic limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1. - c1*(1.-m) - c2*m*np.log(m)


def exponential(m, c1, c2):
    """Exponential limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1. - c1*(1.-m) - c2/(1.-np.e**m)


def three_parameter(m, c1, c2, c3):
    """3-parameter limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.
    c3 : float
        The third limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return 1. - c1*(1.-m) - c2*(1.-m**1.5) - c3*(1.-m**2)


def four_parameter(m, c1, c2, c3, c4):
    """4-parameter limb darkening.

    Parameters
    ----------
    m : ndarray, float
        The normalized radial coordinate.
    c1 : float
        The first limb darkening coefficient.
    c2 : float
        The second limb darkening coefficient.
    c3 : float
        The third limb darkening coefficient.
    c4 : float
        The fourth limb darkening coefficient.

    Returns
    -------
    ndarray, float
        The relative intensity of the star at each position m.
    """
    return (1. - c1*(1.-m**0.5) - c2*(1.-m)
            - c3*(1.-m**1.5) - c4*(1.-m**2))


class LDC:
    """A class to hold all the LDCs you want to run.

    Examples
    --------
    from exoctk.limb_darkening import limb_darkening_fit as lf
    from exoctk import modelgrid
    from svo_filters import Filter
    from pkg_resources import resource_filename
    fits_files = resource_filename('exoctk', 'data/core/modelgrid/')
    model_grid = modelgrid.ModelGrid(fits_files, resolution=700)
    ld = lf.LDC(model_grid)
    bp = Filter('WFC3_IR.G141', n_bins=5)
    ld.calculate(4000, 4.5, 0.0, 'quadratic', bandpass=bp)
    ld.calculate(4000, 4.5, 0.0, '4-parameter', bandpass=bp)
    ld.plot(show=True)
    """

    def __init__(self, model_grid):
        """Initialize an LDC object.

        Parameters
        ----------
        model_grid : exoctk.modelgrid.ModelGrid
            The grid of synthetic spectra from which the coefficients will
            be calculated.
        """
        # Set the model grid
        # if not isinstance(model_grid, modelgrid.ModelGrid):
        #     raise TypeError("'model_grid' must be a "
        #                     "exoctk.modelgrid.ModelGrid object.")

        self.model_grid = model_grid

        # Table for results
        columns = ['name', 'Teff', 'logg', 'FeH', 'profile', 'filter',
                   'coeffs', 'errors', 'wave', 'wave_min', 'wave_eff',
                   'wave_max', 'scaled_mu', 'raw_mu', 'mu_min', 'scaled_ld',
                   'raw_ld', 'ld_min', 'ldfunc', 'flux', 'bandpass', 'color']
        dtypes = ['|S20', float, float, float, '|S20', '|S20', object, object,
                  object, np.float16, np.float16, np.float16, object, object,
                  np.float16, object, object, np.float16, object, object,
                  object, '|S20']
        self.results = at.Table(names=columns, dtype=dtypes)

        self.ld_color = {'quadratic': 'blue', '4-parameter': 'red',
                         'exponential': 'green', 'linear': 'orange',
                         'squareroot': 'cyan', '3-parameter': 'magenta',
                         'logarithmic': 'pink', 'uniform': 'purple',
                         'kipping2013': 'brown'}

        self.count = 1

    @staticmethod
    def bootstrap_errors(mu_vals, func, coeffs, errors, n_samples=1000):
        """Bootstrapping LDC errors.

        Parameters
        ----------
        mu_vals : sequence
            The mu values.
        func : callable
            The LD profile function.
        coeffs : sequence
            The coefficients.
        errors : sequence
            The errors on each coeff.
        n_samples : int; optional
            The number of samples. Defaults to 1000.

        Returns
        -------
        tuple
            The lower and upper errors.
        """
        # Generate n_samples
        vals = []
        for n in range(n_samples):
            co = np.random.normal(coeffs, errors)
            vals.append(func(mu_vals, *co))

        # r = np.array(list(zip(*vals)))
        dn_err = np.min(np.asarray(vals), axis=0)
        up_err = np.max(np.asarray(vals), axis=0)

        return dn_err, up_err

    def calculate(self, Teff, logg, FeH, profile, mu_min=0.05, ld_min=0.01,
                  bandpass=None, name=None, color=None, **kwargs):
        """
        Calculates the limb darkening coefficients for a given synthetic
        spectrum. If the model grid does not contain a spectrum of the given
        parameters, the grid is interpolated to those parameters.

        Reference for limb-darkening laws:
        http://www.astro.ex.ac.uk/people/sing/David_Sing/Limb_Darkening.html

        Parameters
        ----------
        Teff : int
            The effective temperature of the model.
        logg : float
            The logarithm of the surface gravity.
        FeH : float
            The logarithm of the metallicity.
        profile : str
            The name of the limb darkening profile function to use,
            including 'uniform', 'linear', 'quadratic', 'squareroot',
            'logarithmic', 'exponential', and '4-parameter'.
        mu_min : float; optional
            The minimum mu value to consider. Defaults to 0.05.
        ld_min : float; optional
            The minimum limb darkening value to consider. Defaults to 0.01.
        bandpass : svo_filters.svo.Filter(); optional
            The photometric filter through which the limb darkening
            is to be calculated. Defaults to None.
        name : str; optional
            A name for the calculation. Defaults to None.
        color : str; optional
            A color for the plotted result. Defaults to None.
        **kwargs : dict
            Unused.
        """
        # Define the limb darkening profile function
        ldfunc = ld_profile(profile)

        if not ldfunc:
            raise ValueError("No such LD profile:", profile)

        # Get the grid point
        grid_point = self.model_grid.get(Teff, logg, FeH)

        # Retrieve the wavelength, flux, mu, and effective radius
        wave = grid_point.get('wave')
        flux = grid_point.get('flux')
        mu = grid_point.get('mu').squeeze()

        # Use tophat oif no bandpass
        if bandpass is None:
            units = self.model_grid.wave_units
            bandpass = svo.Filter('tophat', wave_min=np.min(wave)*units,
                                  wave_max=np.max(wave)*units)

        # Check if a bandpass is provided
        if not isinstance(bandpass, svo.Filter):
            raise TypeError("Invalid bandpass of type", type(bandpass))

        # # Make sure the bandpass has coverage
        # bp_min = bandpass.wave_min.value
        # bp_max = bandpass.wave_max.value
        # mg_min = self.model_grid.wave_rng[0].value
        # mg_max = self.model_grid.wave_rng[-1].value
        # if bp_min < mg_min or bp_max > mg_max:
        #     raise ValueError('Bandpass {} not covered by model grid of\
        #                       wavelength range {}'.format(bandpass.filterID,
        #                                                   self.model_grid
        #                                                       .wave_rng))

        # Apply the filter
        try:
            # Sometimes this returns a tuple
            flux, _ = bandpass.apply([wave, flux])
        except ValueError:
            # Sometimes it returns one value
            flux = bandpass.apply([wave, flux])

        # Make rsr curve 3 dimensions if there is only one
        # wavelength bin, then get wavelength only
        bp = bandpass.rsr
        if bp.ndim == 2:
            bp = bp[None, :]
        wave = bp[:, 0, :]

        # Calculate mean intensity vs. mu
        wave = wave[None, :] if wave.ndim == 1 else wave
        flux = flux[None, :] if flux.ndim == 2 else flux
        mean_i = np.nanmean(flux, axis=-1)
        mean_i[mean_i == 0] = np.nan

        # Calculate limb darkening, I[mu]/I[1] vs. mu
        ld = mean_i/mean_i[:, np.where(mu == max(mu))].squeeze(axis=-1)

        # Rescale mu values to make f(mu=0)=ld_min
        # for the case where spherical models extend beyond limb
        ld_avg = np.nanmean(ld, axis=0)
        muz = np.interp(ld_min, ld_avg, mu) if any(ld_avg < ld_min) else 0
        mu = (mu - muz) / (1 - muz)

        # Trim to useful mu range
        imu, = np.where(mu > mu_min)
        scaled_mu, scaled_ld = mu[imu], ld[:, imu]

        # Fit limb darkening coefficients for each wavelength bin
        for n, ldarr in enumerate(scaled_ld):

            # Fit polynomial to data
            coeffs, cov = curve_fit(ldfunc, scaled_mu, ldarr, method='lm')

            # Calculate errors from covariance matrix diagonal
            errs = np.sqrt(np.diag(cov))
            wave_eff = bandpass.centers[0, n].round(5)

            # Make a dictionary or the results
            result = {}

            # Check the count
            result['name'] = name or 'Calculation {}'.format(self.count)
            self.count += 1
            if len(bandpass.centers[0]) == len(scaled_ld) and name is None:
                result['name'] = (f'{round(bandpass.centers[0][n], 2)} '
                                  f'{self.model_grid.wave_units}')

            # Set a color if possible
            result['color'] = color or self.ld_color[profile]

            # Add the results
            result['Teff'] = Teff
            result['logg'] = logg
            result['FeH'] = FeH
            result['filter'] = bandpass.filterID
            result['raw_mu'] = mu
            result['raw_ld'] = ld[n]
            result['scaled_mu'] = scaled_mu
            result['scaled_ld'] = ldarr
            result['flux'] = flux[n]
            result['wave'] = wave[n]
            result['mu_min'] = mu_min
            result['bandpass'] = bandpass
            result['ldfunc'] = ldfunc
            result['coeffs'] = coeffs
            result['errors'] = errs
            result['profile'] = profile
            result['n_bins'] = bandpass.n_bins
            result['pixels_per_bin'] = bandpass.pixels_per_bin
            result['wave_min'] = wave[n, 0].round(5)
            result['wave_eff'] = wave_eff
            result['wave_max'] = wave[n, -1].round(5)

            # Add the coeffs
            for n, (coeff, err) in enumerate(zip(coeffs, errs)):
                cname = 'c{}'.format(n + 1)
                ename = 'e{}'.format(n + 1)
                result[cname] = coeff.round(3)
                result[ename] = err.round(3)

                # Add the coefficient column to the table if not present
                if cname not in self.results.colnames:
                    self.results[cname] = [np.nan]*len(self.results)
                    self.results[ename] = [np.nan]*len(self.results)

            # Add the new row to the table
            result = {i: j for i, j in result.items() if i in
                      self.results.colnames}
            self.results.add_row(result)

    def plot_tabs(self, show=False, **kwargs):
        """Plot the LDCs in a tabbed figure

        Parameters
        ----------
        show : bool; optional
            Show the figure. Defaults to False.
        **kwargs : dict
            Unused.

        Returns
        -------
        final : bokeh.models.widgets.Tabs; optional
            The final tabbed figure. Only returned if show==False.
        """
        # Change names to reflect ld profile
        old_names = self.results['name']
        for n, row in enumerate(self.results):
            self.results[n]['name'] = row['profile']

        # Draw a figure for each wavelength bin
        tabs = []
        for wav in np.unique(self.results['wave_eff']):

            # Plot it
            TOOLS = 'box_zoom, box_select, crosshair, reset, hover'
            fig = bkp.figure(tools=TOOLS, x_range=Range1d(0, 1),
                             y_range=Range1d(0, 1),
                             plot_width=800, plot_height=400)
            self.plot(wave_eff=wav, fig=fig)

            # Plot formatting
            fig.legend.location = 'bottom_right'
            fig.xaxis.axis_label = 'mu'
            fig.yaxis.axis_label = 'Intensity'

            tabs.append(Panel(child=fig, title=str(wav)))

        # Make the final tabbed figure
        final = Tabs(tabs=tabs)

        # Put the names back
        self.results['name'] = old_names

        if show:
            bkp.show(final)
        else:
            return final

    def plot(self, fig=None, show=False, **kwargs):
        """Plot the LDCs

        Parameters
        ----------
        fig : matplotlib.pyplot.figure, bokeh.plotting.figure; optional
            An existing figure to plot on. Defaults to None.
        show : bool; optional
            Show the figure. Defaults to False.
        **kwargs : dict
            Additional parameters to be passed into plotting
            and utils.filter_table().

        Returns
        -------
        fig : matplotlib.pyplot.figure, bokeh.plotting.figure; optional
            The resulting figure. Only returned if show==False.
        """
        # Separate plotting kwargs from parameter kwargs
        pwargs = {i: j for i, j in kwargs.items() if i in self.results.columns}
        kwargs = {i: j for i, j in kwargs.items() if i not in pwargs.keys()}

        # Filter the table by given kwargs
        table = utils.filter_table(self.results, **pwargs)

        for row in table:

            # Set color and label for plot
            color = row['color']
            label = row['name']

            # Generate smooth curve
            ldfunc = row['ldfunc']
            mu_vals = np.linspace(0, 1, 1000)
            ld_vals = ldfunc(mu_vals, *row['coeffs'])

            # Generate smooth errors
            dn_err, up_err = self.bootstrap_errors(mu_vals, ldfunc,
                                                   row['coeffs'],
                                                   row['errors'])

            # Matplotlib fig by default
            if fig is None:
                fig = bkp.figure()

            # Add fits to matplotlib
            if isinstance(fig, matplotlib.figure.Figure):

                # Make axes
                ax = fig.add_subplot(111)

                # Plot the fitted points
                ax.errorbar(row['raw_mu'], row['raw_ld'], c='k',
                            ls='None', marker='o', markeredgecolor='k',
                            markerfacecolor='None')

                # Plot the mu cutoff
                ax.axvline(row['mu_min'], color='0.5', ls=':')

                # Draw the curve and error
                ax.plot(mu_vals, ld_vals, color=color, label=label, **kwargs)
                ax.fill_between(mu_vals, dn_err, up_err, color=color,
                                alpha=0.1)
                ax.set_ylim(0, 1)
                ax.set_xlim(0, 1)

            # Or to bokeh!
            else:

                # Set the plot elements
                fig.x_range = Range1d(0, 1)
                fig.y_range = Range1d(0, 1)
                fig.xaxis.axis_label = 'mu'
                fig.yaxis.axis_label = 'Normalized Intensity'
                fig.legend.location = "bottom_right"

                # Plot the fitted points
                fig.circle(row['raw_mu'], row['raw_ld'], fill_color='black')

                # Plot the mu cutoff
                fig.line([row['mu_min']]*2, [0, 1], legend='cutoff',
                         line_color='#6b6ecf', line_dash='dotted')

                # Draw the curve and error
                fig.line(mu_vals, ld_vals, line_color=color, legend=label,
                         **kwargs)
                vals = np.append(mu_vals, mu_vals[::-1])
                evals = np.append(dn_err, up_err[::-1])
                fig.patch(vals, evals, color=color, fill_alpha=0.2,
                          line_alpha=0)

        if show:
            if isinstance(fig, matplotlib.figure.Figure):
                plt.xlabel(r'$\mu$')
                plt.ylabel(r'$I(\mu)/I(\mu = 1)$')
                plt.legend(loc=0, frameon=False)
                plt.show()
            else:
                bkp.show(fig)

        else:
            return fig
