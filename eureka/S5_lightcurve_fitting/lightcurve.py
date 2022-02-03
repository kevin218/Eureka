"""Base and child classes to handle light curve fitting

Author: Joe Filippazzo
Email: jfilippazzo@stsci.edu
"""
import numpy as np
import pandas as pd
#from bokeh.plotting import figure, show
import matplotlib.pyplot as plt

from . import models as m
from . import fitters as f
from .utils import COLORS

#FINDME: Keep reload statements for easy testing
from importlib import reload
reload(m)
reload(f)

class LightCurveFitter:
    def __init__(self, time, flux, model):
        """Fit the model to the flux cube

        Parameters
        ----------
        time:
            1D or 2D time axes
        flux:
            2D flux
        """
        self.flux = np.ones(100)
        self.time = np.arange(100)
        self.results = pd.DataFrame(names=('fit_number', 'wavelength', 'P',
                                           'Tc', 'a/Rs', 'b', 'd', 'ldcs',
                                           'e', 'w', 'model_name', 'chi2'))

    def run(self):
        """Run the model fits"""
        pass

    # Method to return sliced results table
    def master_slicer(self, value, param_name='wavelength'):
        return self.results.iloc[self.results[param_name] == value]


class LightCurve(m.Model):
    def __init__(self, time, flux, channel, nchannel, log, unc=None, parameters=None, time_units='BJD', name='My Light Curve'):
        """
        A class to store the actual light curve

        Parameters
        ----------
        time: sequence
            The time axis in days, [MJD or BJD]
        flux: sequence
            The flux in electrons (not ADU)
        channel: int
            The channel number.
        nChannel: int
            The total number of channels.
        log: logedit.Logedit
            The open log in which notes from this step can be added.
        unc: sequence
            The uncertainty on the flux
        parameters: str, object (optional)
            The orbital parameters of the star/planet system,
            may be a path to a JSON file or a parameter object
        time_units: str
            The time units
        name: str
            A name for the object

        Returns
        -------
        None

        Notes
        -----

        History:
        - Dec 29, 2021 Taylor Bell
            Allowing for a constant uncertainty to be input with just a float.
            Added a channel number.
        """
        # Initialize the model
        super().__init__()

        # Check data
        if len(time) != len(flux):
            raise ValueError('Time and flux axes must be the same length.')

        # Set the data arrays
        if unc is not None:
            if type(unc) == float or type(unc) == np.float64:
                log.writelog('Warning: Only one uncertainty input, assuming constant uncertainty.')
            elif len(unc) != len(time):
                raise ValueError('Time and unc axes must be the same length.')

            self.unc = unc

        else:
            self.unc = np.array([np.nan]*len(time))

        # Set the time and flux axes
        self.time = time
        self.flux = flux

        # Set the units
        self.time_units = time_units
        self.name = name

        # Place to save the fit results
        self.results = []

        self.channel = channel
        self.nchannel = nchannel
        self.log = log

        self.color = next(COLORS)

        return

    def fit(self, model, meta, log, fitter='lsq', **kwargs):
        """Fit the model to the lightcurve

        Parameters
        ----------
        model: eureka.S5_lightcurve_fitting.models.CompositeModel
            The model to fit to the data
        meta: MetaClass
            The metadata object
        log: logedit.Logedit
            The open log in which notes from this step can be added.
        fitter: str
            The name of the fitter to use
        **kwargs:
            Arbitrary keyword arguments.

        Returns
        -------
        None

        Notes
        -----

        History:
        - Dec 29, 2021 Taylor Bell
            Updated documentation and reduced repeated code
        """
        # Empty default fit
        fit_model = None

        model.time = self.time
        # Make sure the model is a CompositeModel
        if not isinstance(model, m.CompositeModel):
            model = m.CompositeModel([model])
            model.time = self.time

        if fitter == 'lmfit':
            self.fitter_func = f.lmfitter
        elif fitter == 'demc':
            self.fitter_func = f.demcfitter
        elif fitter == 'lsq':
            self.fitter_func = f.lsqfitter
        elif fitter == 'emcee':
            self.fitter_func = f.emceefitter
        elif fitter == 'dynesty':
            self.fitter_func = f.dynestyfitter
        else:
            raise ValueError("{} is not a valid fitter.".format(fitter))

        # Run the fit
        fit_model = self.fitter_func(self, model, meta, log, **kwargs)

        # Store it
        if fit_model is not None:
            self.results.append(fit_model)

        return

    def plot(self, meta, fits=True):
        """Plot the light curve with all available fits

        Parameters
        ----------
        fits: bool
            Plot the fit models
        draw: bool
            Show the figure, else return it

        Returns
        -------
        None
        """
        # Make the figure
        fig = plt.figure(int('50{}'.format(str(self.channel).zfill(len(str(self.nchannel))))), figsize=(8,6))
        fig.clf()
        # Draw the data
        ax = fig.gca()
        ax.errorbar(self.time, self.flux, self.unc, fmt='.', color='w', ecolor=self.color, mec=self.color, zorder=0)
        # Draw best-fit model
        ls = ['-', '--', ':', '-.']
        if fits and len(self.results) > 0:
            for i, model in enumerate(self.results):
                model.plot(self.time, ax=ax, color=str(0.3+0.05*i), lw=2, ls=ls[i%4], zorder=np.inf)

        # Format axes
        ax.set_title(f'{meta.eventlabel} - Channel {self.channel}')
        ax.set_xlabel(str(self.time_units), size=14)
        ax.set_ylabel('Normalized Flux', size=14)
        ax.legend(loc='best')
        fig.tight_layout()

        fname = 'figs/fig50{}_all_fits.png'.format(str(self.channel).zfill(len(str(self.nchannel))))
        fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if meta.hide_plots:
            plt.close()
        else:
            plt.pause(0.2)

        return

    def reset(self):
        """Reset the results"""
        self.results = []

        return
