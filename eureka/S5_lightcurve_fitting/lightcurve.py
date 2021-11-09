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
    def __init__(self, time, flux, unc=None, parameters=None, time_units='BJD', name='My Light Curve'):
        """
        A class to store the actual light curve

        Parameters
        ----------
        time: sequence
            The time axis in days, [MJD or BJD]
        flux: sequence
            The flux in electrons (not ADU)
        unc: sequence
            The uncertainty on the flux
        parameters: str, object (optional)
            The orbital parameters of the star/planet system,
            may be a path to a JSON file or a parameter object
        time_units: str
            The time units
        name: str
            A name for the object
        """
        # Initialize the model
        super().__init__()

        # Check data
        if len(time) != len(flux):
            raise ValueError('Time and flux axes must be the same length.')

        # Set the data arrays
        if unc is not None:
            if len(unc) != len(time):
                raise ValueError('Time and unc axes must be the same length.')

            self.unc = unc

        else:
            self.unc = np.array([np.nan]*len(self.time))

        # Set the time and flux axes
        self.time = time
        self.flux = flux

        # Set the units
        self.time_units = time_units
        self.name = name

        # Place to save the fit results
        self.results = []

    def fit(self, model, fitter='lsq', **kwargs):
        """Fit the model to the lightcurve

        Parameters
        ----------
        model: ExoCTK.lightcurve_fitter.models.Model
            The model to fit to the data
        fitter: str
            The name of the fitter to use
        """
        # Empty default fit
        fit_model = None

        model.time = self.time
        # Make sure the model is a CompositeModel
        if not isinstance(model, m.CompositeModel):
            model = m.CompositeModel([model])
            model.time = self.time
        # else:
        #     for n in range(len(model.components)):
        #         model.components[n].time = self.time

        if fitter == 'lmfit':

            # Run the fit
            fit_model = f.lmfitter(self.time, self.flux, model, self.unc, **kwargs)

        elif fitter == 'demc':

            # Run the fit
            fit_model = f.demcfitter(self.time, self.flux, model, self.unc, **kwargs)

        elif fitter == 'lsq':

            # Run the fit
            fit_model = f.lsqfitter(self, model, **kwargs)
            #fit_model = f.lsqfitter(self.time, self.flux, model, self.unc, **kwargs)

        else:
            raise ValueError("{} is not a valid fitter.".format(fitter))

        # Store it
        if fit_model is not None:
            self.results.append(fit_model)

    def plot(self, fits=True, draw=True):
        """Plot the light curve with all available fits

        Parameters
        ----------
        fits: bool
            Plot the fit models
        draw: bool
            Show the figure, else return it

        Returns
        -------
        bokeh.plotting.figure
            The figure
        """
        # Make the figure
        fig = plt.figure(figsize=(8,6))
        # Draw the data
        ax = fig.add_subplot(111)
        ax.errorbar(self.time, self.flux, self.unc, fmt='.', color=next(COLORS), zorder=0)
        # Draw best-fit model
        if fits and len(self.results) > 0:
            for model in self.results:
                model.plot(self.time, ax=ax, color=next(COLORS), zorder=1)

        # fig = figure(width=800, height=400)
        #
        # # Draw the data
        # fig.circle(self.time, self.flux, legend=self.name)
        #
        # # Plot fit models
        # if fits and len(self.results) > 0:
        #     for model in self.results:
        #         model.plot(self.time, fig=fig, color=next(COLORS))
        #
        # Format axes
        ax.set_xlabel(str(self.time_units))
        ax.set_ylabel('Flux')
        fig.tight_layout()

        # Draw or return
        if draw:
            fig.show()
        else:
            return

    def reset(self):
        """Reset the results"""
        self.results = []
