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
from .utils import COLORS, color_gen

#FINDME: Keep reload statements for easy testing
from importlib import reload
from copy import deepcopy
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
    def __init__(self, time, flux, channel, nchannel, log, longparamlist, unc=None, parameters=None, time_units='BJD', name='My Light Curve', share=False):
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
        share: bool
            Whether the fit shares parameters between spectral channels

        Returns
        -------
        None

        Notes
        -----

        History:
        - Dec 29, 2021 Taylor Bell
            Allowing for a constant uncertainty to be input with just a float.
            Added a channel number.
        - Jan. 15, 2022 Megan Mansfield
            Added ability to share fit between all channels
        """
        # Initialize the model
        super().__init__()

        self.name = name
        self.share = share
        self.channel = channel
        self.nchannel = nchannel
        if self.share:
            self.nchannel_fitted = self.nchannel
            self.fitted_channels = np.arange(self.nchannel)
        else:
            self.nchannel_fitted = 1
            self.fitted_channels = np.array([self.channel])

        # Check data
        if len(time)*self.nchannel_fitted != len(flux):
            raise ValueError('Time and flux axes must be the same length.')

        # Set the time and flux axes
        self.flux = flux
        self.time = time
        # Set the units
        self.time_units = time_units

        # Set the data arrays
        if unc is not None:
            if type(unc) == float or type(unc) == np.float64:
                log.writelog('Warning: Only one uncertainty input, assuming constant uncertainty.')
            elif len(time)*self.nchannel_fitted != len(unc):
                raise ValueError('Time and unc axes must be the same length.')

            self.unc = unc
        else:
            self.unc = np.array([np.nan]*len(self.time))

        # Place to save the fit results
        self.results = []

        self.longparamlist = longparamlist

        self.log = log

        self.colors = np.array([next(COLORS) for i in range(self.nchannel_fitted)])

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
        for i, channel in enumerate(self.fitted_channels):
            flux = self.flux
            if "unc_fit" in self.__dict__.keys():
                unc = deepcopy(self.unc_fit)
            else:
                unc = deepcopy(self.unc)
            if self.share:
                flux = flux[channel*len(self.time):(channel+1)*len(self.time)]
                unc = unc[channel*len(self.time):(channel+1)*len(self.time)]
            
            fig = plt.figure(int('54{}'.format(str(channel).zfill(len(str(self.nchannel))))), figsize=(8,6))
            fig.clf()
            # Draw the data
            ax = fig.gca()
            ax.errorbar(self.time, flux, unc, fmt='.', color=self.colors[i], zorder=0)
            
            # Make a new color generator for the models
            plot_COLORS = color_gen("Greys", 6)
            
            # Draw best-fit model
            if fits and len(self.results) > 0:
                for model in self.results:
                    model.plot(self.time, ax=ax, color=next(plot_COLORS), zorder=np.inf, share=self.share, chan=channel)
            
            # Format axes
            ax.set_title(f'{meta.eventlabel} - Channel {self.channel}')
            ax.set_xlabel(str(self.time_units))
            ax.set_ylabel('Normalized Flux', size=14)
            ax.legend(loc='best')
            fig.tight_layout()

            fname = 'figs/fig54{}_all_fits.png'.format(str(channel).zfill(len(str(self.nchannel))))
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
