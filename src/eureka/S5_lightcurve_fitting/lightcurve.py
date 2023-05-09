import numpy as np
import os
import matplotlib.pyplot as plt
from copy import copy

from . import models as m
from . import fitters
from . import gradient_fitters
from .utils import COLORS, color_gen
from ..lib import plots, util
from ..lib.split_channels import split


class LightCurve(m.Model):
    def __init__(self, time, flux, channel, nchannel, log, longparamlist,
                 parameters, unc=None, time_units='BJD',
                 name='My Light Curve', share=False, white=False,
                 multwhite=False, nints=[]):
        """
        A class to store the actual light curve

        Parameters
        ----------
        time : sequence
            The time axis in days.
        flux : sequence
            The flux in electrons (not ADU).
        channel : int
            The channel number.
        nChannel : int
            The total number of channels.
        log : logedit.Logedit
            The open log in which notes from this step can be added.
        unc : sequence
            The uncertainty on the flux
        parameters : str or object
            The orbital parameters of the star/planet system,
            may be a path to a JSON file or a parameter object.
        time_units : str; optional
            The time units.
        name : str; optional
            A name for the object.
        share : bool; optional
            Whether the fit shares parameters between spectral channels.
        white : bool; optional
            Whether the current fit is for a white-light light curve
        multwhite : bool; optional
            Whether the current fit is for a multi white-light lightcurve fit.
        nints : bool; optional
            Number of exposures of each white lightcurve for splitting
            up time array.

        Notes
        -----

        History:
        - Dec 29, 2021 Taylor Bell
            Allowing for a constant uncertainty to be input with just a float.
            Added a channel number.
        - Jan. 15, 2022 Megan Mansfield
            Added ability to share fit between all channels
        - Oct. 2022 Erin May
            Added ability to joint fit WLCs with different time arrays    
        """
        # Initialize the model
        super().__init__()

        self.name = name
        self.share = share
        self.white = white
        self.channel = channel
        self.nchannel = nchannel
        self.multwhite = multwhite
        self.nints = nints
        if self.share or self.multwhite:
            self.nchannel_fitted = self.nchannel
            self.fitted_channels = np.arange(self.nchannel)
        else:
            self.nchannel_fitted = 1
            self.fitted_channels = np.array([self.channel])

        # Check data
        if len(time)*self.nchannel_fitted != len(flux) and not self.multwhite:
            raise ValueError('Time and flux axes must be the same length.')

        # Set the time and flux axes
        self.flux = flux
        self.time = time
        # Set the units
        self.time_units = time_units

        # Set the data arrays
        if unc is not None:
            if type(unc) == float or type(unc) == np.float64:
                log.writelog('Warning: Only one uncertainty input, assuming '
                             'constant uncertainty.')
            elif (len(time)*self.nchannel_fitted != len(unc)
                  and not self.multwhite):
                raise ValueError('Time and unc axes must be the same length.')

            self.unc = unc
        else:
            self.unc = np.array([np.nan]*len(self.time))
        self.unc_fit = np.ma.copy(self.unc)

        if hasattr(parameters, 'scatter_mult'):
            self.unc_fit *= parameters.scatter_mult.value
        elif hasattr(parameters, 'scatter_ppm'):
            self.unc_fit[:] = parameters.scatter_ppm.value/1e6

        # Place to save the fit results
        self.results = []

        self.longparamlist = longparamlist

        self.colors = np.array([next(COLORS)
                                for i in range(self.nchannel_fitted)])

        return

    def fit(self, model, meta, log, fitter='lsq', **kwargs):
        """Fit the model to the lightcurve

        Parameters
        ----------
        model : eureka.S5_lightcurve_fitting.models.CompositeModel
            The model to fit to the data.
        meta : eureka.lib.readECF.MetaClass
            The metadata object.
        log : logedit.Logedit
            The open log in which notes from this step can be added.
        fitter : str
            The name of the fitter to use.
        **kwargs : dict
            Arbitrary keyword arguments.

        Notes
        -----
        History:

        - Dec 29, 2021 Taylor Bell
            Updated documentation and reduced repeated code
        """
        # Empty default fit
        fit_model = None

        model.time = self.time
        model.multwhite = meta.multwhite

        if fitter not in ['exoplanet', 'nuts']:
            # Make sure the model is a CompositeModel
            if not isinstance(model, m.CompositeModel):
                model = m.CompositeModel([model])
                model.time = self.time

        if fitter == 'lmfit':
            self.fitter_func = fitters.lmfitter
        elif fitter == 'lsq':
            self.fitter_func = fitters.lsqfitter
        # elif fitter == 'demc':
        #     self.fitter_func = fitters.demcfitter
        elif fitter == 'emcee':
            self.fitter_func = fitters.emceefitter
        elif fitter == 'dynesty':
            self.fitter_func = fitters.dynestyfitter
        elif fitter == 'exoplanet':
            self.fitter_func = gradient_fitters.exoplanetfitter
        elif fitter == 'nuts':
            self.fitter_func = gradient_fitters.nutsfitter
        else:
            raise ValueError("{} is not a valid fitter.".format(fitter))

        # Run the fit
        fit_model = self.fitter_func(self, model, meta, log, **kwargs)

        # Store it
        if fit_model is not None:
            self.results.append(copy(fit_model))

    def plot(self, meta, fits=True):
        """Plot the light curve with all available fits. (Figs 5103 and 5306)

        Parameters
        ----------
        meta : eureka.lib.readECF.MetaClass
            The metadata object.
        fits : bool; optional
            Plot the fit models. Defaults to True.
        """
        # Make the figure
        for i, channel in enumerate(self.fitted_channels):
            flux = self.flux
            unc = np.ma.copy(self.unc_fit)
            time = self.time
            
            if self.share and not meta.multwhite:
                # Split the arrays that have lengths of the original time axis
                flux, unc = split([flux, unc], meta.nints, channel)
            elif meta.multwhite:
                # Split the arrays that have lengths of the original time axis
                time, flux, unc = split([time, flux, unc],
                                        meta.nints, channel)

            # Get binned data and times
            if not hasattr(meta, 'nbin_plot') or meta.nbin_plot is None or \
               meta.nbin_plot > len(time):
                nbin_plot = len(time)
            else:
                nbin_plot = meta.nbin_plot
            binned_time = util.binData(time, nbin_plot)
            binned_flux = util.binData(flux, nbin_plot)
            binned_unc = util.binData(unc, nbin_plot, err=True)

            fig = plt.figure(5103, figsize=(8, 6))
            fig.clf()
            # Draw the data
            ax = fig.gca()
            ax.errorbar(binned_time, binned_flux, binned_unc, fmt='.',
                        color=self.colors[i], zorder=0)

            # Make a new color generator for the models
            plot_COLORS = color_gen("Greys", 6)

            # Draw best-fit model
            if fits and len(self.results) > 0:
                for model in self.results:
                    model.plot(ax=ax, color=next(plot_COLORS),
                               zorder=np.inf, share=self.share, chan=channel)

            # Format axes
            ax.set_title(f'{meta.eventlabel} - Channel {channel}')
            ax.set_xlabel(str(self.time_units))
            ax.set_ylabel('Normalized Flux', size=14)
            ax.legend(loc='best')
            fig.tight_layout()

            if self.white:
                fname_tag = 'white'
            else:
                ch_number = str(channel).zfill(len(str(self.nchannel)))
                fname_tag = f'ch{ch_number}'
            fname = (f'figs{os.sep}fig5103_{fname_tag}_all_fits' +
                     plots.figure_filetype)
            fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
            if not meta.hide_plots:
                plt.pause(0.2)

            # Show unbinned data as well if requested
            if nbin_plot != len(time) and meta.isplots_S5 >= 3:
                fig = plt.figure(5306, figsize=(8, 6))
                fig.clf()
                # Draw the data
                ax = fig.gca()
                ax.plot(time, flux, '.', color=self.colors[i], zorder=0,
                        alpha=0.01)
                ax.errorbar(binned_time, binned_flux, binned_unc, fmt='.',
                            color=self.colors[i], zorder=1)

                # Make a new color generator for the models
                plot_COLORS = color_gen("Greys", 6)

                # Draw best-fit model
                if fits and len(self.results) > 0:
                    for model in self.results:
                        model.plot(ax=ax, color=next(plot_COLORS),
                                   zorder=np.inf, share=self.share,
                                   chan=channel)

                # Format axes
                ax.set_title(f'{meta.eventlabel} - Channel {channel}')
                ax.set_xlabel(str(self.time_units))
                ax.set_ylabel('Normalized Flux', size=14)
                ax.legend(loc='best')
                fig.tight_layout()

                if self.white:
                    fname_tag = 'white'
                else:
                    ch_number = str(channel).zfill(len(str(self.nchannel)))
                    fname_tag = f'ch{ch_number}'
                fname = (f'figs{os.sep}fig5306_{fname_tag}_all_fits' +
                         plots.figure_filetype)
                fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
                if not meta.hide_plots:
                    plt.pause(0.2)

    def reset(self):
        """Reset the results"""
        self.results = []
