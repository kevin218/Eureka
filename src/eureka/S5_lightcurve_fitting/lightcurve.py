import numpy as np
import os
import matplotlib.pyplot as plt
from copy import copy

from . import models as m
from . import fitters
from .utils import COLORS, color_gen
from ..lib import plots, util
from ..lib.split_channels import get_trim, split


class LightCurve(m.Model):
    def __init__(self, time, flux, channel, nchannel, log, longparamlist,
                 parameters, freenames, unc=None, time_units='BJD',
                 name='My Light Curve', share=False, white=False,
                 multwhite=False, nints=[], **kwargs):
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
        parameters : eureka.lib.readEPF.Parameters
            The Parameters object containing the fitted parameters
            and their priors.
        freenames : list
            The specific names of all fitted parameters (e.g., including _ch#)
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
        **kwargs : dict
            Parameters to set in the LightCurve object.
            Any parameter named log will not be loaded into the
            LightCurve object as Logedit objects cannot be pickled
            which is required for multiprocessing.
        """
        # Initialize the model
        super().__init__(**kwargs)

        self.name = name
        self.share = share
        self.white = white
        self.channel = channel
        self.nchannel = nchannel
        self.multwhite = multwhite
        self.nints = nints
        self.freenames = freenames
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
            if isinstance(unc, (float, np.float64)):
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
            for chan in range(self.nchannel_fitted):
                trim1, trim2 = get_trim(nints, chan)
                name = 'scatter_mult'
                if chan > 0:
                    name += f'_ch{chan}'
                self.unc_fit[trim1:trim2] *= getattr(parameters, name).value
        elif hasattr(parameters, 'scatter_ppm'):
            for chan in range(self.nchannel_fitted):
                trim1, trim2 = get_trim(nints, chan)
                name = 'scatter_ppm'
                if chan > 0:
                    name += f'_ch{chan}'
                self.unc_fit[trim1:trim2] *= \
                    getattr(parameters, name).value/1e6

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
        """
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
            raise NotImplementedError(
                'PyMC3/starry support within Eureka! had to be dropped because'
                ' PyMC3 is now extremely deprecated and incompatible with the '
                'current jwst pipeline version. For the time being, you must '
                'instead use the numpy-based models.')
        elif fitter == 'nuts':
            raise NotImplementedError(
                'PyMC3/starry support within Eureka! had to be dropped because'
                ' PyMC3 is now extremely deprecated and incompatible with the '
                'current jwst pipeline version. For the time being, you must '
                'instead use the numpy-based models.')
        else:
            raise ValueError("{} is not a valid fitter.".format(fitter))

        # Run the fit
        fit_model = self.fitter_func(self, model, meta, log, **kwargs)

        # Store it
        if fit_model is not None:
            self.results.append(copy(fit_model))

    @plots.apply_style
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
            flux = np.ma.copy(self.flux)
            unc = np.ma.copy(self.unc_fit)
            time = np.ma.copy(self.time)

            if self.share and not meta.multwhite:
                # Split the arrays that have lengths of the original time axis
                flux, unc = split([flux, unc], meta.nints, channel)
            elif meta.multwhite:
                # Split the arrays that have lengths of the original time axis
                time, flux, unc = split([time, flux, unc],
                                        meta.nints, channel)

            # Get binned data and times
            if not meta.nbin_plot or meta.nbin_plot > len(time):
                nbin_plot = len(time)
                binned_time = time
                binned_flux = flux
                binned_unc = unc
            else:
                nbin_plot = meta.nbin_plot
                binned_time = util.binData_time(time, time, nbin=nbin_plot)
                binned_flux = util.binData_time(flux, time, nbin=nbin_plot)
                binned_unc = util.binData_time(unc, time, nbin=nbin_plot,
                                               err=True)

            fig = plt.figure(5103)
            fig.set_size_inches(8, 6, forward=True)
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

            # Determine wavelength
            if meta.multwhite:
                wave = meta.wave[0]
            else:
                wave = meta.wave[channel]
            # Format axes
            ax.set_title(f'{meta.eventlabel} - Channel {channel} ' +
                         f'- {wave} microns')
            ax.set_xlabel(str(self.time_units))
            ax.set_ylabel('Normalized Flux', size=14)
            ax.legend(loc='best')

            if self.white:
                fname_tag = 'white'
            else:
                ch_number = str(channel).zfill(len(str(self.nchannel)))
                fname_tag = f'ch{ch_number}'
            fname = (f'figs{os.sep}fig5103_{fname_tag}_all_fits' +
                     plots.get_filetype())
            fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
            if not meta.hide_plots:
                plt.pause(0.2)

            # Show unbinned data as well if requested
            if nbin_plot != len(time) and meta.isplots_S5 >= 3:
                fig = plt.figure(5306, figsize=(8, 6))
                fig.clf()
                # Draw the data
                ax = fig.gca()
                ax.errorbar(time, flux, unc, fmt='.', color=self.colors[i],
                            zorder=0, alpha=0.1)
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

                if self.white:
                    fname_tag = 'white'
                else:
                    ch_number = str(channel).zfill(len(str(self.nchannel)))
                    fname_tag = f'ch{ch_number}'
                fname = (f'figs{os.sep}fig5306_{fname_tag}_all_fits' +
                         plots.get_filetype())
                fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
                if not meta.hide_plots:
                    plt.pause(0.2)

    def reset(self):
        """Reset the results"""
        self.results = []
