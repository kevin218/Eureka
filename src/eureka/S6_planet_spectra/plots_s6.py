from copy import deepcopy
import numpy as np
import os
import matplotlib.pyplot as plt
import re

from ..lib import plots


def plot_spectrum(meta, model_x=None, model_y=None,
                  y_scalar=1, ylabel=r'$R_{\rm p}/R_{\rm *}$',
                  xlabel=r'Wavelength ($\mu$m)',
                  scaleHeight=None, planet_R0=None):
    r"""Plot the planetary transmission or emission spectrum.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The meta data object.
    model_x : ndarray; optional
        The wavelength array for the model to plot, by default None
    model_y : ndarray; optional
        The transmission or emission spectrum from the model, by default None
    y_scalar : int; optional
        The multiplier for the y-axis (100=%, 1e6=ppm), by default 1
    ylabel : str; optional
        The y-axis label, by default r'{\rm p}/R_{\rm \*}$'
    xlabel : str; optional
        The x-axis label, by default r'Wavelength ($\mu)'
    scaleHeight : float; optional
        The planetary atmospheric scale height, by default None
    planet_R0 : float; optional
        The reference radius for the scale height, by default None
    """
    if scaleHeight is not None:
        fig = plt.figure(6301, figsize=(8, 4))
    else:
        fig = plt.figure(6101, figsize=(8, 4))
    plt.clf()
    ax = fig.subplots(1, 1)

    wavelength = deepcopy(meta.wavelengths)
    wavelength_error = deepcopy(meta.wave_errs)
    spectrum = deepcopy(meta.spectrum_median)
    err = deepcopy(meta.spectrum_err)
    model_x = deepcopy(model_x)
    model_y = deepcopy(model_y)

    # Trim repeated wavelengths for multwhite fits
    if len(set(wavelength)) == 1:
        wavelength = wavelength[0]
        wavelength_error = wavelength_error[0]

    if np.all(np.isnan(err)):
        err = None

    spectrum *= y_scalar
    if err is not None:
        err *= y_scalar

    # Set zorder to 0.5 so model can easily be placed above or below
    ax.errorbar(wavelength, spectrum, fmt='o', capsize=3, ms=3,
                xerr=wavelength_error, yerr=err, color='k',
                zorder=0.5)
    if (model_x is not None) and (model_y is not None):
        in_range = np.logical_and(model_x >= wavelength[0]-wavelength_error[0],
                                  model_x <= (wavelength[-1] +
                                              wavelength_error[-1]))
        ax.plot(model_x[in_range], model_y[in_range], color='r',
                zorder=meta.model_zorder)
        if wavelength_error is not None:
            # Compute the binned model for easlier comparisons
            binned_model = []
            for wav, width in zip(wavelength, wavelength_error):
                inds = np.logical_and(model_x >= wav-width,
                                      model_x < wav+width)
                model_val = np.mean(model_y[inds])
                binned_model.append(model_val)
            # Plot the binned model as well
            ax.plot(wavelength, binned_model, 'o', ms=3, color='r', mec='k',
                    mew=0.5, zorder=meta.model_zorder)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if scaleHeight is not None:
        # Make a copy of the figure with scale height as the second y axis
        if r'^2' in ylabel:
            # We're dealing with transit depth, so need to convert accordingly
            expFactor = 2
        else:
            expFactor = 1

        if planet_R0 is None:
            H_0 = np.nanmean(spectrum/y_scalar)**(1/expFactor)/scaleHeight
        else:
            H_0 = planet_R0/scaleHeight

        def H(r):
            return (r/y_scalar)**(1/expFactor)/scaleHeight - H_0

        def r(H):
            return ((H+H_0)*scaleHeight)**expFactor*y_scalar

        # Need to enforce non-negative for H(r) to make sense
        if ax.get_ylim()[0] < 0:
            ax.set_ylim(0)

        ax2 = ax.secondary_yaxis('right', functions=(H, r))
        ax2.set_ylabel('Scale Height')
        if r'^2' in ylabel:
            # To avoid overlapping ticks, use the same spacing as the r axis
            yticks = np.round(H(ax.get_yticks()), 1)
            # Make sure H=0 is shown
            offset = yticks[np.argmin(np.abs(yticks))]
            yticks -= offset
            ax2.set_yticks(yticks)

        fname = 'figs'+os.sep+'fig6301'
    else:
        fname = 'figs'+os.sep+'fig6101'

    clean_y_param = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", meta.y_param)
    fname += '_'+clean_y_param

    fig.savefig(meta.outputdir+fname+plots.figure_filetype,
                bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

    return
