from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from ..lib.plots import figure_filetype

def plot_spectrum(meta, model_x=None, model_y=None,
                  y_scalar=1, ylabel=r'$R_{\rm p}/R_{\rm *}$', xlabel=r'Wavelength ($\mu$m)',
                  scaleHeight=None, planet_R0=None):

    if scaleHeight is not None:
        fig = plt.figure(6301, figsize=(8, 4))
    else:
        fig = plt.figure(6101, figsize=(8, 4))
    plt.clf()
    ax = fig.subplots(1,1)

    wavelength = deepcopy(meta.wavelengths)
    wavelength_error = deepcopy(meta.wave_errs)
    spectrum = deepcopy(meta.spectrum_median)
    err = deepcopy(meta.spectrum_err)
    model_x = deepcopy(model_x)
    model_y = deepcopy(model_y)

    spectrum *= y_scalar
    if err is not None:
        err *= y_scalar

    ax.errorbar(wavelength, spectrum, fmt='o', capsize=3, ms=3, xerr=wavelength_error, yerr=err, color='k')
    if (model_x is not None) and (model_y is not None):
        in_range = np.logical_and(model_x>=wavelength[0]-wavelength_error[0], model_x<=wavelength[-1]+wavelength_error[-1])
        ax.plot(model_x[in_range], model_y[in_range], color='r', zorder=0)
        if wavelength_error is not None:
            binned_model = []
            for wav, width in zip(wavelength, wavelength_error):
                binned_model.append(np.mean(model_y[np.logical_and(model_x>=wav-width, model_x<wav+width)]))
            ax.plot(wavelength, binned_model, 'o', ms=3, color='r', mec='k', mew=0.5, zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if scaleHeight is not None:
        if r'^2' in ylabel:
            # We're dealing with transit depth, so need to convert accordingly
            expFactor = 2
        else:
            expFactor = 1

        if planet_R0 is None:
            H_0 = np.mean(spectrum/y_scalar)**(1/expFactor)/scaleHeight
        else:
            H_0 = planet_R0/scaleHeight

        def H(r):
            return (r/y_scalar)**(1/expFactor)/scaleHeight - H_0

        def r(H):
            return ((H+H_0)*scaleHeight)**expFactor*y_scalar

        ax2 = ax.secondary_yaxis('right', functions=(H, r))
        ax2.set_ylabel('Scale Height')

        fname = 'figs/fig6301'
    else:
        fname = 'figs/fig6101'

    if 'R_' in ylabel:
        fname += '_transmission'
    elif 'F_' in ylabel:
        fname += '_emission'

    fig.savefig(meta.outputdir+fname+figure_filetype, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

    return
