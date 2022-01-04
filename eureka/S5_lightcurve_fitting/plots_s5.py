import numpy as np
import matplotlib.pyplot as plt
import corner

from .likelihood import computeRMS

def plot_fit(lc, model, meta, fitter):
    """Plot the fitted model over the data.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    if type(fitter)!=str:
        raise ValueError('Expected type str for fitter, instead received a {}'.format(type(fitter)))
    
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc
    if meta.isplots_S5 >= 1:
        model.plot(time=lc.time, draw=True)
        
        plt.figure(int('51{}'.format(str(lc.channel).zfill(len(str(lc.nchannel))))), figsize=(8, 6))
        plt.clf()
        fig, ax = plt.subplots(2,1)
        ax[0].errorbar(lc.time, lc.flux, yerr=lc.unc, fmt='.')
        ax[0].plot(lc.time, model_lc, zorder = 10)

        ax[1].errorbar(lc.time, residuals, yerr=lc.unc, fmt='.')

        fname = 'figs/fig51{}_lc_{}.png'.format(str(lc.channel).zfill(len(str(lc.nchannel))), fitter)
        plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if meta.hide_plots:
            plt.close()
        else:
            plt.pause(0.1)
    
    return

def plot_rms(lc, model, meta, fitter):
    """Plot an Allan plot to look for red noise.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    if type(fitter)!=str:
        raise ValueError('Expected type str for fitter, instead received a {}'.format(type(fitter)))
    time = lc.time
    model_lc = model.eval()
    residuals = lc.flux - model_lc
    residuals = residuals[np.argsort(time)]

    rms, stderr, binsz = computeRMS(residuals, binstep=1)
    normfactor = 1e-6
    plt.rcParams.update({'legend.fontsize': 11}) # FINDME: this should not be done here but where the rcparams are defined for Eureka
    plt.figure(int('52{}'.format(str(lc.channel).zfill(len(str(lc.nchannel))))), figsize=(8, 6))
    plt.clf()
    plt.suptitle(' Correlated Noise', size=16)
    plt.loglog(binsz, rms / normfactor, color='black', lw=1.5, label='Fit RMS', zorder=3)  # our noise
    plt.loglog(binsz, stderr / normfactor, color='red', ls='-', lw=2, label='Std. Err.', zorder=1)  # expected noise
    plt.xlim(0.95, binsz[-1] * 2)
    plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("RMS (ppm)", fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend()
    fname = 'figs/fig52{}_'.format(str(lc.channel).zfill(len(str(lc.nchannel))))+'allanplot_'+fitter+'.png'
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.1)

    return
    
def plot_corner(samples, lc, meta, freenames, fitter):
    """Plot a corner plot.

    Parameters
    ----------
    samples: ndarray
        The samples produced by the sampling algorithm
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    freenames: iterable
        The names of the fitted parameters
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    fig = corner.corner(samples, show_titles=True,quantiles=[0.16, 0.5, 0.84],title_fmt='.4', labels=freenames)
    fname = 'figs/fig53{}_corner_{}.png'.format(str(lc.channel).zfill(len(str(lc.nchannel))), fitter)
    fig.savefig(meta.outputdir+fname, bbox_inches='tight', pad_inches=0.05, dpi=250)
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.1)

    return