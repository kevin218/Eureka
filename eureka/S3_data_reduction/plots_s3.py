import numpy as np
import matplotlib.pyplot as plt
from .source_pos import gauss

def lc_nodriftcorr(meta, wave_1d, optspec):
    '''Plot a 2D light curve without drift correction.
    
    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    wave_1d:    
        Wavelength array with trimmed edges depending on xwindow and ywindow which have been set in the S3 ecf
    optspec:    
        The optimally extracted spectrum.

    Returns
    -------
    None
    '''
    plt.figure(3101, figsize=(8, 8))  # ev.n_files/20.+0.8))
    plt.clf()
    wmin = wave_1d.min()
    wmax = wave_1d.max()
    n_int, nx = optspec.shape
    # iwmin       = np.where(ev.wave[src_ypos]>wmin)[0][0]
    # iwmax       = np.where(ev.wave[src_ypos]>wmax)[0][0]
    vmin = 0.97
    vmax = 1.03
    # normspec    = np.mean(ev.optspec,axis=1)/np.mean(ev.optspec[ev.inormspec[0]:ev.inormspec[1]],axis=(0,1))
    normspec = optspec / np.mean(optspec, axis=0)
    plt.imshow(normspec, origin='lower', aspect='auto', extent=[wmin, wmax, 0, n_int], vmin=vmin, vmax=vmax,
               cmap=plt.cm.RdYlBu_r)
    ediff = np.zeros(n_int)
    for m in range(n_int):
        ediff[m] = 1e6 * np.median(np.abs(np.ediff1d(normspec[m])))
        # plt.scatter(ev.wave[src_ypos], np.zeros(nx)+m, c=normspec[m],
        #             s=14,linewidths=0,vmin=vmin,vmax=vmax,marker='s',cmap=plt.cm.RdYlBu_r)
    plt.title("MAD = " + str(np.round(np.mean(ediff), 0).astype(int)) + " ppm")
    # plt.xlim(wmin,wmax)
    # plt.ylim(0,n_int)
    plt.ylabel('Integration Number')
    plt.xlabel(r'Wavelength ($\mu m$)')
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/fig3101-2D_LC.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

def image_and_background(data, meta, n):
    '''Make image+background plot.
    
    Parameters
    ----------
    data:   DataClass
        The data object.
    meta:   MetaClass
        The metadata object.
    n:  int
        The integration number.
    
    Returns
    -------
    None
    '''
    intstart, subdata, submask, subbg = data.intstart, data.subdata, data.submask, data.subbg

    plt.figure(3301, figsize=(8,8))
    plt.clf()
    plt.suptitle(f'Integration {intstart + n}')
    plt.subplot(211)
    plt.title('Background-Subtracted Flux')
    max = np.max(subdata[n] * submask[n])
    plt.imshow(subdata[n] * submask[n], origin='lower', aspect='auto', vmin=0, vmax=max / 10)
    # plt.imshow(subdata[n], origin='lower', aspect='auto', vmin=0, vmax=10000)
    plt.ylabel('Pixel Position')
    plt.subplot(212)
    plt.title('Subtracted Background')
    # plt.imshow(submask[i], origin='lower', aspect='auto', vmax=1)
    median = np.median(subbg[n])
    std = np.std(subbg[n])
    plt.imshow(subbg[n], origin='lower', aspect='auto', vmin=median - 3 * std, vmax=median + 3 * std)
    # plt.imshow(submask[n], origin='lower', aspect='auto', vmin=0, vmax=1)
    plt.ylabel('Pixel Position')
    plt.xlabel('Pixel Position')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/fig3301-' + str(intstart + n) + '-Image+Background.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)


def optimal_spectrum(data, meta, n):
    '''Make optimal spectrum plot.
    
    Parameters
    ----------
    data:   DataClass
        The data object.
    meta:   MetaClass
        The metadata object.
    n:  int
        The integration number.
    
    Returns
    -------
    None
    '''
    intstart, subnx, stdspec, optspec, opterr = data.intstart, meta.subnx, data.stdspec, data.optspec, data.opterr

    plt.figure(3302)
    plt.clf()
    plt.suptitle(f'1D Spectrum - Integration {intstart + n}')
    plt.semilogy(range(subnx), stdspec[n], '-', color='C1', label='Standard Spec')
    # plt.errorbar(range(subnx), stdspec[n], yerr=np.sqrt(stdvar[n]), fmt='-', color='C1', ecolor='C0', label='Std Spec')
    plt.errorbar(range(subnx), optspec[n], opterr[n], fmt='-', color='C2', ecolor='C2', label='Optimal Spec')
    plt.ylabel('Flux')
    plt.xlabel('Pixel Position')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/fig3302-' + str(intstart + n) + '-Spectrum.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)


def source_position(meta, x_dim, pos_max, m,
                    isgauss=False, x=None, y=None, popt=None,
                    isFWM=False, y_pixels=None, sum_row=None, y_pos=None):
    '''Plot source position for MIRI data.
    
    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    x_dim:  int
        The number of pixels in the y-direction in the image.
    pos_max:    float
        The brightest row.
    m:  int
        The file number.
    y_pixels:   1darray
        The indices of the y-pixels.
    sum_row:    1darray
        The sum over each row.
    isgauss:    bool
        Used a guassian centring method.
    popt:   list
        The fitted Gaussian terms.
    isFWM:  bool
        Used a flux-weighted mean centring method.
    y_pos:  float
        The FWM central position of the star.
    
    Returns
    -------
    None

    Notes
    -----
    History:

    - 2021-07-14: Sebastian Zieba
        Initial version.
    - Oct 15, 2021: Taylor Bell
        Tided up the code a bit to reduce repeated code.
    '''
    plt.figure(3303)
    plt.clf()
    plt.plot(y_pixels, sum_row, 'o', label= 'Data')
    if isgauss:
        x_gaussian = np.linspace(0,x_dim,500)
        gaussian = gauss(x_gaussian, *popt)
        plt.plot(x_gaussian, gaussian, '-', label= 'Gaussian Fit')
        plt.axvline(popt[1], ls=':', label= 'Gaussian Center', c='C2')
        plt.xlim(pos_max-meta.spec_hw, pos_max+meta.spec_hw)
    elif isFWM:
        plt.axvline(y_pos, ls='-', label= 'Weighted Row')
    plt.axvline(pos_max, ls='--', label= 'Brightest Row', c='C3')
    plt.ylabel('Row Flux')
    plt.xlabel('Row Pixel Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/fig3303-file' + str(m+1) + '-source_pos.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

def profile(eventdir, profile, submask, n, hide_plots=False):
    '''
    Plot weighting profile from optimal spectral extraction routine

    Parameters
    ----------
    eventdir:   str
        Directory in which to save outupts.
    profile:    ndarray
        Fitted profile in the same shape as the data array.
    submask:   ndarray
        Outlier mask.
    n:  int
        The current integration number.
    hide_plots: 
        If True, plots will automatically be closed rather than popping up.

    Returns
    -------
    None
    '''
    vmax = 0.05*np.max(profile*submask)
    plt.figure(3305)
    plt.clf()
    plt.suptitle(f"Profile - Integration {n}")
    plt.imshow(profile*submask, aspect='auto', origin='lower',vmax=vmax)
    plt.ylabel('Pixel Postion')
    plt.xlabel('Pixel Position')
    plt.tight_layout()
    plt.savefig(eventdir+'figs/fig3305-'+str(n)+'-Profile.png')
    if hide_plots:
        plt.close()
    else:
        plt.pause(0.2)
