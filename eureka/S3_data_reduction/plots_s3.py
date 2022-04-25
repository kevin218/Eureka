import numpy as np
import matplotlib.pyplot as plt
from .source_pos import gauss
from ..lib.plots import figure_filetype

def lc_nodriftcorr(meta, wave_1d, optspec):
    '''Plot a 2D light curve without drift correction. (Fig 3101)

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
    optspec = np.ma.masked_invalid(optspec)
    plt.figure(3101, figsize=(8, 8))
    plt.clf()
    wmin = wave_1d.min()
    wmax = wave_1d.max()
    n_int, nx = optspec.shape
    vmin = 0.97
    vmax = 1.03
    normspec = optspec / np.ma.mean(optspec, axis=0)
    plt.imshow(normspec, origin='lower', aspect='auto', extent=[wmin, wmax, 0, n_int], vmin=vmin, vmax=vmax,
               cmap=plt.cm.RdYlBu_r)
    plt.title("MAD = " + str(np.round(meta.mad_s3, 0).astype(int)) + " ppm")
    plt.ylabel('Integration Number')
    plt.xlabel(r'Wavelength ($\mu m$)')
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/fig3101-2D_LC'+figure_filetype, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

def image_and_background(data, meta, n, m):
    '''Make image+background plot. (Figs 3301)

    Parameters
    ----------
    data:   DataClass
        The data object.
    meta:   MetaClass
        The metadata object.
    n:  int
        The integration number.
    m:  int
        The file number.

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
    plt.colorbar()
    plt.ylabel('Pixel Position')
    plt.subplot(212)
    plt.title('Subtracted Background')
    median = np.median(subbg[n])
    std = np.std(subbg[n])
    plt.imshow(subbg[n], origin='lower', aspect='auto', vmin=median - 3 * std, vmax=median + 3 * std)
    plt.colorbar()
    plt.ylabel('Pixel Position')
    plt.xlabel('Pixel Position')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.savefig(meta.outputdir + f'figs/fig3301_file{file_number}_int{int_number}_ImageAndBackground'+figure_filetype, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def optimal_spectrum(data, meta, n, m):
    '''Make optimal spectrum plot. (Figs 3302)

    Parameters
    ----------
    data:   DataClass
        The data object.
    meta:   MetaClass
        The metadata object.
    n:  int
        The integration number.
    m:  int
        The file number.

    Returns
    -------
    None
    '''
    intstart, subnx, stdspec, optspec, opterr = data.intstart, meta.subnx, data.stdspec, data.optspec, data.opterr

    plt.figure(3302)
    plt.clf()
    plt.suptitle(f'1D Spectrum - Integration {intstart + n}')
    plt.semilogy(np.arange(subnx), stdspec[n], '-', color='C1', label='Standard Spec')
    plt.errorbar(np.arange(subnx), optspec[n], yerr=opterr[n], fmt='-', color='C2', ecolor='C2', label='Optimal Spec')
    plt.ylabel('Flux')
    plt.xlabel('Pixel Position')
    plt.legend(loc='best')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.savefig(meta.outputdir + f'figs/fig3302_file{file_number}_int{int_number}_Spectrum'+figure_filetype, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def source_position(meta, x_dim, pos_max, m,
                    isgauss=False, x=None, y=None, popt=None,
                    isFWM=False, y_pixels=None, sum_row=None, y_pos=None):
    '''Plot source position for MIRI data. (Figs 3303)

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
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    plt.savefig(meta.outputdir + f'figs/fig3303_file{file_number}_source_pos'+figure_filetype, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

def profile(meta, profile, submask, n, m):
    '''Plot weighting profile from optimal spectral extraction routine. (Figs 3304)

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    profile:    ndarray
        Fitted profile in the same shape as the data array.
    submask:   ndarray
        Outlier mask.
    n:  int
        The current integration number.
    m:  int
        The file number.

    Returns
    -------
    None
    '''
    profile = np.ma.masked_invalid(profile)
    submask = np.ma.masked_invalid(submask)
    mask = np.logical_or(np.ma.getmaskarray(profile), np.ma.getmaskarray(submask))
    profile = np.ma.masked_where(mask, profile)
    submask = np.ma.masked_where(mask, submask)
    vmax = 0.05*np.ma.max(profile*submask)
    vmin = np.ma.min(profile*submask)
    plt.figure(3304)
    plt.clf()
    plt.suptitle(f"Profile - Integration {n}")
    plt.imshow(profile*submask, aspect='auto', origin='lower',vmax=vmax, vmin=vmin)
    plt.ylabel('Pixel Postion')
    plt.xlabel('Pixel Position')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.savefig(meta.outputdir + f'figs/fig3304_file{file_number}_int{int_number}_Profile'+figure_filetype, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

def subdata(meta, i, n, m, subdata, submask, expected, loc):
    '''Show 1D view of profile for each column. (Figs 3501)

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    i:  int
        The column number.
    n:  int
        The current integration number.
    m:  int
        The file number.
    subdata:    ndarray
        Background subtracted data.
    submask:    ndarray
        Outlier mask.
    expected:   ndarray
        Expected profile
    loc:    ndarray
        Location of worst outliers.    

    Returns
    -------
    None
    '''
    ny, nx = subdata.shape
    plt.figure(3501)
    plt.clf()
    plt.suptitle(f'Integration {n}, Columns {i}/{nx}')
    plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], subdata[np.where(submask[:,i])[0],i], 'bo')
    plt.plot(np.arange(ny)[np.where(submask[:,i])[0]], expected[np.where(submask[:,i])[0],i], 'g-')
    plt.plot((loc[i]), (subdata[loc[i],i]), 'ro')
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    col_number = str(i).zfill(int(np.floor(np.log10(nx))+1))
    plt.savefig(meta.outputdir + f"figs/fig3501_file{file_number}_int{int_number}_col{col_number}_subdata"+figure_filetype, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)
