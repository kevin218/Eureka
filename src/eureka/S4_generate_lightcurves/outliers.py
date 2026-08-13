import numpy as np
from astropy.stats import sigma_clip

from ..lib import smooth, util


def get_outliers(meta, spec):
    '''Use spectroscopic MAED values to identify outliers.
    Outliers will be appended to `mask_columns` in the Stage 4 ECF.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    spec : Xarray Dataset
        The Dataset object containing spectroscopic LC and time data.

    Returns
    -------
    outliers : 1D array
        An array of detector pixel indices flagged as outliers.
    pp : Dictionary
        A dictionary of plotting parameters for Fig 4106.
    '''
    # Normalize the light curve
    wave_1d = spec.wave_1d.values
    iwmin = np.nanargmin(np.abs(wave_1d - meta.wave_min))
    iwmax = np.nanargmin(np.abs(wave_1d - meta.wave_max))
    optspec = spec.optspec.values[:, iwmin:iwmax]
    opterr = spec.opterr.values[:, iwmin:iwmax]
    optmask = spec.optmask.values[:, iwmin:iwmax]
    norm_lcdata, norm_lcerr = util.normalize_spectrum(meta, optspec, opterr,
                                                      optmask=optmask)
    norm_lcdata = norm_lcdata.filled(np.nan)
    norm_lcerr = norm_lcerr.filled(np.nan)

    # Compute unbinned LC MAED values, then scale
    numx = norm_lcdata.shape[1]
    maed = np.zeros(numx)
    for ii in range(numx):
        maed[ii] = util.get_maed_1d(norm_lcdata[:, ii])

    # Compute mean abs deviation from "white" LC, then scale
    optspec_mean = np.nanmean(norm_lcdata, axis=1)
    dev = np.zeros(numx)
    for ii in range(numx):
        dev[ii] = np.ma.mean(np.ma.abs((norm_lcdata[:, ii] - optspec_mean)))
    dev /= np.nanmean(dev)/np.nanmean(maed)

    # Remove broad trends from native-resolution MAED values
    mask = np.isnan(maed)
    x = spec.x[iwmin:iwmax]
    x_mask = x[~mask]
    smoothed_maed = smooth.medfilt(maed[~mask], window_len=meta.maed_box_width)
    residual_maed = maed[~mask] - smoothed_maed
    smoothed_dev = smooth.medfilt(dev[~mask], window_len=meta.maed_box_width)
    residual_dev = dev[~mask] - smoothed_dev

    # Identify only high outliers from residuals
    masked_maed = sigma_clip(residual_maed, sigma_upper=meta.maed_sigma,
                            sigma_lower=100, maxiters=meta.maxiters,
                            masked=True, copy=True)
    masked_dev = sigma_clip(residual_dev, sigma_upper=meta.maed_sigma,
                            sigma_lower=100, maxiters=meta.maxiters,
                            masked=True, copy=True)
    x_maed_outliers = x_mask[np.ma.getmaskarray(masked_maed)]
    x_dev_outliers = x_mask[np.ma.getmaskarray(masked_dev)]
    outliers = np.union1d(x_maed_outliers, x_dev_outliers)

    # Create dictionary containing plotting parameters for Fig 4106
    pp = {
        "x": x,
        "x_mask": x_mask,
        "x_maed_outliers": x_maed_outliers,
        "x_dev_outliers": x_dev_outliers,
        "maed": maed,
        "dev": dev,
        "masked_maed": masked_maed,
        "masked_dev": masked_dev,
        "smoothed_maed": smoothed_maed,
        "residual_maed": residual_maed,
        "smoothed_dev": smoothed_dev,
        "residual_dev": residual_dev}

    return outliers, pp
