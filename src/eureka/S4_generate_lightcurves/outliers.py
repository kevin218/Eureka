import numpy as np
from astropy.stats import sigma_clip
from ..lib import util, smooth


def get_outliers(meta, spec):
    '''Use spectroscopic MAD values to identify outliers.
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

    # Compute unbinned LC MAD values, then scale
    numx = norm_lcdata.shape[1]
    mad = np.zeros(numx)
    for ii in range(numx):
        mad[ii] = util.get_mad_1d(norm_lcdata[:, ii])

    # Compute mean abs deviation from "white" LC, then scale
    optspec_mean = np.nanmean(norm_lcdata, axis=1)
    dev = np.zeros(numx)
    for ii in range(numx):
        dev[ii] = np.ma.mean(np.ma.abs((norm_lcdata[:, ii] - optspec_mean)))
    dev /= np.nanmean(dev)/np.nanmean(mad)

    # Remove broad trends from native-resolution MAD values
    mask = np.isnan(mad)
    x = spec.x[iwmin:iwmax]
    x_mask = x[~mask]
    smoothed_mad = smooth.medfilt(mad[~mask], window_len=meta.mad_box_width)
    residual_mad = mad[~mask] - smoothed_mad
    smoothed_dev = smooth.medfilt(dev[~mask], window_len=meta.mad_box_width)
    residual_dev = dev[~mask] - smoothed_dev

    # Identify only high outliers from residuals
    masked_mad = sigma_clip(residual_mad, sigma_upper=meta.mad_sigma,
                            sigma_lower=100, maxiters=meta.maxiters,
                            masked=True, copy=True)
    masked_dev = sigma_clip(residual_dev, sigma_upper=meta.mad_sigma,
                            sigma_lower=100, maxiters=meta.maxiters,
                            masked=True, copy=True)
    x_mad_outliers = x_mask[np.ma.getmaskarray(masked_mad)]
    x_dev_outliers = x_mask[np.ma.getmaskarray(masked_dev)]
    outliers = np.union1d(x_mad_outliers, x_dev_outliers)

    # Create dictionary containing plotting parameters for Fig 4106
    pp = {
        "x": x,
        "x_mask": x_mask,
        "x_mad_outliers": x_mad_outliers,
        "x_dev_outliers": x_dev_outliers,
        "mad": mad,
        "dev": dev,
        "masked_mad": masked_mad,
        "masked_dev": masked_dev,
        "smoothed_mad": smoothed_mad,
        "residual_mad": residual_mad,
        "smoothed_dev": smoothed_dev,
        "residual_dev": residual_dev}

    return outliers, pp