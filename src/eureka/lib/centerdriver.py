import numpy as np
from . import imageedit as ie
from . import gaussian as g
from . import gaussian_min as gmin
from ..S3_data_reduction import plots_s3


def centerdriver(method, data, meta, i=None, m=None):
    """
    Use the center method to find the center of a star, starting
    from position guess.

    Parameters
    ----------
    method : string
        Name of the centering method to use.
    data : Xarray Dataset
        The Dataset object in which the centroid data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int; optional
        The current integration. Defaults to None.
    m : int; optional
        The file number. Defaults to None.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the centroid data stored inside.
    """
    # Apply the mask
    mask = data.mask.values[i]
    flux = np.ma.masked_where(mask, data.flux.values[i])
    err = np.ma.masked_where(mask, data.err.values[i])
    saved_ref_median_frame = data.medflux.values

    yxguess = [data.centroid_y.values[i], data.centroid_x.values[i]]

    if method[-4:] == '_sec':
        trim = meta.ctr_cutout_size
    else:
        trim = 0

    if method in ['fgc_pri', 'fgc_sec']:
        # Trim the image if requested
        if trim != 0:
            # Integer part of center
            cen = np.rint(yxguess)
            # Center in the trimed image
            loc = (trim, trim)
            # Do the trim:
            flux, mask, err = ie.trimimage(flux, cen, loc, mask=mask, uncd=err)
        else:
            cen = np.array([0, 0])
            loc = np.rint(yxguess)
        weights = 1.0 / np.abs(err)
    else:
        trim = 0
        loc = yxguess
        cen = np.array([0, 0])
        # Subtract median BG because photutils sometimes has a hard time
        # fitting for a constant offset
        flux -= np.ma.median(flux)

    # If all data is bad:
    if np.all(mask):
        raise Exception('Bad Frame Exception!')

    # Get the center with one of the methods:
    if method in ['fgc_pri', 'fgc_sec']:
        sy, sx, y, x = g.fitgaussian(flux, yxguess=loc, mask=mask,
                                     weights=weights,
                                     fitbg=1, maskg=False)[0][0:4]
    elif method == 'mgmc_pri':
        # Median frame creation + first centroid
        x, y = gmin.pri_cent(flux, mask, meta, data.medflux.values)
        sy, sx = np.nan, np.nan
    elif method == 'mgmc_sec':
        # Second enhanced centroid position + gaussian widths
        sy, sx, y, x = gmin.mingauss(flux, mask, yxguess=loc, meta=meta)

    # only plot when we do the second fit
    if (meta.isplots_S3 >= 3 and method[-4:] == '_sec' and i < meta.nplots):
        plots_s3.phot_centroid_fgc(flux, mask, x, y, sx, sy, i, m, meta)

    # Make trimming correction, then store centroid positions and
    # the Gaussian 1-sigma half-widths
    data.centroid_x.values[i] = x + cen[1] - trim
    data.centroid_y.values[i] = y + cen[0] - trim
    data.centroid_sx.values[i] = sx
    data.centroid_sy.values[i] = sy

    return data
