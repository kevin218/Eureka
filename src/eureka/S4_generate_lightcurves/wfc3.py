import numpy as np


def sum_reads(spec, lc, meta):
    """Sum together the non-destructive reads from each file to reduce noise
    and data volume.

    Missing reads are excluded, and the remaining flux is rescaled to the
    full number of reads. Errors and variances use the same flux scaling.

    spec : Xarray Dataset
        The Dataset object containing the 2D spectra.
    lc : Xarray Dataset
        The Dataset object containing light curve and time data.
    meta : eureka.lib.readECF.MetaClass
        The current metadata object.

    Returns
    -------
    spec : Xarray Dataset
        The updated spec Dataset object.
    lc : Xarray Dataset
        The updated lc Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    # Get a copy of the flux and time arrays
    flux = np.copy(spec.optspec.values)
    err = np.copy(spec.opterr.values)
    optmask = np.copy(spec.optmask.values)
    std_flux = np.copy(spec.stdspec.values)
    std_var = np.copy(spec.stdvar.values)
    time = np.copy(spec.optspec.time.values)

    # Reshape to get (nfiles, nreads, nwaves)
    flux = flux.reshape(-1, meta.nreads, flux.shape[1])
    err = err.reshape(-1, meta.nreads, err.shape[1])
    optmask = optmask.reshape(-1, meta.nreads, optmask.shape[1])
    std_flux = std_flux.reshape(-1, meta.nreads, std_flux.shape[1])
    std_var = std_var.reshape(-1, meta.nreads, std_var.shape[1])
    time = time.reshape(-1, meta.nreads)

    # Convert to masked arrays to safely sum reads
    flux = np.ma.masked_array(flux, mask=optmask)
    err = np.ma.masked_array(err, mask=optmask)
    std_flux = np.ma.masked_array(std_flux, mask=optmask)
    std_var = np.ma.masked_array(std_var, mask=optmask)
    time = np.ma.masked_array(time, mask=np.all(optmask, axis=2))

    # Sum together the reads to get (nfiles, nwaves)
    flux = flux.mean(axis=1)*meta.nreads
    std_flux = std_flux.mean(axis=1)*meta.nreads

    # Rescale uncertainties by the same factor as the surviving flux sum.
    # Mask zero counts so fully masked channels remain masked.
    n_good = np.ma.masked_equal(np.sum(~optmask, axis=1), 0)
    scale = meta.nreads / n_good
    err = np.ma.sqrt(np.ma.sum(err**2, axis=1))*scale
    # stdvar already contains variances, which scale as flux squared.
    std_var = np.ma.sum(std_var, axis=1)*scale**2

    # Mask if all of the reads were masked
    optmask = np.all(optmask, axis=1)

    # Average together the reads' times
    time = time.mean(axis=1)

    # Take every nread value out of the spec and lc object
    spec = spec.isel(time=np.arange(0, len(spec.optspec.time),
                                    meta.nreads))
    lc = lc.isel(time=np.arange(0, len(lc.data.time),
                                meta.nreads))

    # Replace the shared Dataset coordinates with the averaged read times.
    # Assigning a coordinate through an individual DataArray does not update
    # the parent Dataset's coordinate.
    time = np.ma.filled(time, np.nan)
    spec = spec.assign_coords(time=spec.time.copy(data=time))
    lc = lc.assign_coords(time=lc.time.copy(data=time))

    # Update values based on those we've calculated above
    spec.optspec.values = flux
    spec.opterr.values = err
    spec.optmask.values = optmask
    spec.stdspec.values = std_flux
    spec.stdvar.values = std_var

    # Update meta parameters
    meta.nreads = 1
    meta.n_int = len(time)

    return spec, lc, meta
