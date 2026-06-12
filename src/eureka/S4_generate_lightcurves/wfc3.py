import numpy as np


def sum_reads(spec, lc, meta):
    """Sum together the non-destructive reads from each file to reduce noise
    and data volume.

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

    # Sum together the reads to get (nfiles, nwaves)
    flux = flux.sum(axis=1)
    std_flux = std_flux.sum(axis=1)

    # Add errors in quadrature
    err = np.sqrt(np.sum(err**2, axis=1))
    std_var = np.sqrt(np.sum(std_var**2, axis=1))

    # Mask if any of the reads were masked
    optmask = np.any(optmask, axis=1)

    # Average together the reads' times
    time = time.mean(axis=1)

    # Take every nread value out of the spec and lc object
    spec = spec.isel(time=np.arange(0, len(spec.optspec.time),
                                    meta.nreads))
    lc = lc.isel(time=np.arange(0, len(lc.data.time),
                                meta.nreads))

    # Update values based on those we've calculated above
    spec.optspec.values = flux
    spec.optspec['time'] = time
    spec.opterr.values = err
    spec.opterr['time'] = time
    spec.optmask.values = optmask
    spec.optmask['time'] = time
    spec.stdspec.values = std_flux
    spec.stdspec['time'] = time
    spec.stdvar.values = std_var
    spec.stdvar['time'] = time

    # For these values, just use the first read's value
    spec.centroid_x['time'] = time
    spec.centroid_y['time'] = time
    spec.guess['time'] = time
    spec.scandir['time'] = time
    lc.centroid_x['time'] = time
    if hasattr(lc, 'centroid_sx'):
        lc.centroid_sx['time'] = time
    lc.centroid_y['time'] = time
    lc.scandir['time'] = time
    if meta.correctDrift:
        lc.driftmask['time'] = time

    # Update meta parameters
    meta.nreads = 1
    meta.n_int = len(time)

    return spec, lc, meta
