import numpy as np
from ..S3_data_reduction import sigrej as sr


def chunkbad(meta, data, uncd, mask, nimpos, sigma, szchunk, fp,
             minchunk=None):
    """
    This function searches for bad pixels in Spitzer Space
    Telescope Infrared Array Camera subarray mode data,
    flags them, and replaces the bad data values with reasonable
    ones.  It should be run after sstphot_badmask (without its SIGMA
    parameter defined) and instead of sstphot_badfix.pro.

    Parameters
    ----------
    data    : 4D ndarray
            This array contains the data to be analyzed. Of shape
            [nim,nx,ny,npos], nx and ny are the image dimensions,
            nim is the maximum number of images in the largest set,
            and npos is the number of sets (or 'positions').

    uncd    : float ndarray
            uncertainties of corresponding points in data, same
            shape as data.

    mask    : byte ndarray
            mask where 1 indicates the corresponding pixel in data
            is good and 0 indicates it is bad. same shape as data.

    nimpos  : [npos] array giving the number of good images at each
            photometry position.

    sigma   : Passed to sigrej (see sigrej documentation).

    szchunk : The number of images in a processing chunk,
            usually equal to the number in a subarray-mode readout
            set.

    fp      : frame parameters variable.

    minchunk: Minimum number of images to allow in the last
            chunk.  Default is 30.

    Returns
    -------
    This function returns an [nx, ny, npos] array giving the mean
    images made from good pixels in each photometry position.


    Notes
    -----
    Flagging: For each block of szchunk images, it subtracts the
    median image value from each image and does sigma rejection in
    each pixel position (i.e., on
    Data[i, j, nchunk*k : nchunk*k + szchunk - 1, pos]), flagging
    any outliers and recording them appropriately in the INSIGREJ
    column of FP.  It replaces flagged pixels with the median for
    that location within the chunk, adding back on the original
    frame median.  If all pixels are flagged in that location
    within the chunk, it uses just the frame median.

    Mean images: For each pixel position in Data (i.e., Data[i,j,*,pos])
    it calculates the mean of pixels that are not flagged as bad.

    History:

    Written by:	Joseph Harrington, Cornell. 24-11-2005
                     jh@oobleck.astro.cornell.edu
    29-11-2005 jh       Filled in header.
    29-05-2007 jh       Handle data with non-N*64 size.
    29-05-2007 khorning Handle data with only 3 dimensions
    14-06-2007 khorning Moved chuck size to within the loop
    16-07-2007 jh       Ensured all loop variables are longs, not ints.
    13-04-2008 ks       Changed message to print (line 134)
    13-04-2008 ks       Converted long to string in print (line 134)
    20-07-2010 patricio Converted to python.
    """

    # Sizes
    n_int, ny, nx = meta.n_int, meta.ny, meta.nx
    # szchunk = n_int

    # Allocation
    meanim = np.zeros((ny, nx))
    skysub = np.zeros((ny, nx))

    # Median sky
    fp.medsky = np.zeros((n_int))
    # Our rejects
    fp.nsigrej = np.zeros((n_int))

    # High value constant
    highval = 1e100

    # Default minimum chunk size
    if minchunk is None:
        minchunk = 30

    # Initialize a Timer to report progress
    # clock = t.Timer(np.sum(nimpos) * 1.0 / szchunk)

    for pos in np.arange(1):
        totdat = np.zeros((ny, nx))
        totmsk = np.zeros((ny, nx))
        # chunk sizes
        iszchunk = szchunk
        nchunk = np.int(np.ceil(np.double(nimpos[pos]) / iszchunk))

        for chunk in np.arange(nchunk):
            #      print('chunk: ' + str(chunk) + time.strftime(' %a %b %d %H:%M:%S %Z %Y'))
            start = chunk * szchunk

            # If its the last chunk size, truncate it
            if (nimpos[pos] - start) < iszchunk:
                iszchunk = nimpos[pos] - start
                if iszchunk <= np.amax(minchunk):
                    print('poet_chunkbad: Final chunk is too small: %i' % iszchunk)

            # Subtract approximate sky for every image in chunk
            for k in np.arange(iszchunk):
                fp.medsky[start + k, pos] = np.median(
                    (data[start + k, :, :, pos])[np.where(mask[start + k, :, :, pos])])
                if np.isfinite(fp.medsky[start + k, pos]):
                    skysub[k, :, :] = data[start + k, :, :, pos] - fp.medsky[start + k, pos]

            # Do sigma rejection within the chunk, replace that and pre-masked stuff
            # it's ok if there are pre-flagged data in the sigma rejection input
            keepmsk, fmean = sr.sigrej(skysub[0:iszchunk, :, :], sigma,
                                       mask=mask[start:start + iszchunk, :, :, pos],
                                       fmean=True)
            keepmsk = keepmsk & mask[start:start + iszchunk, :, :, pos]

            # Counts of badpixels
            count = np.sum(keepmsk, axis=0)
            fmean[np.where(count == 0)] = 0.0  # handle full row rejection

            # Update fp.nsigrej
            countperframe = np.sum(np.sum(1 - keepmsk, axis=1), axis=1)
            fp.nsigrej[start:start + iszchunk, pos] += countperframe

            # Fix and flag bad data:

            # Indexes
            rejloc = np.where(keepmsk == 0)  # loc. in chunk's mask
            rejec = (start + rejloc[0], rejloc[1], rejloc[2])  # loc. in data

            # broadcasts fmean(ny, nx) with fp.medsky(izchunk,1,1) at the bad
            # pixel positions(rejloc), and assigns that values to the data in rejec
            (data[:, :, :, pos])[rejec] = (fmean +
                                           fp.medsky[start:start + iszchunk,
                                                     pos].reshape(iszchunk, 1, 1))[rejloc]
            (uncd[:, :, :, pos])[rejec] = highval
            (mask[:, :, :, pos])[rejec] = 0

            # Accumulate data for mean image for this position.
            # Based on stackmaskmean.pro, but done this way to get median subtraction.
            subdata = np.copy(data[start:start + iszchunk, :, :, pos])
            subdata[np.where(np.isfinite(subdata) == 0)] = 0
            submask = mask[start:start + iszchunk, :, :, pos]

            totdat += np.sum(subdata * submask, 0)
            totmsk += np.sum(submask, 0)

            # Report progress
            # clock.check(np.sum(nimpos[0:pos]) * 1.0 / szchunk + chunk)

        # calculate mean image with zero background
        totmsk[np.where(totmsk == 0)] = 1.0
        meant = totdat / totmsk
        meant[np.where(totmsk == 0)] = 0.0  # assume all-bad pixels are
        # background (else what?)
        meanim[:, :, pos] = meant

    # Nsigrej now holds all the bad pixels.  How many are new?
    fp.nsigrej = np.transpose(fp.nsigrej)
    fp.nsigrej = fp.nsigrej - fp.nsstrej - fp.userrej
    fp.medsky = np.transpose(fp.medsky)

    return meanim
