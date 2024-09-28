import numpy as np
from . import disk as di
from . import meanerr as me
from . import interp2d as i2d


def apphot(meta, image, ctr, photap, skyin, skyout, betahw, targpos,
           mask=None, imerr=None, skyfrac=0.0,
           med=False, nochecks=False,
           expand=1, order=1,
           aperr=False, nappix=False, skylev=False, skyerr=False,
           nskypix=False, nskyideal=False, status=False, isbeta=False,
           betaper=False, aperture_shape=None):
    """
    Perform aperture photometry on the input image.

    Parameters
    ----------
    image : 2D ndimage
        Float array containing object to measure.
    ctr : 2 elements tuple
        x,y location of object's center.
    photap : Scalar
        Size of photometry apperture in pixels.
    skyin : Scalar
        Inner sky annulus edge, in pixels.
    skyout : Scalar
        Outer sky annulus edge, in pixels.
    betahw : Scalar
        Half-width of box size around centroid for beta calculation.
    targpos : 2 elements tuple
        x,y location of object's center calculated from mean image.
    mask : 2D ndimage, boolean
        Boolean array giving status of corresponding pixel in
        Image: bad pixel=True, good pixel=False.  Default: only non-finite
        values are masked. Same shape as image.
    imerr : 2D ndimage
        Error estimate for each pixel in the image.  Suggest
        sqrt(image/flat/gain+rdnoise^2), with proper adjustment
        if a sky, dark, or bias frame has been removed. Same
        shape as image.
    skyfrac : Scalar
        Minimum fraction of sky pixels required to be good.
        Must be in range 0 < skyfrac < 1.
    med : Boolean
        If True use median rather than mean in sky level estimation.
    nochecks : Boolean
        Set to True to skip checks of input sanity.
    expand : Integer scalar
        Positive integer factor by which to blow up image, for
        more accurate aperture arithmetic.  If expand=5, each
        pixel becomes a 5x5 block of pixels.  If the pixel is
        on the edge of an aperture or annulus radius, some of
        the 5x5 block will be counted and some will not.
    order : Integer scalar
        Set order to 0 to do nearest-neighbor interpolation if expand.
        Default: 1, bilinear interpolation.
    aperr : Boolean
        Set to True to return flux error.
    nappix : Boolean
        Set to True to return number of total pixels in aperture.
    skylev : Boolean
        Set to True to return the sky level.
    skyerr : Boolean
        Set to True to return the error in the sky level.
    nskypix : boolean
        Set to True to return the number of good pixels in sky annulus.
    nskyideal: Boolean
             Set to True to return the number of pixels that should
             be in sky annulus.
    status : Boolean
        Set to True to return a status flag.
        If status = 0, result is good.  Bits:
        0 = there are NaN(s) in the photometry aperture
        1 = there are masked pixel(s) in the photometry aperture
        2 = the aperture is off the edge of the image
        3 = a fraction less than skyfrac of the sky annulus pixels
        is in the image and not masked
    isbeta : boolean
        If True photometric extraction aperture scales with noise pixel
        parameter (beta).
    betaper  : Scalar
        Returns aperture size used for beta.
    aperture_shape : String
        Specifies shape of the extraction aperture used, currently
        only "circle" and "hexagon" are supported.

    Returns
    -------
    This function returns the flux within Photap pixels of
    [Cx,Cy], after subtracting the average per-pixel flux in the
    sky annulus.  See POCEDURE for details of the calculation.

    Notes
    -----
    The sky level is the mean, error-weighted mean (if errors are
    present), or median (if /med is set) of the non-masked Image
    pixels in the sky annulus.  NaN values and values whose errors
    are zero (for the error-weighted mean) are not included in the
    average.  No flagging is done if these values are found in the
    sky annulus.  SKYERR is the error in the mean, even for the
    median calculation.  Errors in the median can only be estimated
    using the compute-intensive bootstrap Monte-Carlo method.

    The sky value is subtracted from the Image.  The photometric
    flux is then the total of Image pixels in the aperture, whether
    in the Mask or not.  NaN values are not included in the
    calculation, and set the STATUS flag.  It would be much better
    for the user to pass an interpolated image instead.

    For expansion, it is recommended to use bilinear
    interpolation, which is flux-conserving:

    sz = [50, 50]
    expand = 4
    a = dblarr(sz)
    a[25, 25] = 1
    b = rebin(a, expand * sz) / expand^2
    print, total(b)
    1.0000000
    a[25, 26] = 1
    b = rebin(a, expand * sz) / expand^2
    print, total(b)
    2.0000000
    a[26, 26] = 3
    b = rebin(a, expand * sz) / expand^2
    print, total(b)
    5.0000000

    Of course, pixels on the high-indexed edge will not follow that.
    Neither will integer-arithmetic images, particularly at low
    integer values (such as masks).

    If either the entire sky annulus or the entire aperture is bad
    or filled with NaNs, the function sets a flag and returns NaN
    for all incalculable values.

    History:

    - 27-02-2004: Joseph Harrington, Cornell. jh@oobleck.astro.cornell.edu
        Initial version
    - 18-03-2004: jh
        Added nochecks keyword.
    - 19-03-2004: jh
        Added error calculation.
    - 13-01-2005: jh
        Fixed header comment.  Added NAN keyword.
    - 14-10-2005: jh
        Found and fixed major bug in sky mask
        calculation (needed parens around
        subtraction next to mask multiplication).
        Added skyfrac.
    - 07-11-2005: shl35
        Added STATUS keyword, error-weighted sky mean.
    - 16-11-2005: jh
        Rewrote, using meanerr.  Fixed major bug in
        error calc.  Added scaling and test cases.
    - 24-11-2005: jh
        Return NAPPIX, NSKYPIX, NSKYIDEAL (all renamed).
    - 30-11-2005: jh
        Changed NAPPIX, NSKYPIX, NSKYIDEAL to give
        fractional, unexpanded pixels.
    - 21-07-2010: patricio
        Converted to python.
    - 2024-06-05: Yoni Brande, jbrande@ku.edu
        Added ability for non-circular apertures with aperture_shape
        parameter. Currently supporting hexagonal apertures for
        eureka!

    Examples
    --------
    This being one of my most important routines, and being also
    complex, the following examples are also its test suite.  Any
    changes should produce exactly the same numerical results.
    These tests should also be available in the accompanying file
    apphottest.pro.


    .. highlight:: python
    .. code-block:: python

        >>> import sys
        >>> sys.path.append('/home/esp01/code/python/photpipe/lib/')
        >>> import apphot as ap
        >>> import myasym as asy

        >>> test      = 0
        >>> ntest     = 11
        >>> testout   = np.zeros((5, ntest))
        >>> testright = np.zeros((5, ntest))
        >>> sz        = [50, 50]
        >>> sig       = 3.0, 3.0
        >>> ctr       = 25.8, 25.2
        >>> photap    = 12
        >>> skyin     = 12
        >>> skyout    = 15
        >>> ampl      = 1000.0
        >>> sky       = 100.0

        >>> # height that make integral equal to ampl
        >>> h = ampl/(2*np.pi*sig[0]*sig[1])

        >>> f = asy.gaussian(h, ctr[0], ctr[1], sig[0], sig[1])
        >>> r,l = np.indices(sz)
        >>> image = f(r,l) + sky

        >>> plt.figure(1,(9,7))
        >>> plt.clf()
        >>> plt.title('Gaussian')
        >>> plt.xlabel('X coordinate')
        >>> plt.ylabel('Y coordinate')
        >>> plt.pcolor(r, l, image, cmap=plt.cm.gray)
        >>> plt.axis([0, sz[1]-1, 0, sz[0]-1])
        >>> plt.colorbar()
        >>> plt.show()

        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout,
        >>>                    aperr=True, skylev=True, skyerr=True,
        >>>                    status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # A bit of the Gaussian leaks from aperture to sky, rest is right.

        >>> mask = np.zeros(sz, bool)
        >>> mask[24,24] = True
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout, mask=mask,
        >>> aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, 0, skylev, 0, status]
        >>> test += 1
        >>> # We use the bad value since it's in the aperture, but we flag it.

        >>> image[25,24] = np.nan
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout, mask=mask,
        >>> aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, 0, skylev, 0, status]
        >>> test += 1
        >>> # We can't use a NaN! Flagged, and value changes.
        >>> # Bad value still flagged.

        >>> ctr2 = [48.8, 48.2]
        >>> f = asy.gaussian(h, ctr2[0], ctr2[1], sig[0], sig[1])
        >>> image2 = f(r,l) + sky

        >>> plt.figure(2,(9,7))
        >>> plt.clf()
        >>> plt.title('Gaussian')
        >>> plt.xlabel('X coordinate')
        >>> plt.ylabel('Y coordinate')
        >>> plt.pcolor(r, l, image2, cmap=plt.cm.gray)
        >>> plt.axis([0, sz[1]-1, 0, sz[0]-1])
        >>> plt.colorbar()
        >>> plt.show()

        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image2, ctr2, photap, skyin, skyout, mask=mask,
        >>> aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, 0, skylev, 0, status]
        >>> test += 1
        >>> # Flagged that we're off the image.

        >>> skyfrac = 0.5
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image2, ctr2, photap, skyin, skyout,
        >>> mask=mask, skyfrac=skyfrac,
        >>> aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, 0, skylev, 0, status]
        >>> test += 1
        >>> # Flagged that we are off the image and have insufficient sky.
        >>> # Same numbers.

        >>> f = asy.gaussian(h, ctr[0], ctr[1], sig[0], sig[1])
        >>> image = f(r,l) + sky
        >>> imerr = np.sqrt(image)
        >>> mask = np.zeros(sz, bool)

        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout, mask=mask,
        >>>                   imerr=imerr, aperr=True, skylev=True,
        >>>                   skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # Estimates for errors above.  Basic numbers don't change.

        >>> imerr[25, 38] = 0
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout, mask=mask,
        >>> imerr=imerr, aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # The zero-error pixel is ignored in the sky average.
        >>> # Small changes result.

        >>> imerr[25, 38] = np.nan
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout, mask=mask,
        >>> imerr=imerr, aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # The NaN in the sky error is ignored, with the same result.

        >>> image[25, 38] = np.nan
        >>> imerr = sqrt(image)
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout, mask=mask,
        >>> imerr=imerr, aperr=True, skylev=True, skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # The NaN in the sky data is ignored, with the same result.
        >>> # FINDME: my aplev is changing

        >>> ##
        >>> f = asy.gaussian(h, ctr[0], ctr[1], sig[0], sig[1])
        >>> image = f(r,l) + sky

        >>> imerr  = sqrt(image)
        >>> expand = 5
        >>> order  = 0 # sample = 1

        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout,
        >>>                    expand=expand, mask=mask, imerr=imerr,
        >>>                    order=order, aperr=True, skylev=True,
        >>>                    skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # Slight changes.

        >>> expand = 5
        >>> order  = 1  # IDL sample = 0
        >>> aplev, aperr, skylev, skyerr, \
        >>> status = ap.apphot(image, ctr, photap, skyin, skyout,
        >>>                    expand=expand, mask=mask, imerr=imerr,
        >>>                    order=order, aperr=True, skylev=True,
        >>>                    skyerr=True, status=True)

        >>> print(aplev, aperr, skylev, skyerr, status)
        >>> testout  [:, test] = [aplev, aperr, skylev, skyerr, status]
        >>> test += 1
        >>> # Slight changes.  Why the flag?

        >>> # skyerr estimate
        >>> skyerrest = np.sqrt(sky/(np.pi * (skyout**2 - skyin**2)))
        >>> #      0.62687732
        >>> print('Correct:')
        >>> print([ampl,
        >>>        # aperr estim.
        >>>        np.sqrt(ampl + np.pi * photap**2 * (sky+skyerrest**2)),
        >>>        # Note that background flux of 100 contributes a lot!
        >>>        #       215.44538
        >>>        sky, skyerrest, 0])
        >>> print('Test results:')
        >>> print(testout)
        >>> print('Correct results:')
        >>> print(testright)
        >>> print('Differences:')
        >>> print(testout - testright)
    """

    # tini = time.time()

    # Returned values (include aperture size)
    retidx = [True, aperr, nappix, skylev,
              skyerr, nskypix, nskyideal, status, betaper]
    # indexes
    # consider return the whole list always
    # FINDME: use iaplev etc ... or use a dicctionary
    aplev, aperr, nappix, skylev, \
        skyerr, nskypix, nskyideal, stat, betaper = np.arange(9)
    ret = np.zeros(9)  # changed from 8 to 9
    ret[:] = np.nan

    # set error status to 'good' = 0
    ret[stat] = 0
    status = 0

    # bit flag definitions
    statnan = 2 ** 0
    statbad = 2 ** 1
    statap = 2 ** 2
    statsky = 2 ** 3

    # internal skyfrac, so we don't set it in caller
    iskyfrac = skyfrac

    # Check inputs
    sz = np.shape(image)

    if not nochecks:
        if np.ndim(image) != 2:
            # FINDME: raise exceptions instead
            print('image must be a 2D array')
            # break
            return ret[np.where(retidx)]

        # Default mask: only non-finite values are bad
        if mask is None:
            mask = ~np.isfinite(image)

        if np.ndim(mask) != 2:
            print('mask must be a 2D array')
            return ret[np.where(retidx)]

        if (np.shape(image) != np.shape(mask)):
            print('image and mask sizes differ')
            return ret[np.where(retidx)]

        if imerr is not None:
            if (np.shape(image) != np.shape(imerr)):
                print('image and imerr sizes differ')
                return ret[np.where(retidx)]

        if (iskyfrac < 0.0) or (iskyfrac > 1.0):
            print('skyfrac must be in range [0,1]')
            return ret[np.where(retidx)]

        if expand != np.compat.long(expand) or expand < 1:
            print('invalid expand')
            return ret[np.where(retidx)]

    # Expand
    iexpand = int(expand)
    isz = np.array(sz, dtype=int)+(np.array(sz, dtype=int)-1)*(iexpand-1)
    ictr = iexpand * np.array(ctr)
    iphotap = iexpand * photap
    iskyin = iexpand * skyin
    iskyout = iexpand * skyout

    y, x = np.arange(sz[0]), np.arange(sz[1])
    yi, xi = np.linspace(0, sz[0]-1, isz[0]), np.linspace(0, sz[1]-1, isz[1])
    iimage = i2d.interp2d(image, expand=iexpand, y=y, x=x, yi=yi, xi=xi)
    imask = i2d.interp2d(mask.astype(float), expand=iexpand, y=y, x=x,
                         yi=yi, xi=xi)
    # Need to convert fractions to booleans
    imask = imask > 0.5
    if imerr is not None:
        iimerr = i2d.interp2d(imerr, expand=iexpand, y=y, x=x, yi=yi, xi=xi)

    # Specify aperture shape function
    if aperture_shape == "hexagon":
        apFunc = di.hex
    else:
        apFunc = di.disk

    # SKY
    # make sky annulus mask (True outside annulus, False inside annulus)
    skyann = ~np.logical_and(apFunc(iskyout, ictr, isz),
                             apFunc(iskyin, ictr, isz))

    # combine masks to mask all bad pixels and pixels outside annulus
    skymask = skyann | imask | ~np.isfinite(iimage)  # flag NaNs to eliminate
    # from nskypix

    # Check for skyfrac violation
    # FINDME: include NaNs and zero errors
    ret[nskypix] = np.sum(~skymask) / iexpand ** 2.0
    szsky = (int(np.ceil(iskyout)) * 2 + 3) * np.array([1, 1], dtype=int)
    ctrsky = (ictr % 1.0) + np.ceil(iskyout) + 1.0
    # nskyideal = all pixels in sky
    ret[nskyideal] = (np.sum(
                      ~np.logical_and(apFunc(iskyout, ctrsky, szsky),
                                      apFunc(iskyin, ctrsky, szsky)))
                      / iexpand**2.0)

    if ret[nskypix] < iskyfrac * ret[nskyideal]:
        status |= statsky

    if ret[nskypix] == 0:  # no good pixels in sky?
        status |= statsky
        ret[stat] = status
        print('no good pixels in sky')
        return ret[np.where(retidx)]

    # Calculate the sky and sky error values:
    # Ignore the status flag from meanerr, it will skip bad
    # values intelligently.
    if med:  # Do median sky
        iimage_temp = np.ma.masked_where(skymask, iimage)
        ret[skylev] = np.ma.median(iimage_temp)
        if imerr is not None:
            # FINDME: We compute the standard deviation of the mean, not the
            # median.  The standard deviation of the median is complicated and
            # can only be estimated statistically using the bootstrap method.
            # It's also very computationally expensive. We could alternatively
            # use the maximum of the standard deviation of the mean and the
            # mean separation of values in the middle of the distribution.
            dummy, ret[skyerr] = me.meanerr(iimage, iimerr, mask=skymask,
                                            err=True)
            # Expand correction. Since repeats are correlated,
            ret[skyerr] *= iexpand
            # error in mean was improved by sqrt(iexpand^2).
    else:  # Do mean
        if imerr is not None:
            ret[skylev], ret[skyerr] = me.meanerr(iimage, iimerr,
                                                  mask=skymask, err=True)
            ret[skyerr] *= iexpand  # Expand correction.
        else:
            iimage_temp = np.ma.masked_where(skymask, iimage)
            ret[skylev] = np.ma.mean(iimage)

    if meta.skip_apphot_bg:
        ret[skylev] = np.zeros_like(ret[skylev])

    # Calculate Beta values. If True photometric extraction aperture scales
    # with noise pixel parameter (beta).
    if isbeta == 1 and betahw > 0:
        # Using target position from mean image
        ctr_y = int(targpos[1])
        ctr_x = int(targpos[0])
        betahw = int(betahw)

        # Create a box of width and length (betahw) around the target position
        betabox = image[ctr_y-betahw:ctr_y+betahw+1,
                        ctr_x-betahw:ctr_x+betahw+1]

        # Subtract the background
        betabox -= ret[skylev]

        # beta = sum(I(i))^2 / sum(I(i)^2) see details in link describing the
        # noise pixels: https://irachpp.spitzer.caltech.edu/page/noisepix
        beta = np.sum(betabox) ** 2 / np.sum(betabox ** 2)

        iphotap += iexpand * (np.sqrt(beta) + photap)

        # Return aperture size used for beta.
        ret[betaper] = iphotap

    elif betahw == 0:
        raise ValueError("Could not evalaute beta. Please update POET "
                         "photom.pcf to betahw > 0.")

    # APERTURE
    # make aperture mask, extract data and mask
    apmask, dstatus = apFunc(iphotap, ictr, isz, status=True)
    if dstatus:  # is the aperture fully on the image?
        status |= statap

    aploc = np.where(~apmask)  # report number of pixels in aperture
    # make it unexpended pixels
    ret[nappix] = np.sum(~apmask[aploc])/iexpand**2.0
    if ret[nappix] == 0:  # is there *any* good aperture?
        status |= statbad
        ret[stat] = status
        return ret[np.where(retidx)]

    if np.all(~np.isfinite(iimage[aploc])):  # all aperture pixels are NaN?
        status |= statnan
        ret[stat] = status
        return ret[np.where(retidx)]

    # subtract sky here to get a flag if it's NaN
    apdat = iimage[aploc] - ret[skylev]
    apmsk = imask[aploc]

    # flag NaNs and bad pixels
    goodies = ~np.isfinite(apdat)
    if np.any(goodies):
        status |= statnan

    if np.any(apmsk):
        status |= statbad

    # PHOTOMETRY
    # Do NOT multiply by bad pixel mask!  We need to use the interpolated
    # pixels here, unfortunately.
    ret[aplev] = np.sum(apdat[~goodies])

    # Expand correction.  We overcount by iexpand^2.
    ret[aplev] /= iexpand ** 2.0

    # if we have uncertainties...
    if imerr is not None:
        # Flag NaNs.  Zeros are ok, if weird.
        apunc = iimerr[aploc]
        apuncloc = np.isfinite(apunc)
        if not np.all(apuncloc):
            status |= statnan

        # Multiply by mask for the aperture error calc.  The error on a
        # replaced pixel is unknown.  In one sense, it's infinite.  In
        # another, it's zero, or should be close.  So, ignore those points.
        # Sky error still contributes.
        apunc[np.where(apuncloc == 0)] = 0
        ret[aperr] = np.sqrt(np.sum(~apmsk * apunc ** 2.0) +
                             np.size(aploc) * ret[skyerr] ** 2.0)

        # Expand correction.  We overcount by iexpand^2, but that's
        # inside sqrt:
        # sqrt(sum(iexpand^2 * (indep errs))), so div. by iexpand
        ret[aperr] /= iexpand

    ret[stat] = status
    # ttotal =  time.time() - tini

    return ret[np.where(retidx)]


def apphot_status(data):
    """
    Prints a warning if aperture step had errors.
    Bit flag definitions from the apphot function:

        | statnan = 2 ** 0
        | statbad = 2 ** 1
        | statap = 2 ** 2
        | statsky = 2 ** 3
        | E.g., If the flag is 6 then is was created by a flag in
        | statap and statbad as 2 ** 2 + 2 ** 1 = 6.
        | This function is converting the flags back to binary
        | and checking which flags were triggered.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    """
    if sum(data.status != 0) > 0:
        unique_flags = np.unique(data.status)
        unique_flags_binary = ['{:08b}'.format(int(i)) for i in unique_flags]
        for binary_flag in unique_flags_binary:
            print('A warning by the aperture photometry routine:')
            if binary_flag[-1] == '1':
                print('There are NaN(s) in the photometry aperture')
            if binary_flag[-2] == '1':
                print('There are masked pixel(s) in the photometry aperture')
            if binary_flag[-3] == '1':
                print('The aperture is off the edge of the image')
            if binary_flag[-4] == '1':
                print('A fraction less than skyfrac of the sky annulus '
                      'pixels is in the image and not masked')
