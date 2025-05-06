import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
from functools import partial
from photutils.aperture import (aperture_photometry, CircularAperture,
                                EllipticalAperture, RectangularAperture,
                                CircularAnnulus, EllipticalAnnulus,
                                RectangularAnnulus)
from . import disk as di
from . import meanerr as me
from . import interp2d as i2d


def apphot(data, meta, i):
    """
    Perform aperture photometric extraction and annular sky subtraction on the
    input image using code from POET.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the photometry data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The current integration number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object containing the extracted photometry data.
    """
    image = data.flux.values[i]
    mask = data.mask.values[i]
    imerr = data.err.values[i]
    # POET apphot code requires (y,x) coordinates instead of (x,y)
    if meta.moving_centroid:
        ctr = [data.centroid_y.values[i],
               data.centroid_x.values[i]]
    else:
        ctr = [np.median(data.centroid_y.values),
               np.median(data.centroid_x.values)]

    # set initial error status to 'good' = 0
    status = 0
    # bit flag definitions
    statnan = 2 ** 0
    statbad = 2 ** 1
    statap = 2 ** 2
    statsky = 2 ** 3

    # Expand
    sz = np.shape(image)
    isz = np.array(sz, dtype=int)+(np.array(sz, dtype=int)-1)*(meta.expand-1)
    ictr = meta.expand*np.array(ctr)
    iphotap = meta.expand*meta.photap
    iskyin = meta.expand*meta.skyin
    iskyout = meta.expand*meta.skyout

    y, x = np.arange(sz[0]), np.arange(sz[1])
    yi, xi = np.linspace(0, sz[0]-1, isz[0]), np.linspace(0, sz[1]-1, isz[1])
    iimage = i2d.interp2d(image, expand=meta.expand, y=y, x=x, yi=yi, xi=xi)
    iimerr = i2d.interp2d(imerr, expand=meta.expand, y=y, x=x, yi=yi, xi=xi)
    imask = i2d.interp2d(mask.astype(float), expand=meta.expand, y=y, x=x,
                         yi=yi, xi=xi)
    imask = imask > 0.5  # Need to convert fractions back to booleans

    # Specify aperture shape function
    if meta.aperture_shape == "hexagon":
        apFunc = di.hex
    elif meta.aperture_shape == "circle":
        apFunc = di.disk

    # SKY
    # make sky annulus mask (True outside annulus, False inside annulus)
    skyann = ~np.logical_and(apFunc(iskyout, ictr, isz),
                             apFunc(iskyin, ictr, isz))

    # combine masks to mask all bad pixels and pixels outside annulus
    # flag NaNs to eliminate them and any pixels with zero or negative errors
    skymask = skyann | imask | ~np.isfinite(iimage) | (iimerr <= 0)
    # from nskypix

    # Check for skyfrac violation
    nskypix = np.sum(~skymask)/meta.expand**2
    szsky = (int(np.ceil(iskyout))*2+3)*np.array([1, 1], dtype=int)
    ctrsky = (ictr % 1)+np.ceil(iskyout)+1
    # nskyideal = all pixels in sky
    nskyideal = (np.sum(~np.logical_and(
        apFunc(iskyout, ctrsky, szsky), apFunc(iskyin, ctrsky, szsky)))
        / meta.expand**2)

    if nskypix == 0:  # no good pixels in sky?
        raise ValueError('There were no good pixels in the sky annulus')
    elif nskypix < meta.minskyfrac*nskyideal:
        status |= statsky

    # Calculate the sky and sky error values:
    # Ignore the status flag from meanerr, it will skip bad
    # values intelligently.
    if meta.skip_apphot_bg:
        skylev = 0
        skyerr = 0
    elif meta.bg_method == 'median':
        iimage_temp = np.ma.masked_where(skymask, iimage)
        skylev = np.ma.median(iimage_temp)
        # FINDME: We compute the standard deviation of the mean, not the
        # median.  The standard deviation of the median is complicated and
        # can only be estimated statistically using the bootstrap method.
        # It's also very computationally expensive. We could alternatively
        # use the maximum of the standard deviation of the mean and the
        # mean separation of values in the middle of the distribution.
        _, skyerr = me.meanerr(iimage, iimerr, mask=skymask, err=True)
        # Expand correction. Since repeats are correlated,
        skyerr *= meta.expand
    elif meta.bg_method == 'mean':
        skylev, skyerr = me.meanerr(iimage, iimerr, mask=skymask, err=True)
        # Expand correction. Since repeats are correlated,
        skyerr *= meta.expand
    else:
        raise ValueError(f'Unknown bg_method "{meta.bg_method}"')

    if meta.betahw is not None:
        # Calculate beta values and scale photometric extraction aperture
        # with noise pixel parameter (beta).

        ctr_y = int(ctr[0])
        ctr_x = int(ctr[1])
        betahw = int(meta.betahw)

        # Create a box of width and length (betahw) around the target position
        betabox = image[ctr_y-betahw:ctr_y+betahw+1,
                        ctr_x-betahw:ctr_x+betahw+1]

        # Subtract the background
        betabox -= skylev

        # beta = sum(I(i))^2 / sum(I(i)^2) see details in link describing the
        # noise pixels: https://irachpp.spitzer.caltech.edu/page/noisepix
        beta = np.sum(betabox)**2/np.sum(betabox**2)

        iphotap += meta.expand*(np.sqrt(beta)+meta.photap)

        # Return aperture size used for beta.
        betaper = iphotap
    else:
        betaper = 0

    # APERTURE
    # make aperture mask, extract data and mask
    apmask, dstatus = apFunc(iphotap, ictr, isz, status=True)
    if dstatus:  # is the aperture fully on the image?
        status |= statap

    aploc = np.where(~apmask)  # report number of pixels in aperture
    # make it unexpended pixels
    nappix = np.sum(~apmask[aploc])/meta.expand**2
    if nappix == 0:
        raise ValueError('All pixels in the source aperture were masked')
    elif np.all(~np.isfinite(iimage[aploc])):  # all aperture pixels are NaN?
        raise ValueError('All pixels in the source aperture are NaN')

    # subtract sky here to get a flag if it's NaN
    apdat = iimage[aploc] - skylev
    apmsk = imask[aploc]

    # flag NaNs and inf pixels
    nanOrInf = ~np.isfinite(apdat)
    if np.any(nanOrInf):
        status |= statnan

    if np.any(apmsk):
        status |= statbad

    # PHOTOMETRY
    # Do NOT multiply by bad pixel mask!  We need to use the interpolated
    # pixels here, unfortunately.
    aplev = np.sum(apdat[~nanOrInf])
    # Expand correction.  We overcount by expand^2.
    aplev /= meta.expand**2

    # Flag NaNs. Zeros are ok, if weird.
    apunc = iimerr[aploc]
    apuncNan = ~np.isfinite(apunc)
    if np.any(apuncNan):
        status |= statnan

    # Multiply by mask for the aperture error calc.  The error on a
    # replaced pixel is unknown.  In one sense, it's infinite.  In
    # another, it's zero, or should be close.  So, ignore those points.
    # Sky error still contributes.
    apunc[apuncNan] = 0
    apunc[apmsk] = 0
    aperr = np.sqrt(np.nansum(apunc**2) + (np.size(aploc)*skyerr**2))

    # Expand correction.  We overcount by expand^2, but that's
    # inside sqrt:
    # sqrt(sum(expand^2 * (indep errs))), so divide by expand
    aperr /= meta.expand

    # Store the results in the data object
    data['aplev'][i] = aplev
    data['aperr'][i] = aperr
    data['nappix'][i] = nappix
    data['skylev'][i] = skylev
    data['skyerr'][i] = skyerr
    data['nskypix'][i] = nskypix
    data['nskyideal'][i] = nskyideal
    data['status'][i] = status
    data['betaper'][i] = betaper

    return data


def apphot_status(data):
    """Prints a warning if aperture step had errors.

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
        print('    A warning by the aperture photometry routine:')
        for binary_flag in unique_flags_binary:
            if binary_flag[-1] == '1':
                print('      There are NaN(s) or infinities in the photometry'
                      ' aperture')
            if binary_flag[-2] == '1':
                print('      There are masked pixel(s) in the photometry '
                      'aperture')
            if binary_flag[-3] == '1':
                print('      The aperture is off the edge of the image')
            if binary_flag[-4] == '1':
                print('      A fraction less than skyfrac of the sky annulus '
                      'pixels is in the image and not masked')


def transform_pixels(x, y, position, a, b=0, theta=0):
    """Transform the pixel coordinates based on an aperture's parameters.

    Parameters
    ----------
    x : ndarray
        Pixel x-coordinates to be transformed.
    y : ndarray
        Pixel y-coordinates to be transformed.
    position : list
        [cx, cy] list for the center of the aperture.
    a : float
        The semi-major axis of the ellipse or rectangle or the radius of
        the circle.
    b : float; optional
        The semi-minor axes of the ellipse or rectangle or the radius of
        the circle. Defaults to 0, in which case b will be set to a.
    theta : float; optional
        Rotation angle in degrees. Defaults to 0.

    Returns
    -------
    x : ndarray
        Transformed pixel x-coordinates.
    y : ndarray
        Transformed pixel y-coordinates.
    """
    theta *= np.pi/180
    cx, cy = position
    if b == 0:
        b = a
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta*(x-cx) + sin_theta*(y-cy)
    y_rot = -sin_theta*(x-cx) + cos_theta*(y-cy)
    return x_rot/a, y_rot/b


def circle_overlap(xpx, ypx, position, a, b=0, theta=0):
    """Helper function for circle and ellipse overlap.

    Parameters
    ----------
    xpx : ndarray
        Pixel x-coordinates.
    ypx : ndarray
        Pixel y-coordinates.
    position : list
        [cx, cy] list for the center of the aperture.
    a : float
        The semi-major axis of the ellipse or the radius of the circle.
    b : float; optional
        The semi-minor axes of the ellipse or the radius of the circle.
        Defaults to 0, in which case b will be set to a.
    theta : float; optional
        Rotation angle of the ellipse in degrees. Defaults to 0.

    Returns
    -------
    fractionalArea : float
        The fractional area of the circle or ellipse that overlaps with the
        pixel.
    """
    xmin, ymin = transform_pixels(xpx-0.5, ypx-0.5, position, a, b, theta)
    xmax, ymax = transform_pixels(xpx+0.5, ypx+0.5, position, a, b, theta)
    r = 1  # We've just scaled the pixels

    pixel_polygon = Polygon([
        (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    circle = Point(0, 0).buffer(r)
    intersection = pixel_polygon.intersection(circle)
    if intersection.is_empty:
        return 0.0
    else:
        pixel_area = (xmax-xmin)*(ymax-ymin)
        return intersection.area/pixel_area


def rectangle_overlap(xpx, ypx, position, a, b=0, theta=0):
    """Helper function for circle and ellipse overlap.

    Parameters
    ----------
    xpx : ndarray
        Pixel x-coordinates.
    ypx : ndarray
        Pixel y-coordinates.
    position : list
        [cx, cy] list for the center of the aperture.
    a : float
        Half the x-axis size of the rectangle.
    b : float; optional
        Half the y-axis size of the rectangle. Defaults to 0, in which case
        b will be set to a.
    theta : float; optional
        Rotation angle of the rectangle in degrees. Defaults to 0.

    Returns
    -------
    fractionalArea : float
        The fractional area of the rectangle that overlaps with the pixel.
    """
    if b == 0:
        b = a

    xmin = xpx - 0.5 - position[0]
    xmax = xpx + 0.5 - position[0]
    ymin = ypx - 0.5 - position[1]
    ymax = ypx + 0.5 - position[1]

    pixel_polygon = Polygon([
        (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    rect_polygon = Polygon([
        (-a, -b), (a, -b), (a, b), (-a, b)])
    rotated_rect = rotate(rect_polygon, theta, origin=(0, 0))
    intersection = pixel_polygon.intersection(rotated_rect)
    if intersection.is_empty:
        return 0.0
    else:
        return intersection.area


def optphot(data, meta, i, saved_photometric_profile):
    """
    Perform optimal photometric extraction and annular sky subtraction on the
    input image.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the photometry data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The current integration number.
    saved_photometric_profile : 2D array
        The saved profile to be used for optimal photometric extraction.
        If None, the profile will be generated from the median flux array.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object containing the extracted photometry data.
    profile : 2D array
        The profile that was used for optimal photometric extraction.
    """
    if meta.moving_centroid:
        position = [data.centroid_x.values[i], data.centroid_y.values[i]]
    else:
        position = [np.median(data.centroid_x.values),
                    np.median(data.centroid_y.values)]

    if saved_photometric_profile is None:
        profile = np.ma.masked_invalid(data.medflux.values)
        xpx = np.arange(profile.shape[1])
        ypx = np.arange(profile.shape[0])
        xpx, ypx = np.meshgrid(xpx, ypx)
        if meta.aperture_edge == 'center':
            xpx_scaled, ypx_scaled = transform_pixels(
                xpx, ypx, position, meta.photap, meta.photap_b,
                meta.photap_theta)
            if meta.aperture_shape in ['circle', 'ellipse']:
                weight = (np.sqrt(xpx_scaled**2+ypx_scaled**2) <= 1
                          ).astype(float)
            elif meta.aperture_shape == 'rectangle':
                weight = ((np.abs(xpx_scaled) <= 1) &
                          (np.abs(ypx_scaled) <= 1)).astype(float)
            else:
                raise ValueError('aperture_shape must be "circle", "ellipse", '
                                 'or "rectangle", but got '
                                 f'{meta.aperture_shape}')
        elif meta.aperture_edge == 'exact':
            if meta.aperture_shape in ['circle', 'ellipse']:
                overlap_func = circle_overlap
            elif meta.aperture_shape == 'rectangle':
                overlap_func = rectangle_overlap
            else:
                raise ValueError('aperture_edge must be "center" or "exact", '
                                 f'but got {meta.aperture_edge}')
            overlap_func = partial(overlap_func, position=position,
                                   a=meta.photap, b=meta.photap_b,
                                   theta=meta.photap_theta)
            weight = np.vectorize(overlap_func)(xpx, ypx)
        else:
            raise ValueError('aperture_edge must be "center" or "exact", '
                             f'but got {meta.aperture_edge}')
        profile *= weight
        # Force positivity
        profile[np.where(profile < 0)] = 0
        # Get normalized error-weighted profile
        with np.errstate(divide='ignore', invalid='ignore'):
            profile /= np.ma.sqrt(profile)
        profile /= np.ma.sum(profile)
    else:
        profile = saved_photometric_profile

    # Grab the current frame and mask
    mask = data.mask.values[i]
    good_pixels = (~mask).astype(float)
    flux = np.ma.masked_where(mask, data.flux.values[i])
    derr = np.ma.masked_where(mask, data.err.values[i])

    # Setup the sky annulus
    if meta.aperture_shape == 'circle':
        sky_annul = CircularAnnulus(position, meta.skyin, meta.skyout)
    elif meta.aperture_shape == 'ellipse':
        # Sky annulus has the same aspect ratio and orientation as source aper
        theta = meta.photap_theta*np.pi/180
        sky_annul = EllipticalAnnulus(position, meta.skyin, meta.skyout,
                                      meta.skyout*(meta.photap_b/meta.photap),
                                      meta.skyin*(meta.photap_b/meta.photap),
                                      theta)
    elif meta.aperture_shape == 'rectangle':
        # Sky annulus has the same aspect ratio and orientation as source aper
        theta = meta.photap_theta*np.pi/180
        sky_annul = RectangularAnnulus(position, meta.skyin, meta.skyout,
                                       meta.skyout*(meta.photap_b/meta.photap),
                                       meta.skyin*(meta.photap_b/meta.photap),
                                       theta)
    else:
        raise ValueError(f'Unknown aperture_shape "{meta.aperture_shape}"')

    # Compute the number of good pixels in object aperture and sky annulus
    if not meta.skip_apphot_bg:
        nskypix = aperture_photometry(good_pixels, sky_annul,
                                      method=meta.aperture_edge
                                      )['aperture_sum'].data[0]
        nskyideal = aperture_photometry(np.ones_like(good_pixels), sky_annul,
                                        method=meta.aperture_edge
                                        )['aperture_sum'].data[0]
    nappix = np.ma.sum(good_pixels * (profile != 0))

    # Compute the total source flux
    aplev = np.ma.sum(flux*profile)/np.sum(good_pixels*profile)*nappix
    aperr = (1/np.sqrt(np.ma.sum(1/derr**2*good_pixels*profile))
             * np.sqrt(nappix/np.sum(good_pixels*profile)))

    if not meta.skip_apphot_bg:
        # Compute the sky flux per pixel
        skytable = aperture_photometry(flux, sky_annul, derr, mask,
                                       method=meta.aperture_edge)
        skylev = skytable['aperture_sum'].data[0]/nskypix
        skyerr = skytable['aperture_sum_err'].data[0]/nskypix

        # Subtract the sky level that fell within the source aperture
        skylev_total = (np.sum(skylev*good_pixels*profile)
                        / np.sum(good_pixels*profile)*nappix)
        aplev -= skylev_total

        # Add the skyerr to the aperr in quadrature
        aperr = np.sqrt(aperr**2 + nappix*skyerr**2)

    # Store the results in the data object
    data['aplev'][i] = aplev
    data['aperr'][i] = aperr
    data['nappix'][i] = nappix
    if not meta.skip_apphot_bg:
        data['skylev'][i] = skylev
        data['skyerr'][i] = skyerr
        data['nskypix'][i] = nskypix
        data['nskyideal'][i] = nskyideal
    data['status'][i] = 0

    return data, profile


def photutils_apphot(data, meta, i):
    """Perform aperture photometric extraction and annular sky subtraction on
    the input image using photutils.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the photometry data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The current integration number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object containing the extracted photometry data.
    """
    if meta.moving_centroid:
        position = [data.centroid_x.values[i], data.centroid_y.values[i]]
    else:
        position = [np.median(data.centroid_x.values),
                    np.median(data.centroid_y.values)]

    # Grab the current frame and mask
    mask = data.mask.values[i]
    good_pixels = (~mask).astype(float)
    flux = np.ma.masked_where(mask, data.flux.values[i])
    derr = np.ma.masked_where(mask, data.err.values[i])

    # Setup the object aperture and the sky annulus
    if meta.aperture_shape == 'circle':
        obj_aper = CircularAperture(position, meta.photap)
        sky_annul = CircularAnnulus(position, meta.skyin, meta.skyout)
    elif meta.aperture_shape == 'ellipse':
        # Sky annulus has the same aspect ratio and orientation as source aper
        theta = meta.photap_theta*np.pi/180
        skyin_b = meta.skyin*(meta.photap_b/meta.photap)
        skyout_b = meta.skyout*(meta.photap_b/meta.photap)

        obj_aper = EllipticalAperture(position, meta.photap, meta.photap_b,
                                      theta)
        sky_annul = EllipticalAnnulus(position, meta.skyin, meta.skyout,
                                      skyout_b, skyin_b, theta)
    elif meta.aperture_shape == 'rectangle':
        # Sky annulus has the same aspect ratio and orientation as source aper
        theta = meta.photap_theta*np.pi/180
        skyin_b = meta.skyin*(meta.photap_b/meta.photap)
        skyout_b = meta.skyout*(meta.photap_b/meta.photap)

        obj_aper = RectangularAperture(position, 2*meta.photap,
                                       2*meta.photap_b, theta)
        sky_annul = RectangularAnnulus(position, 2*meta.skyin, 2*meta.skyout,
                                       2*skyout_b, 2*skyin_b, theta)
    else:
        raise ValueError(f'Unknown aperture_shape "{meta.aperture_shape}"')

    # Compute the number of good pixels in object aperture and sky annulus
    if not meta.skip_apphot_bg:
        nskypix = aperture_photometry(good_pixels, sky_annul,
                                      method=meta.aperture_edge
                                      )['aperture_sum'].data[0]
        nskyideal = aperture_photometry(np.ones_like(good_pixels), sky_annul,
                                        method=meta.aperture_edge
                                        )['aperture_sum'].data[0]
    nappix = aperture_photometry(good_pixels, obj_aper,
                                 method=meta.aperture_edge
                                 )['aperture_sum'].data[0]

    # Compute the total source flux
    aptable = aperture_photometry(flux, obj_aper, derr, mask,
                                  method=meta.aperture_edge)
    aplev = aptable['aperture_sum'].data[0]
    aperr = aptable['aperture_sum_err'].data[0]

    if not meta.skip_apphot_bg:
        # Compute the sky flux per pixel
        skytable = aperture_photometry(flux, sky_annul, derr, mask,
                                       method=meta.aperture_edge)
        skylev = skytable['aperture_sum'].data[0]/nskypix
        skyerr = skytable['aperture_sum_err'].data[0]/nskypix

        # Subtract the sky level that fell within the source aperture
        aplev -= skylev*nappix

        # Add the skyerr to the aperr in quadrature
        aperr = np.sqrt(aperr**2 + nappix*skyerr**2)

    # Store the results in the data object
    data['aplev'][i] = aplev
    data['aperr'][i] = aperr
    data['nappix'][i] = nappix
    data['skylev'][i] = skylev
    data['skyerr'][i] = skyerr
    data['nskypix'][i] = nskypix
    data['nskyideal'][i] = nskyideal
    data['status'][i] = 0

    return data
