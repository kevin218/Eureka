import numpy as np
import scipy.ndimage.interpolation as si
import scipy.optimize as so
from . import gaussian as g

"""
  File:
  -----
  psf_fit.py

  Description:
  ------------
  Routines for creating PSF (from a supersampled PSF) and fitting of
  PSF images to data images. There are two functions to make a PSF,
  either interpolating or binning down from a supersampled PSF.

  Package Contents:
  -----------------

  There are two main types of routines: 'constructors' to make a PSF
  image, and 'PSF fitting' routines to fit a PSF to data. The package
  also contains 'wrappers' to easy implement in Spitzer data, and
  other subroutines.

  PSF Constructors:
  - make_psf_interp:  Makes a PSF image by shifting, rescaling, and
                      setting the stellar and sky fluxes of a super
                      sampled PSF.

  - make_psf_binning: Makes a PSF image by binning down a super sampled
                      PSF. Sets then the stellar and sky fluxes.

  PSF Fitting:
  - psf_fit: Fits a supersampled PSF to a data image. The position is
             fitted at discrete positions while the stellar and sky
             fluxes are fitted with scipy's leastsq function.

  Spitzer Wrapper:
  - spitzer_fit:  Routine wrapper for easy plug-in into POET pipeline.
                  Fits a PSF in a data frame from Spitzer.

  Subroutines:
  - binarray:  Resamples a 2D image by stacking and adding every bin
               of pixels along each dimension.

  - residuals: Calculates the residuals of a weighted, stellar flux +
               sky background fit of a model to data.

  - gradient: Calculates the gradient of the parameters in residuals.

  Modification History:
  ---------------------
  2011-07-21  patricio  Wrapped up the PSF fitting routines into this file.
                        pcubillos@fulbrightmail.org
"""


# :::: PSF Constructors ::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def make_psf_interp(spsf, shape, scale, params, psfctr, *args):
    """
      Makes a PSF image by shifting, rescaling, and setting the stellar
      and sky fluxes of a super sampled PSF.

      Parameters:
      -----------
      spsf: 2D ndarray
           The supersampled PSF image.

      shape: 2-element tuple
             The shape of the output image.

      scale:  Scalar
              Ratio of the PSF and data pixel-scales.

      params: 4-elements tuple [yshift, xshift, flux, sky]
              y, x-shift: The desired position of the center of the PSF
                          relative to the center of the image.
              flux: The total flux for the star.
              sky:  The sky flux level.

      psfctr: 2-element tuple [y, x]
              y, x-position of the center of the supersampled PSF.

      Returns:
      --------
      psf:    2D ndimage
              Resampled PSF image.

      center: 2-elements tuple
              The position of the center of the returned PSF.

      Example:
      --------
      >>> import psf_fit as pf
      >>> import sys
      >>> import numpy as np
      >>> sys.path.append('/home/esp01/events/wa008b-patricio/wa008bs41/lib/')
      >>> sys.path.append('/home/patricio/ast/esp01/convert/lib/python/gaussian/')
      >>> import manageevent as me

      >>> # Let's obtain a PSF from a Spitzer data set:
      >>> e = me.loadevent('/home/esp01/events/wa008b-patricio/wa008bs41/run/fgc/wa008bs41_ctr',
      >>>                  load=['data','uncd','mask'])
      >>> sst_psf = np.copy(e.psfim)
      >>> # The PSF center position:
      >>> psfctr = np.copy(e.psfctr)

      >>> # Make a PSF of shape 21 by 21:
      >>> shape = 21,21
      >>> # The Spitzer provided PSF has a pixel scale 5 times finer:
      >>> scale = 5.0
      >>> # Make a PSF shifted 3 pixels down and 6 pixes to the left of the
      >>> # center of the image. Star flux is 1000 and the sky level 100:
      >>> params = [-3.0, 6.0, 1e4, 1e2]
      >>> psf, pos = pf.make_psf_interp(sst_psf, shape, scale, params, psfctr)

      >>> plt.figure(0)
      >>> plt.clf()
      >>> plt.imshow(psf, interpolation='nearest', origin='ll')
      >>> plt.colorbar()
      >>> # Print the position of the center:
      >>> print(pos)

      Modification History:
      ---------------------
      2011-07-24  patricio  Added return center.
      2011-05-19  Patricio  Initial version.  pcubillos@fulbrightmail.org
    """

    # We will extract a section of the supersampled PSF. Calculate its shape:
    shape = np.asarray(shape, float)
    psf_shape = 1 + scale * (shape - 1)
    # Calculate the zoom factor:
    # (See my notes on scipy.ndimage.interpolation.zoom for detailed
    #  explanation:   /home/patricio/ast/esp01/notes/scipy_notes.txt)
    zoom = (shape + 0.5) / psf_shape

    # Extract a sub-section of the PSF around psfctr:
    lims = np.array([np.around(psfctr) - np.around(psf_shape / 2),
                     np.around(psfctr) - np.around(psf_shape / 2) + psf_shape])
    spsf = np.copy(spsf[lims[0, 0]:lims[1, 0], lims[0, 1]:lims[1, 1]])

    # Shift the PSF:
    shift = np.asarray(params[0:2]) * scale
    shiftpsf = si.shift(spsf, shift, mode='nearest')

    # Resample the PSF (zoom uses a spline interpolation):
    psf = si.zoom(shiftpsf, zoom, mode='nearest')

    # Normalize:
    psf = psf / np.sum(psf)

    # Construct the PSF (set stellar and sky flux):
    psf = psf * params[2] + params[3]

    # Remember, we subtract 0.5 because the origin is at the center of
    # the first pixel:
    center = shape / 2.0 - 0.5 + np.asarray(params[0:2])
    return psf, center


def make_psf_binning(spsf, shape, scale, params, psfctr, subpsf=None):
    """
      Makes a PSF image by binning down a super sampled PSF. Sets then
      the stellar and sky fluxes.

      Parameters:
      -----------
      spsf: 2D ndarray
            The supersampled PSF image.

      shape:  2-element tuple
              The shape of the output image.

      scale:  Scalar
              Ratio of the PSF and data pixel-scales.

      params: 4-elements tuple [y, x, flux, sky]
              y, x: Subpixel position where to put the sPSF center. As this
                    is a pixel position, y,x must be integers (see Example).
              flux: The total flux from the star.
              sky:  The sky flux level.

      psfctr: 2-element tuple  [y, x]
              y, x-position of the center of the PSF.

      subpsf: 2D ndarray
              An array where to write the subsection of the supersampled
              PSF. It should have shape: shape*scale. It will be
              overrited.

      Return:
      -------
      binpsf : 2D ndarray
               Rebinned PSF image.
      pos : 2-elements tuple
            The position of the center of the PSF in binpsf.

      Example:
      --------
      >>> import psf_fit as pf
      >>> import pyfits  as pyf
      >>> import matplotlib.pyplot as plt

      >>> ttpsf = pyf.getdata('/home/esp01/events/wa008b-patricio/Tiny_tim/irac4_5600K_100x.fits')
      >>> psfctr = np.asarray(np.shape(ttpsf))/2
      >>> scale = 100

      >>> shape = 21,21
      >>> # To put the center at the image center, calculate the corresponding
      >>> # subpixel: center = shape * scale / 2
      >>> params = 1050, 1050, 1.0, 0.0
      >>> psf,pos = pf.make_psf_binning(ttpsf, shape, scale, params, psfctr)

      >>> plt.figure(10)
      >>> plt.clf()
      >>> plt.imshow(psf, origin='lower left', interpolation='nearest')
      >>> plt.colorbar()
      >>> print(pos)

      Notes:
      ------
      make_psf_binning works under the premise that the supersampled PSF
      is sampled fine enough that differences less than a supersampled
      pixel is undistinguishable. Then the sPSF can be shifted
      in units of subpixels only.

      Note also the input parameter 'param' is different than in
      make_psf_interp.

      Modification History:
      ---------------------
      2011-07-20  patricio  Added subpsf parameter (makes it faster).
      2011-05-19  patricio  Initial version.  pcubillos@fulbrightmail.org
    """
    # Pixel of the center of the PSF:
    yctr, xctr = (np.around(psfctr)).astype(int)

    # Sub-PSF shape:
    ns = (np.asarray(shape, float) * scale).astype(int)

    # Trim the psf to the specified area:
    if subpsf is None:
        subpsf = np.zeros(ns)

    # Extract sub-section from supersampled PSF:
    # params = params
    # print(params)
    subpsf[:] = spsf[yctr - params[0]:yctr - params[0] + ns[0],
                     xctr - params[1]:xctr - params[1] + ns[1]]

    # Resampled image:
    binpsf = binarray(subpsf, scale)

    # Normalize:
    binpsf = binpsf / np.sum(binpsf)
    # Set flux and add sky:
    binpsf = binpsf * params[2] + params[3]
    # Position in binpsf:
    pos = (np.asarray(params[0:2], float) - (scale - 1) / 2.0) / scale

    # Return the array and the postion of the center:
    return binpsf, pos


# :::: PSF Fitting Routines ::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def psf_fit(data, fluxguess, spsf, psfctr, scale, shift, make="bpf",
            mask=None, weights=None, step=None, pos=None):
    """
      Fits a supersampled PSF to a data image. The position is fitted at
      discrete postions while the stellar and sky fluxes are fitted with
      scipy's leastsq function.

      Parameters:
      -----------
      data:  2D ndarray
             The science image we are trying to fit.

      fluxguess: 2-element tuple  [flux, sky]
             Tuple giving the starting point to fit the total star flux
             and sky flux level.

      spsf: 2D ndarray
            The supersampled PSF image.

      psfctr: 2-element tuple  [y, x]
              y, x-position of the center of the PSF.

      scale:  scalar
              Ratio of the PSF and data pixel-scales.

      shift: 2-element tuple  [yshift, xshift]
             Each element is a 1D array containing the shifts of the
             center of the PSF to the center of the image at which the
             fit will be evaluated.

      mask : ndarray
             Mask of bad pixel values, same shape as data. Good pixels
             have value 1; bad pixels have value 0, and will not be
             considered in the fit.

      weights: ndarray
               Weights for the minimization, for scientific data the
               weights should be 1/sqrt(variance). Same shape as data.

      step : scalar
             The initial step of the number of elements to jump when
             evaluating shift.

      pos : 2-element list
            The index of the elements in shift where to start the
            evaluation.

      Example:
      --------

      >>> import psf_fit as pf
      >>> import sys, os, time
      >>> import numpy as np
      >>> sys.path.append('/home/esp01/events/wa008b-patricio/wa008bs41/lib/')
      >>> sys.path.append('/home/patricio/ast/esp01/convert/lib/python/gaussian/')
      >>> import manageevent as me
      >>> import pyfits      as pyf

      >>> # Example #1:
      >>> # Using a Spitzer supplied PSF and make_psf_interp:

      >>> # Get a PSF and its center:
      >>> e = me.loadevent('/home/esp01/events/wa008b-patricio/wa008bs41/run/fgc/wa008bs41_ctr',
      >>>                  load=['data','uncd','mask'])
      >>> sst_psf = np.copy(e.psfim)
      >>> psfctr = np.copy(e.psfctr)

      >>> # The scale factor:
      >>> scale = 5.0

      >>> # Let's create an image to fit:
      >>> # The image size will be 21 by 21:
      >>> shape = np.array([21,21])

      >>> # Define the position of the center of the PSF, and fluxes:
      >>> params = [1.75, 0.5, 5e4, 2e2]
      >>> # Make the image:
      >>> image, center = pf.make_psf_interp(sst_psf, shape, scale, params, psfctr)
      >>> # Add some noise:
      >>> noise = np.sqrt(image) * np.random.randn(21,21)
      >>> # The image to fit:
      >>> y = image + noise
      >>> var = np.abs(image)

      >>> # Let's say our prior guess lies whitin 1 pixel from the given position:
      >>> yguess = params[0] + 2*(np.random.rand()-0.5)
      >>> xguess = params[1] + 2*(np.random.rand()-0.5)

      >>> # Array of Y,X shifs around our guess where to search:
      >>> noffset = 201
      >>> offsetrad = 1.0  # search within a 1 pixel radius:
      >>> offset = offsetrad * np.linspace(-1.0, 1.0, noffset)

      >>> # The shifts are relative to the center of the image:
      >>> yshift = yguess + offset
      >>> xshift = xguess + offset
      >>> shift = (yshift, xshift)

      >>> # Starting point, guess for the fluxes:
      >>> fluxguess = (0.1e5, 80)

      >>> # Find the best fit:
      >>> pos, bestp, chisq = pf.psf_fit(y, fluxguess, sst_psf, psfctr, scale,
      >>>                                shift, mask=None, weights=1/var, make='ipf')
      >>> # Best position:
      >>> print(pos)
      >>> # Best flux fit:
      >>> print(bestp)

      >>> # Example #2:
      >>> # Using a Tiny Tim supplied PSF and make_psf_binning:

      >>> # Get a PSF and its center:
      >>> ttpsf = pyf.getdata('/home/esp01/events/wa008b-patricio/Tiny_tim/irac4_5600K_100x.fits')
      >>> psfctr = np.asarray(np.shape(ttpsf))/2
      >>> # The scale factor:
      >>> scale = 100

      >>> # Create an image to fit:
      >>> shape = np.array([21,21])
      >>> params = [1043, 915, 5e5, 200]
      >>> image, center = pf.make_psf_binning(ttpsf, shape, scale, params, psfctr)
      >>> # Add some noise:
      >>> noise = np.sqrt(image) * np.random.randn(21,21)
      >>> # The image to fit:
      >>> y = image + noise
      >>> var = np.abs(image)

      >>> # Let's say our guess is whitin 1 pixel from the given position:
      >>> yguess = params[0] + np.random.randint(-scale,scale)
      >>> xguess = params[1] + np.random.randint(-scale,scale)

      >>> # Array of Y,X shifs around our guess where to search:
      >>> offsetrad = 1.0  # search within a 1 pixel radius:
      >>> noffset = int(2*scale*offsetrad + 1)
      >>> offset = np.arange(noffset) - noffset/2

      >>> # The shifts are relative to the position of the PSF:
      >>> yshift = yguess + offset
      >>> xshift = xguess + offset
      >>> shift = (yshift, xshift)

      >>> # Starting point, guess for the fluxes:
      >>> fluxguess = (1e4, 80)

      >>> # Find the best fit:
      >>> tini = time.time()
      >>> pos, bestp, chisq = pf.psf_fit(y, fluxguess, ttpsf, psfctr, scale,
      >>>                                shift, mask=None, weights=1/var, make='bpf')
      >>> print(time.time()-tini)
      >>> # Best position:
      >>> print(pos)
      >>> # Best flux fit:
      >>> print(bestp)

      Modification History:
      ---------------------
      2011-05-21  patricio  Initial version.  pcubillos@fulbrightmail.org
      2011-05-27  patricio  Include gradient parameter in leastsq.
      2011-07-26  patricio  Unified both make_psf.
    """
    shape = np.shape(data)

    # Default mask: all good
    if mask is None:
        mask = np.ones(shape)

    # Default weights: no weighting
    if weights is None:
        weights = np.ones(shape)

    # Unpack shift
    y, x = shift
    # Lengths of the dependent varables:
    ny = len(y)
    nx = len(x)

    # Default initial step:
    if step is None:
        step = int(ny / 2)

    # Default initial position:
    if pos is None:
        pos = [int(ny / 2), int(nx / 2)]

    # Allocate space for subpsf in make_psf_bin outside the loop:
    ns = (np.asarray(shape, float) * scale).astype(int)
    subpsf = np.zeros(ns)

    # Define PSF constructor:
    if make == "ipf":
        maker = make_psf_interp
        # Discard values on the edge of the mask:
        j = 2
        mask[0:j, :] = mask[:, 0:j] = mask[-j:, :] = mask[:, -j:] = 0

    elif make == "bpf":
        maker = make_psf_binning
    else:
        print("Unacceptable PSF constructor. Must be 'ipf' or 'bpf'")
        return

    # Initialize a chi-square grid:
    chisq = -np.ones((ny, nx))

    # goodratio = np.sum(mask)/np.size(mask)
    # print(goodratio)

    while (step > 0):
        # Calculate chisq in the surrounding:
        for shifty in np.arange(-1, 2):
            # y position to evaluate:
            posy = np.clip(pos[0] + shifty * step, 0, ny - 1)
            for shiftx in np.arange(-1, 2):
                # x position to evaluate:
                posx = np.clip(pos[1] + shiftx * step, 0, nx - 1)
                if chisq[posy, posx] == -1:
                    # Make a psf model for given y,x position:
                    model, center = maker(spsf, shape, scale,
                                          [int(y[posy]), int(x[posx]), int(1.0), int(0.0)],
                                          psfctr, subpsf)

                    # Weighted, masked values:
                    mmodel = model[np.where(mask)]
                    mdata = data[np.where(mask)]
                    mweights = weights[np.where(mask)]
                    args = (mdata, mmodel, mweights)
                    # The fitting:
                    p, cov, info, msg, flag = so.leastsq(residuals, fluxguess, args,
                                                         Dfun=gradient, full_output=True,
                                                         col_deriv=1)
                    # err = np.sqrt(np.diagonal(cov))
                    # Chi-square per degree of freedom:
                    cspdof = (np.sum((info['fvec']) ** 2.0) /
                              (len(info["fvec"]) - len(fluxguess)))
                    chisq[posy, posx] = cspdof

        # Is the current position the minimum chi-square?
        # Minimum chi-square position:
        mcp = np.where(chisq == np.amin(chisq[np.where(chisq >= 0)]))

        # If it is, then reduce the step size:
        if pos[0] == mcp[0][0] and pos[1] == mcp[1][0]:
            step = int(np.round(step / 2.0))
        # If not, then move to the position of min. chi-square:
        else:
            pos[0] = mcp[0][0]
            pos[1] = mcp[1][0]

    # The best fitting parameters at the best position:
    model, center = maker(spsf, shape, scale, [int(y[pos[0]]), int(x[pos[1]]), 1, 0],
                          psfctr, subpsf)

    # This is the fix I need to do:
    mmodel = model[np.where(mask)]
    mdata = data[np.where(mask)]
    mweights = weights[np.where(mask)]
    args = (mdata, mmodel, mweights)
    p, cov, info, msg, flag = so.leastsq(residuals, fluxguess, args,
                                         Dfun=gradient, full_output=True, col_deriv=1)
    # err = np.sqrt(np.diagonal(cov))

    # Return the position of min chisq, the best parameters, and the chisq grid:
    return center, p, chisq


# :::: Spitzer Wrapper Routine :::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def spitzer_fit(data, mask, weights, psf, psfctr, scale, make,
                offsetrad=1.0, noffset=201):
    """
      Routine wrapper for easy plug-in into POET pipeline.
      Fits a PSF in a data frame from Spitzer.

      Parameters:
      -----------
      data:    2D ndarray
               Data image to fit the PSF.

      mask:    2D ndarray
               Mask of bad pixel values, same shape as data. Good pixels
               have value 1; bad pixels have value 0, and will not be
               considered in the fit.

      weights: 2D ndarray
               Weights for the minimization, for scientific data the
               weights should be 1/sqrt(variance). Same shape as data.

      psf:     2D ndimage
               The supersampled PSF image.

      psfctr:  2-elements tuple [y, x]
               y, x-position of the center of the PSF.

      scale:   Scalar
               Ratio of the PSF and data pixel-scales.

      noffset: Scalar
               Radii around the guess position where to look for best fit.

      Returns:
      --------
      bestfit: 4-elements tuple [y, x, starflux, skyflux]
               position and fluxes of the PSF that best fit the data.

      Modification History:
      ---------------------
      2011-07-26  patricio  First documented version.
                            pcubillos@fulbrightmail.org
    """
    # Initial flux guess:
    skyguess = np.median(data)
    starguess = np.sum(data - skyguess)
    fluxguess = [starguess, skyguess]

    # Use fit gaussian for a first YX guess:
    datashape = np.asarray(np.shape(data))
    fit, err = g.fitgaussian(data, fitbg=1, yxguess=datashape / 2)
    yxguess = fit[2:4]

    # Obtain the position guess (depending on the make used):
    if make == 'bpf':
        # Scale to the PSF scale:
        yguess, xguess = np.around(scale * (yxguess + 0.5) - 0.5)
    elif make == 'ipf':
        # Guess with respect to the center of the image:
        yguess = yxguess[0] - np.shape(data)[0] / 2.0 - 0.5
        xguess = yxguess[1] - np.shape(data)[1] / 2.0 - 0.5

    # Array of shifs around our guess where to search:
    if make == 'bpf':
        noffset = int(2 * scale * offsetrad + 1)
        offset = np.arange(noffset) - noffset / 2
    elif make == 'ipf':
        offset = offsetrad * np.linspace(-1.0, 1.0, noffset)

    yshift = yguess + offset
    xshift = xguess + offset
    shift = (yshift, xshift)

    # Do the fit:
    pos, bestp, chisq = psf_fit(data, fluxguess, psf, psfctr, scale,
                                shift, mask=mask, weights=weights, make=make)

    # Return best fit: [y, x, starflux, skyflux]
    return (pos[0], pos[1], bestp[0], bestp[1])


# :::: Sub Routines ::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def binarray(image, binsize):
    """
      Resamples a 2D image by stacking and adding every bin of pixels
      along each dimension.

      Parameters:
      -----------
      image : 2D ndarray
      scale : integer scalar

      Return:
      -------
      rimage : 2D ndarray
               Resampled image.
      Example:
      --------
      >>>import stack_psf as s
      >>>a = np.array([[1,0,1,0,1,2],
                       [1,1,1,1,1,1],
                       [0,0,0,1,1,1],
                       [1,0,0,1,0,0]])
      >>>b = s.binarray(a,2)
      Y dimension is incommensurable, ignoring last incomplete stack.
      >>>print(b)
      [[ 3.,  3.,  3.],
       [ 1.,  2.,  2.]]
      >>>b = s.binarray(a,2)
      >>>print(b)
      [[ 5.  9.]]

      Modfication History:
      --------------------
      2011-06-04  patricio  Very first version.
                            pcubillos@fulbrightmail.org
    """
    ny, nx = np.shape(image)
    # Shout if stack is incommensurable:
    if ny % binsize != 0:
        print("Y dimension is incommensurable, ignoring last incomplete stack.")
    if nx % binsize != 0:
        print("X dimension is incommensurable, ignoring last incomplete stack.")
    # Output resampled array:
    newshape = int(ny / int(binsize)), int(nx / int(binsize))

    binarr = np.zeros(newshape)

    # Stack and add the values:
    for j in np.arange(newshape[0]):
        ystart = j * binsize
        yend = (j + 1) * binsize
        for i in np.arange(newshape[1]):
            xstart = i * binsize
            xend = (i + 1) * binsize
            binarr[j, i] = np.sum(image[ystart:yend, xstart:xend])

    return binarr


def residuals(params, data, model, weights):
    """
      Calculates the residuals of a weighted, stellar flux + sky
      background fit of a model to data.

      Parameters:
      -----------
      params : 2-element tuple  [flux, sky]
               The model parameters to fit. Flux is the scaling factor,
               while sky is a constant background.
      data : 1D ndarray
             An array with the data values.
      model : 1D ndarray
              Same shape as data, this array contains the stellar model.
      weights : ndarray
                Same shape as data, this array contains weighting
                factors to ponderate the fit. Usually corresponds to:
                weights = 1/standard deviation.

      Result:
      -------
      This routine return a 1D ndarray with the weighted differences
      between the model and the data.

      Modification History:
      ---------------------
      2011-05-27  patricio  Initial Version.
                            pcubillos@fulbrightmail.org
    """
    return (model * params[0] + params[1] - data) * weights


def gradient(params, data, model, weights):
    """
      Calculates the gradient of the parameters in residuals.

      Parameters:
      -----------
      params : 2-element tuple  [flux, sky]
               The model parameters to fit. Flux is the scaling factor,
               while sky is a constant background.
      data : 1D ndarray
             An array with the data values.
      model : 1D ndarray
              Same shape as data, this array contains the stellar model.
      weights : ndarray
                Same shape as data, this array contains weighting
                factors to ponderate the fit. Usually corresponds to:
                weights = 1/standard deviation.

      Result:
      -------
      This routine return a tuple of 1D ndarrays. Each element in the
      tuple corresponds to the derivative of residuals with respect to
      each element in params.

      Modification History:
      ---------------------
      2011-05-27  patricio  Initial Version.
                            pcubillos@fulbrightmail.org
    """
    return [model * weights, weights]
