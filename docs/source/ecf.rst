.. _ecf:

Eureka Control File (.ecf)
============================

Stage 3
--------

.. include:: S3_template.ecf
   :literal:

ncpu
'''''''''''
Sets the number of cores being used when Eureka! is executed.
Currently, the only parallelized part of the code is the **background subtraction** for every individual integration and is being initialized in s3_reduce.py with:

.. code-block:: python
    
    util.BGsubtraction(dat, md, log, md.isplots_S3)


suffix
'''''''''''

If your data directory (topdir + datadir, see below) has different dataformats, you want to consider setting this variable.

E.g.: Simulated NIRCam Data:

Stage 2 – For NIRCam, Stage 2 consists of the flat field correction, WCS/wavelength solution, and photometric calibration (counts/sec -> MJy). Note that this is specifically for NIRCam: the steps in Stage 2 change a bit depending on the instrument. The Stage 2 outputs are rougly equivalent to a “flt” file from HST.

Stage 2 Outputs/*calints.fits - Fully calibrated images (MJy) for each individual integration. This is the one you want if you’re starting with Stage 2 and want to do your own spectral extraction.

Stage 2 Outputs/*x1dints.fits - A FITS binary table containing 1D extracted spectra for each integration in the “calint” files.


As we want to do our own spectral extraction, we set this variable to ``calints``.

Note that other Instruments might used different suffixes!


inst
'''''''''''

The instrument you want to analyze data from.

Possible values:

- ``nircam``
- ``niriss``
- ``nirspec``
- ``miri``

ywindow & xwindow
''''''''''''''''''''

Can be set if one wants to remove edge effects (e.g.: many nans at the edges).

Below an example with the following setting:

.. code-block:: python
    
    ywindow     [5,64]
    xwindow     [100,1700]

.. image:: xywindow.png

Everything outside of the box will be discarded and not used in the analysis.

bg_hw & spec_hw
'''''''''''''''''

``bg_hw`` and  ``spec_hw`` set the background and spectrum aperture relative to the source position.

Let's looks at an example with the followind settings:

.. code-block:: python
    
    bg_hw    = 23
    spec_hw  = 18


Looking at the fits file science header,  we can determine the source position:

.. code-block:: python
    
    src_xpos = hdulist['SCI',1].header['SRCXPOS']-xwindow[0]
    src_ypos = hdulist['SCI',1].header['SRCYPOS']-ywindow[0]

In this example ``src_ypos = 29``.

(xwindow[0] and ywindow[0] corrects for the trimming of the dataframe as the edges were removed with the xwindow and ywindow parameters)

The plot below shows you which parts will be used for the background calculation (shaded in white; between the edge and src_ypos - bg_hw, and src_ypos + bg_hw and the edge) and which for the spectrum flux calculation (shaded in red; between src_ypos - spec_hw and src_ypos + spec_hw).

.. image:: bg_hw.png


bg_thresh
'''''''''''

Double-iteration X-sigma threshold for outlier rejection along time axis.
The flux of every background pixel will be considered over time for the current data segment. 
e.g: bg_thresh = [5,5] : Two iterations of 5-sigma clipping will be performed in time for every background pixel. Putliers will be masked and not considered in the background flux calculation.

bg_deg
'''''''''''

Sets the degree of the background subtraction.

The function is defined in S3_data_reduction/optspex.fitbg

Possible values:

- ``bg_deg = None``: No backgound subtraction will be performed.
- ``bg_deg < 0``: The median flux value in the background area will be calculated and subtracted from the entire 2D Frame for this paticular integration.
- ``bg_deg => 0``: A polynomial of degree `bg_deg` will be fitted to every background column (background at a specific wavelength). If the background data has an outlier (or several) which is (are) greater than 5  * (Mean Absolute Deviation), this value will be not considered as part of the background. Step-by-step:
1. Take background pixels of first column
2. Fit a polynomial of degree  ``bg_deg`` to the background pixels.
3. Calculate the residuals (flux(bg_pixels) - polynomial_bg_deg(bg_pixels))
4. Calculate the MAD (Mean Absolute Deviation) of the greatest background outlier.
5. If MAD of the greatest background outlier is greater than 5, remove this background pixel from the background value calculation. Repeat from Step 2. and repeat as long as there is no 5*MAD outlier in the background column.
6. Calculate the flux of the polynomial of degree  ``bg_deg`` (calculated in Step 2) at the spectrum and subtract it.




p3thresh
'''''''''''

Only important if ``bg_deg => 0``: X-sigma threshold for interative outlier rejection during background subtraction (see above).


'''''''''''

Aperture of the spectrum.


p5thresh
'''''''''''

TODO something in optspec


p7thresh
'''''''''''

TODO something in optspec


fittype
'''''''''''

TODO something in optspec


window_len
'''''''''''

TODO something in optspec


prof_deg
'''''''''''

TODO something in optspec


isplots_S3
'''''''''''

Sets how many plots should be saved when running Stage 3.

TODO

testing_S3
'''''''''''

If set to True only the last segement (which is usually the smallest) in the datadir is being run.

topdir + datadir
'''''''''''''''''

The path to the directory containing the Stage 2 JWST data.


topdir + ancildir
'''''''''''''''''

The path to the directory containing the ancillary data.

E.g.: NIRCam needs a photometic file and a gainfile file to convert MJy/sr to DN (Data Numbers) and from DN to Electrons, respectively.
The names of the the files needed are given in the header as hdulist[0].header['R_PHOTOM'] and hdulist[0].header['R_GAIN'] and can be downloaded here: `<https://jwst-crds.stsci.edu/browse_db/>`_











Stage 4
--------


.. include:: S4_template.ecf
   :literal:

nspecchan
'''''''''''


wave_min
'''''''''''


wave_max
'''''''''''


isplots_S4
'''''''''''




