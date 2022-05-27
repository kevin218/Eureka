.. Eureka documentation master file.

Welcome to Eureka!'s documentation!
====================================

**Welcome to the documentation for Eureka!.**

``Eureka!`` will eventually be capable of reducing data from any JWST instrument and fitting light curves.
At the moment the package is under heavy development, and currently works on NIRCam, NIRSpec, and MIRI data only.
The code is not officially associated with JWST or the ERS team.

The code is separated into six parts or "Stages":

- Stage 1: An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped if you'd rather use the ``jwst`` pipeline's Stage 1 outputs.
- Stage 2: An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped if you'd rather use the ``jwst`` pipeline's Stage 2 outputs, although it is recommended that you skip the photom step in that pipeline.
- Stage 3: Starts with the files produced by Stage 2 and performs background subtraction. This stage also reduces the data in order to convert 2D spectra into a time-series of 1D spectra or 2D photometric images into a 1D time-series.
- Stage 4: Bins the 1D spectral time-series along the wavelength axis and generates light curves. Also removes 1D spectral drift/jitter and sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- Stage 6: Creates a table and a figure summarizing the transmission and/or emission spectra from your Stage 5 fit(s).

The full code for ``Eureka!`` is available on `GitHub <http://github.com/kevin218/Eureka>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   ecf
   outputs
   contribute
   api
   faq
   copyright


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
