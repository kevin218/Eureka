.. Eureka documentation master file.

Welcome to Eureka!'s documentation!
====================================

**Welcome to the documentation for Eureka!.**

``Eureka!`` will eventually be capable of reducing data from any JWST instrument and fitting light curves.
At the moment the package is under heavy development, and currently works on NIRCam, NIRSpec, and MIRI data only.
The code is not officially associated with JWST or the ERS team.

The code is separated into six parts or "Stages":

- Stage 1: An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 1 outputs from the ``jwst`` pipeline.
- Stage 2: An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 2 outputs from the ``jwst`` pipeline, although at present it is recommended that you skip the photom step in the ``jwst`` pipeline.
- Stage 3: Using Stage 2 outputs, performs background subtraction and optimal spectral extraction. For spectroscopic observations, this stage generates a time series of 1D spectra. For photometric observations, this stage generates a single light curve of flux versus time.
- Stage 4: Using Stage 3 outputs, generates spectroscopic light curves by binning the time series of 1D spectra along the wavelength axis. Optionally removes drift/jitter along the dispersion direction and/or sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- Stage 6: Displays the planet spectrum in figure and table form using results from the Stage 5 fits.

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
