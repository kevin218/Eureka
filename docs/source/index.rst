.. Eureka documentation master file.

Welcome to Eureka!'s documentation!
====================================

**Welcome to the documentation for Eureka!.**

``Eureka!`` will eventually be capable of reducing data from any JWST instrument and fitting light curves.
At the moment the package is under heavy development, and currently works on NIRCam, NIRSpec, and MIRI data only.
The code is not officially associated with JWST or the ERS team.

The code is separated into five parts or "Stages":

- Stage 1: An optional step that calibrates Raw data (converts ramps to slopes). This step can be skipped if you'd rather use STScI's JWST pipeline's Stage 1 outputs.
- Stage 2: An optional step which calibrates Stage 1 data (performs flatfielding, unit conversion, etc.). This step can be skipped if you'd rather use STScI's JWST pipeline's  Stage 2 outputs.
- Stage 3: Starts with Stage 2 data and reduces the data (performs background subtraction, etc.) in order to convert 2D spectra into a time-series of 1D spectra
- Stage 4: Bins the 1D Spectra and generates light curves
- Stage 5: Fits the light curves (under development)

The full code for ``Eureka!`` is available on `GitHub <http://github.com/kevin218/Eureka>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   ecf
   contribute
   api
   faq
   copyright



.. toctree::
   :maxdepth: 1
   :caption: Examples

   hackathon-day2-tutorial



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
