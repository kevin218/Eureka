.. Eureka documentation master file.

.. image:: ../media/Eureka_logo.png
  :width: 256
  :align: center
  :alt: Eureka Logo

|

Welcome to Eureka!'s documentation!
====================================

``Eureka!`` will eventually be capable of reducing data from any JWST instrument and fitting light curves.
At the moment the package is under heavy development, and currently works on NIRCam, NIRSpec, and MIRI data only.
The code is not officially associated with JWST or the ERS team.

The code is separated into six parts or "Stages":

- Stage 1: An optional step that calibrates Raw data (converts ramps to slopes). This step can be skipped if you'd rather use STScI's JWST pipeline's Stage 1 outputs.
- Stage 2: An optional step that calibrates Stage 1 data (performs flatfielding, unit conversion, etc.). This step can be skipped if you'd rather use STScI's JWST pipeline's Stage 2 outputs.
- Stage 3: Starts with Stage 2 data and further calibrates (performs background subtraction, etc.) and reduces the data in order to convert 2D spectra into a time-series of 1D spectra.
- Stage 4: Bins the 1D Spectra and generates light curves. Also removes 1D spectral drift/jitter and sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models.
- Stage 6: Creates a table and a figure summarizing the transmission and/or emission spectra from your many fits.

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
