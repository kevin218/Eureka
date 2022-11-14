.. Eureka documentation master file.

.. image:: ../media/Eureka_logo.png
  :width: 256
  :align: center
  :alt: Eureka Logo

|

Welcome to Eureka!'s documentation!
===================================

``Eureka!`` is a data reduction and analysis pipeline for exoplanet time-series observations, with a particular focus on James Webb Space Telescope (JWST) data.
``Eureka!`` is capable of of reducing JWST time-series data (starting from raw, uncalibrated FITS files) and turning it into precise exoplanet transmission and emission spectra.
At the moment the package is under heavy development. The code is not officially associated with JWST or the ERS team.


The code is broken down into six parts or "Stages", which are as follows (see also :ref:`Eureka!'s Stages <stages>`):

- **Stage 1:** An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 1 outputs from the ``jwst`` pipeline.
- **Stage 2:** An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 2 outputs from the ``jwst`` pipeline.
- **Stage 3:** Using Stage 2 outputs, performs background subtraction and optimal spectral extraction. For spectroscopic observations, this stage generates a time series of 1D spectra. For photometric observations, this stage generates a single light curve of flux versus time.
- **Stage 4:** Using Stage 3 outputs, generates spectroscopic light curves by binning the time series of 1D spectra along the wavelength axis. Optionally removes drift/jitter along the dispersion direction and/or sigma clips outliers.
- **Stage 5:** Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- **Stage 6:** Displays the planet spectrum in figure and table form using results from the Stage 5 fits.

The full code for ``Eureka!`` is available on `GitHub <http://github.com/kevin218/Eureka>`_.


Citing ``Eureka!`` and its dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to just cite the use of ``Eureka!`` in published work, please use the following citation to the `JOSS paper <https://joss.theoj.org/papers/10.21105/joss.04503>`_. 

.. code-block::

    @article{Bell2022, 
         doi = {10.21105/joss.04503}, 
         url = {https://doi.org/10.21105/joss.04503}, 
         year = {2022}, 
         publisher = {The Open Journal}, 
         volume = {7}, 
         number = {79}, 
         pages = {4503}, 
         author = {Taylor J. Bell and Eva-Maria Ahrer and 
                   Jonathan Brande and Aarynn L. Carter and Adina D. Feinstein 
                   and Giannina {Guzman Caloca} and Megan Mansfield and 
                   Sebastian Zieba and Caroline Piaulet and Björn Benneke and 
                   Joseph Filippazzo and Erin M. May and Pierre-Alexis Roy and 
                   Laura Kreidberg and Kevin B. Stevenson}, 
         title = {Eureka!: An End-to-End Pipeline for 
                  JWST Time-Series Observations}, 
        journal = {Journal of Open Source Software} 
    }

Citation information for ``Eureka!`` dependencies is available in the ``meta`` objects returned at each stage. ``meta.citations`` stores a list of names of all Python packages and JWST instruments used in the analysis,
and ``meta.bibliography`` stores a list of key-value pairs where the keys are the elements of ``meta.citations`` and the values are lists of the relevant BibTeX entries for each citable dependency or instrument. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   stages
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
