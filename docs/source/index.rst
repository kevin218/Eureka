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


Citing ``Eureka!``
~~~~~~~~~~~~~~~~~~

To make citing ``Eureka!`` and its dependencies easier, you can simply print the ``citations`` and ``bibliography`` attributes of the Meta object returned at each stage of analysis. 
``meta.citations`` will return a list of named python packages (and the JWST instrument being analyzed, if applicable), while ``meta.bibliography`` will return a list of formatted BibTeX 
entries for each citation. If you wish to cite just the use of ``Eureka!`` in published work, please use the following citation to the `JOSS paper <https://arxiv.org/abs/2207.03585>`_.

.. code-block::

    @ARTICLE{Bell2022,
          author = {{Bell}, Taylor J. and {Ahrer}, Eva-Maria and {Brande}, Jonathan and {Carter}, Aarynn L. and {Feinstein}, Adina D. and {Guzman Caloca}, Giannina and {Mansfield}, Megan and {Zieba}, Sebastian and {Piaulet}, Caroline and {Benneke}, Bj{\"o}rn and {Filippazzo}, Joseph and {May}, Erin M. and {Roy}, Pierre-Alexis and {Kreidberg}, Laura and {Stevenson}, Kevin B.},
            title = "{Eureka!: An End-to-End Pipeline for JWST Time-Series Observations}",
          journal = {arXiv e-prints},
        keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
            year = 2022,
            month = jul,
              eid = {arXiv:2207.03585},
            pages = {arXiv:2207.03585},
    archivePrefix = {arXiv},
          eprint = {2207.03585},
    primaryClass = {astro-ph.IM},
          adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220703585B},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }



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
