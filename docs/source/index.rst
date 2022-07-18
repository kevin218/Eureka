.. Eureka documentation master file.

.. image:: ../media/Eureka_logo.png
  :width: 256
  :align: center
  :alt: Eureka Logo

|

Welcome to Eureka!'s documentation!
====================================

``Eureka!`` is a data reduction and analysis pipeline for exoplanet time-series observations, with a particular focus on James Webb Space Telescope (JWST) data.
``Eureka!`` is capable of of reducing JWST time-series data (starting from raw, uncalibrated FITS files) and turning it into precise exoplanet transmission and emission spectra.
At the moment the package is under heavy development. The code is not officially associated with JWST or the ERS team.

The code is separated into six parts or "Stages":

- Stage 1: An optional step that calibrates Raw data (converts ramps to slopes). This step can be skipped if you'd rather use STScI's JWST pipeline's Stage 1 outputs.
- Stage 2: An optional step that calibrates Stage 1 data (performs flatfielding, unit conversion, etc.). This step can be skipped if you'd rather use STScI's JWST pipeline's Stage 2 outputs.
- Stage 3: Starts with Stage 2 data and further calibrates (performs background subtraction, etc.) and reduces the data in order to convert 2D spectra into a time-series of 1D spectra.
- Stage 4: Bins the 1D Spectra and generates light curves. Also removes 1D spectral drift/jitter and sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models.
- Stage 6: Creates a table and a figure summarizing the transmission and/or emission spectra from your many fits.

The full code for ``Eureka!`` is available on `GitHub <http://github.com/kevin218/Eureka>`_.


Citing ``Eureka!``
~~~~~~~~~~~~~~~~~~

If you wish to cite the use of ``Eureka!`` in published work, please use the following citation to the `JOSS paper <https://arxiv.org/abs/2207.03585>`_.

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
