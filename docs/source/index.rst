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
At the moment, the package is still under development. The code is not officially associated with JWST or the ERS team.


The code is broken down into six parts or "Stages", which are as follows (see also :ref:`Eureka!'s Stages <stages>`):

- **Stage 1:** An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 1 outputs from the ``jwst`` pipeline.
- **Stage 2:** An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 2 outputs from the ``jwst`` pipeline, although at present it is recommended that you skip the photom step in the ``jwst`` pipeline.
- **Stage 3:** Using Stage 2 outputs, performs background subtraction and optimal spectral extraction. For spectroscopic observations, this stage generates a time series of 1D spectra. For photometric observations, this stage generates a single light curve of flux versus time.
- **Stage 4:** Using Stage 3 outputs, generates spectroscopic light curves by binning the time series of 1D spectra along the wavelength axis. Optionally removes drift/jitter along the dispersion direction and/or sigma clips outliers.
- **Stage 5:** Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- **Stage 6:** Displays the planet spectrum in figure and table form using results from the Stage 5 fits.

In general, it is recommended to interface with Eureka! using "Eureka! Control Files" (ECFs) and running command line scripts.
This helps to increase the automation of the pipeline and increases the reproducibility of your results as the ECF you used
will be copied to the output folder and your analysis will follow a pre-defined order. That way if somebody asks you how you
analyzed your data, you can just send them your copied ECF files and the version number of Eureka! that you used.

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


Similar Tools
~~~~~~~~~~~~~

Below we discuss the broader data reduction and fitting ecosystem in which `Eureka!` lives. Several similar open-source tools are discussed
below to provide additional context, but this is not meant to be a comprehensive list.

As mentioned above, `Eureka!` makes use of the first two stages of `jwst <https://github.com/spacetelescope/jwst>`_ while offering
significantly different extraction routines and novel spectral binning and fitting routines beyond what is contained in `jwst`.
`Eureka!` bears similarities to the `POET <https://github.com/kevin218/POET>`_ and
`WFC3 <https://github.com/kevin218/WFC3>`_ pipelines, developed for Spitzer/IRAC and HST/WFC3 observations
respectively; in fact, much of the code from those pipelines has been incorporated into `Eureka!`. `Eureka!` is near feature
parity with `WFC3`, but the Spitzer specific parts of the `POET` pipeline have not been encorporated into `Eureka!`.
The `SPCA <https://github.com/lisadang27/SPCA>`_ pipeline developed for the reduction and fitting of Spitzer/IRAC observations also bears
some similarity to this pipeline, and some snippets of that pipeline have also been encorporated into `Eureka!`. The
`tshirt <https://github.com/eas342/tshirt>`_ package also offers spectral and photometric extraction routines that work for HST
and JWST data. `PACMAN <https://github.com/sebastian-zieba/PACMAN>`_ is another open-source end-to-end pipeline
developed for HST/WFC3 observations. The `exoplanet <https://github.com/exoplanet-dev/exoplanet>`_
and `juliet <https://github.com/nespinoza/juliet>`_ packages offer some similar capabilities as the observation fitting parts of `Eureka!`.



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
