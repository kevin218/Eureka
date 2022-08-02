.. _stages:

Eureka's Stages
===============


`Eureka!` is broken down into six parts or "Stages", which are as follows (see also :ref:`Figure 1 <overview_flowchart>`):

- Stage 1: An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 1 outputs from the ``jwst`` pipeline.
- Stage 2: An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 2 outputs from the ``jwst`` pipeline.
- Stage 3: Using Stage 2 outputs, performs background subtraction and optimal spectral extraction. For spectroscopic observations, this stage generates a time series of 1D spectra. For photometric observations, this stage generates a single light curve of flux versus time.
- Stage 4: Using Stage 3 outputs, generates spectroscopic light curves by binning the time series of 1D spectra along the wavelength axis. Optionally removes drift/jitter along the dispersion direction and/or sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- Stage 6: Displays the planet spectrum in figure and table form using results from the Stage 5 fits.

.. _overview_flowchart:
.. image:: ../media/stages_flowchart.png
  :width: 512
  :align: center
  :alt: An overview flowchart of the 6 stages of Eureka!.

  An overview flowchart of the 6 stages of ``Eureka!``.

.. figure:: ../media/stages_flowchart.png
  :alt: An overview flowchart of the 6 stages of Eureka!.

  An overview flowchart of the 6 stages of ``Eureka!``.
