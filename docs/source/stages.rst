.. _stages:

Eureka's Stages
===============


`Eureka!` is broken down into six "stages" which each consist of many "steps". The six stages are as follows (a visual overview is also provided in :ref:`Figure 1 <overview_flowchart>`):

- Stage 1: An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 1 outputs from the ``jwst`` pipeline.
- Stage 2: An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped within ``Eureka!`` if you would rather use the Stage 2 outputs from the ``jwst`` pipeline.
- Stage 3: Using Stage 2 outputs, performs background subtraction and optimal spectral extraction. For spectroscopic observations, this stage generates a time series of 1D spectra. For photometric observations, this stage generates a single light curve of flux versus time.
- Stage 4: Using Stage 3 outputs, generates spectroscopic light curves by binning the time series of 1D spectra along the wavelength axis. Optionally removes drift/jitter along the dispersion direction and/or sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- Stage 6: Displays the planet spectrum in figure and table form using results from the Stage 5 fits.

.. _overview_flowchart:

.. figure:: ../media/stages_flowchart.png
  :alt: An overview flowchart of the 6 stages of Eureka!.

  Figure 1: An overview flowchart of the 6 stages of ``Eureka!``.


A more detailed discussion of the steps taken in stages 3--6 are summarized below. Stages 1 and 2 are left out as they basically just offer a wrapper for the ``jwst`` pipeline which has already documented their `Stage 1 <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html>`_, `Stage 2 (spectroscopy) <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html>`_, amd `Stage 2 (photometry) <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_image2.html>`_.

Stage 3: Data Reduction
-----------------------


Stage 4: Generating Lightcurves
-------------------------------


Stage 5: Lightcurve Fitting
---------------------------


Stage 6: Plotting Plantary Spectra
----------------------------------
