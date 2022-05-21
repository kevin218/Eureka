---
title: '`Eureka!`: An End-to-End Pipeline for JWST and HST Exoplanet Observations'
tags:
  - Python
  - JWST
  - HST
  - astronomy
  - exoplanets
  - spectroscopy
  - photometry
authors:
  - name: Taylor J. Bell
    orcid: 0000-0003-4177-2149
    affiliation: 1
  - name: Eva-Maria Ahrer
    orcid: 0000-0003-0973-8426
    affiliation: 2
  - name: Jonathan Brande
    orcid: 0000-0002-2072-6541
    affiliation: 3
  - name: Aarynn Carter
    orcid: 0000-0001-5365-4815
    affiliation: 4
  - name: Adina Feinstein
    orcid: 0000-0002-9464-8101
    affiliation: 5
  - name: Giannina Guzman
    orcid: 0000-0001-6340-8220
    affiliation: 6
  - name: Megan Mansfield
    orcid: 0000-0003-4241-7413
    affiliation: 7
  - name: Sebastian Zieba
    orcid: 0000-0003-0562-6750
    affiliation: 8
  - name: Caroline Piaulet
    orcid: 0000-0002-2875-917X
    affiliation: 9
  - name: Joseph Filippazzo
    orcid: 0000-0002-0201-8306
    affiliation: 10
  - name: Erin M. May
    orcid: 0000-0002-2739-1465
    affiliation: 11
  - name: Kevin B. Stevenson
    orcid: 0000-0002-7352-7941
    affiliation: 11
  - name: Laura Kreidberg
    orcid: 0000-0003-0514-1147
    affiliation: 8
affiliations:
  - name: BAER Institute, NASA Ames Research Center, Moffet Field, CA 94035, USA
    index: 1
  - name: Centre for Exoplanets and Habitability, University of Warwick, Gibbet Hill Road, CV4 7AL Coventry, UK
    index: 2
  - name: Department of Physics, Astronomy, University of Kansas, 1082 Malott, 1251 Wescoe Hall Dr., Lawrence, KS 66045, USA
    index: 3
  - name: Department of Astronomy and Astrophysics, University of California, Santa Cruz, CA 95064, USA
    index: 4
  - name: Department of Astronomy & Astrophysics, University of Chicago, 5640 S. Ellis Avenue, Chicago, IL 60637, USA
    index: 5
  - name: Department of Astronomy, University of Maryland, College Park, MD USA
    index: 6
  - name: Steward Observatory, University of Arizona, Tucson, AZ 85719, USA
    index: 7
  - name: Max-Planck-Institut für Astronomie, Königstuhl 17, D-69117 Heidelberg, Germany
    index: 8
  - name: Department of Physics and Institute for Research on Exoplanets, Université de Montréal, Montreal, QC, Canada
    index: 9
  - name: Space Telescope Science Institute, 3700 San Martin Drive, Baltimore, MD 21218, USA
    index: 10
  - name: Johns Hopkins APL, 11100 Johns Hopkins Road, Laurel, MD 20723, USA
    index: 11
date: 30 May 2022
bibliography: paper.bib

---

# Summary

`Eureka!` is a data reduction and analysis pipeline for exoplanet time series observations, with a particular focus on James Webb Space Telescope (JWST) observations. The goal of `Eureka!` is to provide an end-to-end pipline which starts with raw, uncalibrated FITS files and ultimately results in high quality exoplanet transmission and/or emission spectra. The pipeline has a modular structure with six stages, and each stage uses a "Eureka! Control File" (ECFs use the .ecf file extension) to allow for easy control of the pipeline's behaviour. We have provided template ECFs for the MIRI, NIRCam, NIRISS, and NIRSpec instruments on JWST and the WFC3 instrument on the Hubble Space Telescope (HST); these templates give users a good starting point for their analyses, but `Eureka!` is not intended to be used as a black-box tool, and users should expect to fine-tune some settings for each observation in order to achieve optimal results. Throughout the pipeline, many intermediate figures and outputs are created to allow users to compare `Eureka!`'s performance using different parameter settings or to compare `Eureka!` with an independent pipeline. The ECF used to run each stage is also copied into the output folder from each stage to increase reproducibility.


# Outline of `Eureka!`'s Stages

The six stages of `Eureka!` are as follows:

- Stage 1: An optional step that calibrates Raw data (converts ramps to slopes for JWST observations). This step can be skipped if you'd rather use STScI's `jwst` pipeline's Stage 1 outputs.
- Stage 2: An optional step that calibrates Stage 1 data (performs flatfielding, unit conversion, etc. for JWST observations). This step can be skipped if you'd rather use STScI's `jwst` pipeline's Stage 2 outputs, although it is recommended that you skip the photom step in that pipeline.
- Stage 3: Starts with the "_calints.fits" files produced by Stage 2 data and performs background subtraction and some final calibration. This stage also reduces the data in order to convert 2D spectra into a time series of 1D spectra or 2D photometric images into a 1D time series.
- Stage 4: Bins the 1D Spectra and generates light curves. Also removes 1D spectral drift/jitter and sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- Stage 6: Creates a table and a figure summarizing the transmission and/or emission spectra from your Stage 5 fit(s).

(INSERT OVERVIEW FLOWCHART OF EUREKA AND FLOWCHARTS OF EACH STAGE)


# Statement of Need

Transit spectrsocopy is really hard! (MORE TO COME)


# Documentation

Documentation for `Eureka!` is available at [https://eurekadocs.readthedocs.io/en/latest/](https://eurekadocs.readthedocs.io/en/latest/). 


# Similar Tools

jwst pipeline, exoplanet, juliet, POET, SPCA


# Acknowledgements

ERS collaboration

`Eureka!` allows for some variations upon the STScI's `jwst` pipeline for Stages 1 and 2, but presently these stages mostly act as wrappers around the `jwst` pipeline allowing this code to be run in the same way as the later stages of the `Eureka!` pipeline. `Eureka!` then uses its own custom code for additional calibration steps, spectral or photometric extraction, and light curve fitting. Several parts of the spectroscopy-focused code in Stages 3 and 4 of `Eureka!` were inspired by or were initially written for the kevin218/WFC3 pipeline (hosted at [https://github.com/kevin218/WFC3]). Other parts of the spectroscopy code and several parts of the photometry focused code in Stage 3 were inspired by or were initially written for the kevin218/POET pipeline (hosted at [https://github.com/kevin218/POET]; `@Stevenson:2012; @Cubillos:2013`).


# References
