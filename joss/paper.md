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
  - name: Taylor James Bell
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

Eureka! is a data reduction and analysis pipeline for exoplanet time series observations, with a particular focus on James Webb Space Telescope observations. The pipeline is end-to-end, and includes calibration, spectral extraction, light curve fitting, high level outputs, ... 


# Statement of need
Transit spectrsocopy is really hard!


# Outline of the stages
The Eureka! pipeline is broken into 6 different stages that allow users to go from raw, uncalibrated FITS files to transmission and/or emission spectra. The stages are as follows:

- Stage 1: An optional step that calibrates Raw data (converts ramps to slopes). This step can be skipped if you’d rather use STScI’s JWST pipeline’s Stage 1 outputs.
- Stage 2: An optional step that calibrates Stage 1 data (performs flatfielding, unit conversion, etc.). This step can be skipped if you’d rather use STScI’s JWST pipeline’s Stage 2 outputs.
- Stage 3: Starts with Stage 2 data and further calibrates (performs background subtraction, etc.) and reduces the data in order to convert 2D spectra into a time-series of 1D spectra.
- Stage 4: Bins the 1D Spectra and generates light curves. Also removes 1D spectral drift/jitter and sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models.
- Stage 6: Creates a table and a figure summarizing the transmission and/or emission spectra from your many fits.


# Documentation

Documentation for `Eureka!` is available at [https://eurekadocs.readthedocs.io/en/latest/](https://eurekadocs.readthedocs.io/en/latest/). 

# Similar tools

jwst pipeline, exoplanet, juliet, POET, SPCA


# Acknowledgements

ERS collaboration

# References


