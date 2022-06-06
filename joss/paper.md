---
title: '`Eureka!`: An End-to-End Pipeline for JWST Time-Series Observations'
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
  - name: Aarynn L. Carter
    orcid: 0000-0001-5365-4815
    affiliation: 4
  - name: Adina D. Feinstein
    orcid: 0000-0002-9464-8101
    affiliation: 5
  - name: Giannina {Guzman Caloca}
    orcid: 0000-0001-6340-8220
    affiliation: 6
  - name: Megan Mansfield
    orcid: 0000-0003-4241-7413
    affiliation: "7, 8"
  - name: Sebastian Zieba
    orcid: 0000-0003-0562-6750
    affiliation: 9
  - name: Caroline Piaulet
    orcid: 0000-0002-2875-917X
    affiliation: 10
  - name: Joseph Filippazzo
    orcid: 0000-0002-0201-8306
    affiliation: 11
  - name: Erin M. May
    orcid: 0000-0002-2739-1465
    affiliation: 12
  - name: Laura Kreidberg
    orcid: 0000-0003-0514-1147
    affiliation: 9
  - name: Kevin B. Stevenson
    orcid: 0000-0002-7352-7941
    affiliation: 12
affiliations:
  - name: BAER Institute, NASA Ames Research Center, Moffet Field, CA 94035, USA
    index: 1
  - name: Department of Physics, University of Warwick, Gibbet Hill Road, CV4 7AL Coventry, UK
    index: 2
  - name: Department of Physics and Astronomy, University of Kansas, 1082 Malott, 1251 Wescoe Hall Dr., Lawrence, KS 66045, USA
    index: 3
  - name: Department of Astronomy and Astrophysics, University of California, Santa Cruz, 1156 High Street, Santa Cruz, CA 95064, USA
    index: 4
  - name: Department of Astronomy & Astrophysics, University of Chicago, 5640 S. Ellis Avenue, Chicago, IL 60637, USA
    index: 5
  - name: Department of Astronomy, University of Maryland, College Park, MD USA
    index: 6
  - name: Steward Observatory, University of Arizona, Tucson, AZ 85719, USA
    index: 7
  - name: NHFP Sagan Fellow
    index: 8
  - name: Max-Planck-Institut für Astronomie, Königstuhl 17, D-69117 Heidelberg, Germany
    index: 9
  - name: Department of Physics and Institute for Research on Exoplanets, Université de Montréal, Montreal, QC, Canada
    index: 10
  - name: Space Telescope Science Institute, 3700 San Martin Drive, Baltimore, MD 21218, USA
    index: 11
  - name: Johns Hopkins APL, 11100 Johns Hopkins Road, Laurel, MD 20723, USA
    index: 12
date: 30 May 2022
bibliography: paper.bib

---

# Summary

`Eureka!` is a data reduction and analysis pipeline for exoplanet time-series observations, with a particular focus on James Webb Space Telescope (JWST) data. The goal of `Eureka!` is to provide an end-to-end pipeline that starts with raw, uncalibrated FITS files and ultimately yields precise exoplanet transmission and/or emission spectra. The pipeline has a modular structure with six stages, and each stage uses a "Eureka! Control File" (ECF; these files use the .ecf file extension) to allow for easy control of the pipeline's behavior. Stage 5 also uses a "Eureka! Parameter File" (EPF; these files use the .epf file extension) to control the fitted parameters. We provide template ECFs for the MIRI, NIRCam, NIRISS, and NIRSpec instruments on JWST and the WFC3 instrument on the Hubble Space Telescope (HST). These templates give users a good starting point for their analyses, but `Eureka!` is not intended to be used as a black-box tool, and users should expect to fine-tune some settings for each observation in order to achieve optimal results. At each stage, the pipeline creates intermediate figures and outputs that allow users to compare `Eureka!`'s performance using different parameter settings or to compare `Eureka!` with an independent pipeline. The ECF used to run each stage is also copied into the output folder from each stage to enhance reproducibility. Finally, while `Eureka!` has been optimized for exoplanet observations (especially the latter stages of the code), much of the core functionality could also be repurposed for JWST time-series observations in other research domains thanks to `Eureka!`'s modularity.


# Outline of `Eureka!`'s Stages

`Eureka!` is broken down into six stages, which are as follows (also summarized in \autoref{fig:overview}):

- Stage 1: An optional step that calibrates raw data (converts ramps to slopes for JWST observations). This step can be skipped within `Eureka!` if you would rather use the Stage 1 outputs from the `jwst` pipeline.
- Stage 2: An optional step that further calibrates Stage 1 data (performs flat-fielding, unit conversion, etc. for JWST observations). This step can be skipped within `Eureka!` if you would rather use the Stage 2 outputs from the `jwst` pipeline.
- Stage 3: Using Stage 2 outputs, performs background subtraction and optimal spectral extraction. For spectroscopic observations, this stage generates a time series of 1D spectra. For photometric observations, this stage generates a single light curve of flux versus time.
- Stage 4: Using Stage 3 outputs, generates spectroscopic light curves by binning the time series of 1D spectra along the wavelength axis. Optionally removes drift/jitter along the dispersion direction and/or sigma clips outliers.
- Stage 5: Fits the light curves with noise and astrophysical models using different optimization or sampling algorithms.
- Stage 6: Displays the planet spectrum in figure and table form using results from the Stage 5 fits.

![An overview flowchart showing the processing done at each stage in `Eureka!`. The outputs of each stage are used as the inputs to the subsequent stage along with the relevant settings file(s). \label{fig:overview}](figures/stages_flowchart.png){width=100%}

# Statement of Need

The calibration, reduction, and fitting of exoplanet time-series observations is a challenging problem with many tunable parameters across many stages, many of which will significantly impact the final results. Typically, the default calibration pipeline from astronomical observatories is insufficiently tailored for exoplanet time-series observations as the pipeline is more optimized for other science use cases. As such, it is common practice to develop a custom data analysis pipeline that starts from the original, uncalibrated images. Historically, data analysis pipelines have often been proprietary, so each new user of an instrument or telescope has had to develop their own pipeline. Also, clearly specifying the analysis procedure can be challenging, especially with proprietary code, which erodes reproducibility. `Eureka!` seeks to be a next-generation data analysis pipeline for next-generation observations from JWST with open-source and well-documented code for easier adoption; modular code for easier customization while maintaining a consistent framework; and easy-to-use but powerful inputs and outputs for increased automation, increased reproducibility, and more thorough intercomparisons. By also allowing for analyses of HST observations within the same framework, users will be able to combine new and old observations to develop a more complete understanding of individual targets or even entire populations.


# Documentation

Documentation for `Eureka!` is available at [https://eurekadocs.readthedocs.io/en/latest/](https://eurekadocs.readthedocs.io/en/latest/). 


# Similar Tools

[`jwst`](https://github.com/spacetelescope/jwst) [@jwst:2022], [`exoplanet`](https://github.com/exoplanet-dev/exoplanet) [@exoplanet:2021], [`juliet`](https://github.com/nespinoza/juliet) [@juliet:2019], [`POET`](https://github.com/kevin218/POET) [@Stevenson:2012; @Cubillos:2013], [`WFC3`](https://github.com/kevin218/WFC3) [@Stevenson:2014a], [`PACMAN`](https://github.com/sebastian-zieba/PACMAN) [@Kreidberg:2014; @pacman:2022], [`SPCA`](https://github.com/lisadang27/SPCA) [@Dang:2018; @Bell:2021]


# Acknowledgements

`Eureka!` allows for some variations upon the STScI's [`jwst`](https://github.com/spacetelescope/jwst) pipeline [@jwst:2022] for Stages 1 and 2, but presently these stages mostly act as wrappers around the `jwst` pipeline. This allows `Eureka!` to run the `jwst` pipeline in the same manner as `Eureka!`'s latter stages. `Eureka!` then uses its own custom code for additional calibration steps, spectral or photometric extraction, and light curve fitting. Several parts of the spectroscopy-focused code in Stages 3 and 4 of `Eureka!` were inspired by, or were initially written for, the [`WFC3`](https://github.com/kevin218/WFC3) [@Stevenson:2014a] pipeline. Other parts of the spectroscopy code and several parts of the photometry focused code in Stage 3 were inspired by, or were initially written for, the [`POET`](https://github.com/kevin218/POET) pipeline [@Stevenson:2012; @Cubillos:2013]. Some of the Stage 5 code comes from @Kreidberg:2014 and [`PACMAN`](https://github.com/sebastian-zieba/PACMAN) [@pacman:2022]. Small pieces of the [`SPCA`](https://github.com/lisadang27/SPCA) [@Dang:2018; @Bell:2021] and [`Bell_EBM`](https://github.com/taylorbell57/Bell_EBM) [@Bell:2018] repositories have also been reused.

ALC is supported by a grant from STScI (_JWST_-ERS-01386) under NASA contract NAS5-03127. ADF acknowledges support by the National Science Foundation Graduate Research Fellowship Program under Grant No. (DGE-1746045). CP acknowledges financial support by the Fonds de Recherche Québécois—Nature et Technologie (FRQNT; Québec), the Technologies for Exo-Planetary Science (TEPS) Trainee Program and the Natural Sciences and Engineering Research Council (NSERC) Vanier Scholarship. JB acknowledges support from the NASA Interdisciplinary Consortia for Astrobiology Research (ICAR). KBS is supported by _JWST_-ERS-01366. MM acknowledges support through the NASA Hubble Fellowship grant HST-HF2-51485.001-A awarded by STScI, which is operated by the Association of Universities for Research in Astronomy, Inc., for NASA, under contract NAS5-26555. We also thank Ivelina Momcheva for useful discussions. Support for this work was provided in part by NASA through a grant from the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-03127. In addition, we would like to thank the Transiting Exoplanet Community Early Release Science program for organizing meetings that contributed to the writing of `Eureka!`.


# References
