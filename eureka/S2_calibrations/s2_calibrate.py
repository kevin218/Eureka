#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline

"""
# Proposed Steps
# -------- -----
# 1.  Read in Stage 1 data products
# 2.  Change default trimming if needed
# 3.  Run the JWST pipeline with any requested modifications
# 4.  Save Stage 2 data products
# 5.  Produce plots
"""

import os, sys, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline
from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import readECF as rd

class MetaClass:
    def __init__(self):
        return

class EurekaS2Pipeline(Spec2Pipeline):

  def run_eurekaS2(self, eventlabel):
    '''
    Reduces rateints files ouput from Stage 1 of the JWST pipeline into calints and x1dints.

    Parameters
    ----------
    eventlabel  : str, Unique label for this dataset

    Returns
    -------
    meta        : Metadata object

    Remarks
    -------


    History
    -------
    Code fragments written by Eva-Maria Ahrer and Aarynn Carter   June 2021
    Written by Taylor Bell      October 2021

    '''

    t0 = time.time()

    # Initialize metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel
    meta.suffix = 'rateints'  # This will break for any instruments/observations that do not result in rateints

    # Load Eureka! control file and store values in Event object
    ecffile = 'S2_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Create directories for Stage 2 processing outputs
    # This tempdir code allows the input and output files to be stored outside of the Eureka! folder
    tempdir = meta.topdir
    # Do some annoying stuff to make sure we end up with a valid path
    if tempdir[-1]=='/':
      tempdir = tempdir[:-1]
    if meta.outputdir[0]=='/':
      tempdir += meta.outputdir
    else:
      tempdir += '/'+meta.outputdir
    if tempdir[-1]!='/':
      tempdir += '/'
    run = util.makedirectory(meta, tempdir+'S2')
    meta.workdir = util.pathdirectory(meta, tempdir+'S2', run)
    # Add a trailing slash so we don't need to add it everywhere below
    meta.workdir += '/'
    # Make a separate folder for plot outputs
    if not os.path.exists(meta.workdir+'figs'):
        os.makedirs(meta.workdir+'figs')

    # Output S2 log file
    meta.logname = meta.workdir + 'S2_' + meta.eventlabel + ".log"
    log = logedit.Logedit(meta.logname)
    log.writelog("\nStarting Stage 2 Reduction")

    # Copy ecf
    log.writelog('Copying S2 control file')
    shutil.copy(ecffile, meta.workdir)

    # Create list of file segments
    meta = util.readfiles(meta)
    num_data_files = len(meta.segment_list)
    log.writelog(f'\nFound {num_data_files} data file(s) ending in {meta.suffix}.fits')

    # If testing, only run the last file
    if meta.testing_S2:
        istart = num_data_files - 1
    else:
        istart = 0

    # Run the pipeline on each file sequentially
    for m in range(istart, num_data_files):
      # Report progress
      log.writelog(f'Starting file {m + 1} of {num_data_files}')
      filename = meta.segment_list[m]

      with fits.open(filename) as hdulist:
        # Figure out which instrument we are using
        inst = hdulist[0].header['INSTRUME']
        if inst == 'NIRSPEC':
          # Figure out what grating and filter we're using
          # (needed to change the aperture used for NIRSpec outputs)
          grating = hdulist[0].header['GRATING']
          filt = hdulist[0].header['FILTER']

      if inst == 'NIRSPEC' and grating == 'PRISM':
        #Controls the cross-dispersion extraction - FIX: check if this is overridden
        self.assign_wcs.slit_y_low = meta.slit_y_low
        self.assign_wcs.slit_y_high = meta.slit_y_high
        # Modify the existing file to broaden the dispersion extraction
        with datamodels.open(filename) as m:
          #Control the dispersion extraction - FIX: Does not actually change dispersion direction extraction
          log.writelog('Editing (in place) the waverange in the input file')
          m.meta.wcsinfo.waverange_start = meta.waverange_start
          m.meta.wcsinfo.waverange_end = meta.waverange_end
          m.save(filename)
      elif inst == 'NIRSPEC':
        raise ValueError("I don't understand how to adjust the extraction aperture for this grating/filter yet!")

      # Skip steps according to input ecf file
      self.bkg_subtract.skip = meta.skip_bkg_subtract
      self.imprint_subtract.skip = meta.skip_imprint_subtract
      self.msa_flagging.skip = meta.skip_msa_flagging
      self.extract_2d.skip = meta.skip_extract_2d
      self.srctype.skip = meta.skip_srctype
      self.master_background.skip = meta.skip_master_background
      self.wavecorr.skip = meta.skip_wavecorr
      self.flat_field.skip = meta.skip_flat_field
      self.straylight.skip = meta.skip_straylight
      self.fringe.skip = meta.skip_fringe
      self.pathloss.skip = meta.skip_pathloss
      self.barshadow.skip = meta.skip_barshadow
      self.photom.skip = meta.skip_photom
      self.resample_spec.skip = meta.skip_resample_spec
      self.cube_build.skip = meta.skip_cube_build
      self.extract_1d.skip = meta.skip_extract_1d
      # Save outputs if requested to the folder specified in the ecf
      self.save_results = (not meta.testing_S2)
      self.output_dir = meta.workdir

      # Call the main Spec2Pipeline function (defined in the parent class)
      log.writelog('Running the Spec2Pipeline')
      # Must call the pipeline in this way to ensure the skip booleans are respected
      self(filename)

      # Produce some summary plots if requested
      if not meta.testing_S2 and not self.extract_1d.skip:
        log.writelog('Generating x1dints figure')
        with datamodels.open(meta.workdir+'_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints.fits') as sp1d:
          fig, ax = plt.subplots(1,1, figsize=[15,5])
          
          for i in range(len(sp1d.spec)):
              plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'], sp1d.spec[i].spec_table['FLUX'])

          plt.title('Time Series Observation: Extracted spectra')
          plt.xlabel('Wavelenth (micron)')
          plt.ylabel('Flux')
          plt.savefig(meta.workdir+'figs/'+'_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints.png', bbox_inches='tight', dpi=300)
          plt.close()

    # Calculate total run time
    total = (time.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    # Save results
    if not meta.testing_S2:
      log.writelog('Saving Metadata')
      me.saveevent(meta, meta.workdir + 'S2_' + meta.eventlabel + "_Meta_Save", save=[])

    return meta
