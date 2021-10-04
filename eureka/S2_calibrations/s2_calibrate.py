#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline

import os, sys, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import jwst
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline
from ..lib import logedit
from ..lib import util

from ..lib import readECF as rd

class MetaClass:
    def __init__(self):
        return

class EurekaS2Pipeline(Spec2Pipeline):

  def run_eurekaS2(self, eventlabel):

    # Initialize metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Load Eureka! control file and store values in Event object
    ecffile = 'S2_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Create directories for Stage 2 processing
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S')
    run = util.makedirectory(meta, 'S2')
    meta.workdir = util.pathdirectory(meta, 'S2', run)
    meta.workdir = './'+meta.workdir+'/'
    # meta.workdir = os.path.abspath(os.path.expanduser(os.path.expandvars(meta.workdir)))
    # meta.workdir = '/home/taylor/Downloads/'+meta.workdir
    # meta.workdir = '/home/taylor/Downloads/'
    print(meta.workdir)
    if not os.path.exists(meta.workdir+'figs'):
        os.makedirs(meta.workdir+'figs')

    # Output S2 log file
    meta.logname = meta.workdir + 'S2_' + meta.eventlabel + ".log"
    log = logedit.Logedit(meta.logname)
    log.writelog("\nStarting Stage 2 Reduction")

    # Copy ecf
    log.writelog('Copying S2 control file')
    shutil.copy(ecffile, meta.workdir)

    with fits.open(meta.filename) as hdulist:
      # Figure out which instrument we are using
      inst = hdulist[0].header['INSTRUME']
      if inst == 'NIRSPEC':
        # Figure out what grating and filter we're using
        # Needed to change the aperture used for NIRSpec outputs
        grating = hdulist[0].header['GRATING']
        filt = hdulist[0].header['FILTER']

    if inst == 'NIRSPEC' and grating == 'PRISM':
      #Controls the cross-dispersion extraction
      self.assign_wcs.slit_y_low = meta.slit_y_low
      self.assign_wcs.slit_y_high = meta.slit_y_high
      # Modify the existing file to broaden the dispersion extraction - FIX: DOES NOT WORK CURRENTLY
      with datamodels.open(meta.filename) as m:
        #Control the dispersion extraction - FIX: DOES NOT WORK CURRENTLY
        m.meta.wcsinfo.waverange_start = meta.waverange_start
        m.meta.wcsinfo.waverange_end = meta.waverange_end
        m.save(meta.filename)
    elif inst == 'NIRSPEC':
      raise ValueError("I don't understand how to adjust the extraction aperture for this grating/filter yet!")

    # FIX: This will likely overwritten by the cfg file later on
    self.assign_wcs.skip = meta.skip_assign_wcs
    self.extract_2d.skip = meta.skip_extract_2d
    self.srctype.skip = meta.skip_srctype
    self.flat_field.skip = meta.skip_flat_field
    self.photom.skip = meta.skip_photom
    self.extract_1d.skip = meta.skip_extract_1d

    self.call(meta.filename, output_dir=meta.workdir, save_results=True)

    if meta.save_results:
      with datamodels.open(meta.workdir+'_'.join(meta.filename.split('/')[-1].split('_')[:-1])+'_x1dints.fits') as sp1d:
        fig, ax = plt.subplots(1,1, figsize=[15,5])
        
        for i in range(len(sp1d.spec)):
            plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'], sp1d.spec[i].spec_table['FLUX'])

        plt.title('Time Series Observation: Extracted spectra')
        plt.xlabel('Wavelenth (micron)')
        plt.ylabel('Flux')
        plt.savefig(meta.workdir+'figs/'+'_'.join(meta.filename.split('/')[-1].split('_')[:-1])+'_x1dints.png', bbox_inches='tight', dpi=300)
        plt.close()

    return meta
