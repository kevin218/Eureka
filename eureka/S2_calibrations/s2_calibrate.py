#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import jwst
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline

class EurekaS2Pipeline(Spec2Pipeline):

  def run_eureka(self, filename, skip_assign_wcs=False, skip_extract_2d=False, skip_srctype=False,
                 skip_flat_field=True, skip_photom=False, skip_extract_1d=False, output_dir=None, save_results=True):

    if output_dir==None or len(output_dir)==0:
      # If there was no output_dir provided, output into the same folder as the input files were in
      output_dir = '/'.join(filename.split('/')[:-1])
      if len(output_dir)>0:
        # Only add a trailing slash if the input file wasn't in the current working directory
        # (otherwise it'll try to save in the root directory)
        output_dir += '/'

    with fits.open(filename) as hdulist:
      # Figure out which instrument we are using
      inst = hdulist[0].header['INSTRUME']
      if inst == 'NIRSPEC':
        # Figure out what grating and filter we're using
        # Needed to change the aperture used for NIRSpec outputs
        grating = hdulist[0].header['GRATING']
        filt = hdulist[0].header['FILTER']

    if inst == 'NIRSPEC' and grating == 'PRISM':
      #Controls the cross-dispersion extraction
      self.assign_wcs.slit_y_low = -1
      self.assign_wcs.slit_y_high = 50
      # Modify the existing file to broaden the dispersion extraction - FIX: DOES NOT WORK CURRENTLY
      with datamodels.open(filename) as m:
        #Control the dispersion extraction - FIX: DOES NOT WORK CURRENTLY
        m.meta.wcsinfo.waverange_start = 6e-08
        m.meta.wcsinfo.waverange_end = 6e-06
        m.save(filename)
    elif inst == 'NIRSPEC':
      raise ValueError("I don't understand how to adjust the extraction aperture for this grating/filter yet!")

    # FIX: This will likely overwritten by the cfg file later on
    self.assign_wcs.skip = skip_assign_wcs
    self.extract_2d.skip = skip_extract_2d
    self.srctype.skip = skip_srctype
    self.flat_field.skip = skip_flat_field
    self.photom.skip = skip_photom
    self.extract_1d.skip = skip_extract_1d

    self.call(filename, output_dir=output_dir, save_results=save_results)

    if save_results:
      with datamodels.open(output_dir+'_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints.fits') as sp1d:
        fig, ax = plt.subplots(1,1, figsize=[15,5])
        
        for i in range(len(sp1d.spec)):
            plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'], sp1d.spec[i].spec_table['FLUX'])

        plt.title('Time Series Observation: Extracted spectra')
        plt.xlabel('Wavelenth (micron)')
        plt.ylabel('Flux')
        plt.savefig(output_dir+'_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints.png', bbox_inches='tight', dpi=300)
        plt.close()

def main(filename):
  pipeline = EurekaS2Pipeline()
  # FIX: Check ecf file to get skip booleans and enter them here
  pipeline.run_eureka(filename)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)==1:
      filename = args
    else:
      raise ValueError('You must enter the science filename! (expected 1 arg, got {})'.format(len(args)))
    main(filename)
