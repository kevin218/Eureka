import os, sys, shutil, time
import numpy as np
from astropy.io import fits

from jwst import datamodels
from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

from eureka.S1_detector_processing.ramp_fitting import Eureka_RampFitStep

from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import readECF as rd

class MetaClass:
	def __init__(self):
		return

class EurekaS1Pipeline(Detector1Pipeline):

	def run_eurekaS1(self, eventlabel):
		'''
		Process a Stage 0, *_uncal.fits file to Stage 1 *_rate.fits and *_rateints.fits files. 
		Steps taken to perform this processing can follow the default JWST pipeline, or alternative methods.  

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
			Code fragments from Taylor Bell					October 2021
			Written by Aarynn Carter and Eva-Maria Ahrer   	October 2021
		'''
		t0 = time.time()

		# Initialize metadata object
		meta = MetaClass()
		meta.eventlabel = eventlabel
		meta.suffix = 'uncal'  # This will break for any instruments/observations that do not result in uncal

		# Load Eureka! control file and store values in Event object
		ecffile = 'S1_' + eventlabel + '.ecf'
		ecf     = rd.read_ecf(ecffile)
		rd.store_ecf(meta, ecf)

		# Shouldn't be too relevant for Stage 1, but assign raw input and output directories
		meta.inputdir_raw = meta.inputdir
		meta.outputdir_raw = meta.outputdir

		# Create directories for Stage 1 processing outputs
		# This code allows the input and output files to be stored outside of the Eureka! folder
		outputdir = os.path.join(meta.topdir, *meta.outputdir.split(os.sep))
		if outputdir[-1]!='/':
		  outputdir += '/'
		run = util.makedirectory(meta, 'S1')
		meta.workdir = util.pathdirectory(meta, 'S1', run)
		# Add a trailing slash so we don't need to add it everywhere below
		meta.workdir += '/'
		# Make a separate folder for plot outputs
		if not os.path.exists(meta.workdir+'figs'):
			os.makedirs(meta.workdir+'figs')

		# Output S2 log file
		meta.logname = meta.workdir + 'S1_' + meta.eventlabel + ".log"
		log = logedit.Logedit(meta.logname)
		log.writelog("\nStarting Stage 1 Processing")

		# Copy ecf
		log.writelog('Copying S1 control file')
		shutil.copy(ecffile, meta.workdir)

		# Create list of file segments
		meta = util.readfiles(meta)
		num_data_files = len(meta.segment_list)
		log.writelog(f'\nFound {num_data_files} data file(s) ending in {meta.suffix}.fits')

		# If testing, only run the last file
		if meta.testing_S1:
			istart = num_data_files - 1
		else:
			istart = 0

		# Run the pipeline on each file sequentially
		for m in range(istart, num_data_files):
			# Report progress
			log.writelog(f'Starting file {m + 1} of {num_data_files}')
			filename = meta.segment_list[m]

			with fits.open(filename) as f:
				instrument = f[0].header['INSTRUME'] 

			# Reset suffix and assign whether to save and the output directory
			self.suffix = None
			self.save_results = (not meta.testing_S1)
			self.output_dir = meta.outputdir
			
			# Instrument Non-Specific Steps
			self.group_scale.skip = meta.skip_group_scale
			self.dq_init.skip = meta.skip_dq_init
			self.saturation.skip = meta.skip_saturation
			self.ipc.skip = meta.skip_ipc
			self.refpix.skip = meta.skip_refpix
			self.linearity.skip = meta.skip_linearity
			self.dark_current.skip = meta.skip_dark_current
			self.jump.skip = meta.skip_jump
			self.gain_scale.skip = meta.skip_gain_scale

			# Instrument Specific Steps
			if instrument in ['NIRCAM', 'NIRISS', 'NIRSPEC']:
				self.persistence.skip = meta.skip_persistence
				self.superbias.skip = meta.skip_superbias
			elif instrument in ['MIRI']:
				self.firstframe.skip = meta.skip_firstframe
				self.lastframe.skip = meta.skip_lastframe
				self.rscd.skip = meta.skip_rscd

			# Define ramp fitting procedure
			self.ramp_fit = Eureka_RampFitStep()
			self.ramp_fit.algorithm = meta.ramp_fit_algorithm
			self.ramp_fit.maximum_cores = meta.ramp_fit_max_cores
			self.ramp_fit.skip = meta.skip_ramp_fitting

			# Default ramp fitting settings
			if self.ramp_fit.algorithm == 'default':
				self.ramp_fit.weighting = meta.default_ramp_fit_weighting
				# Some weighting methods need additional parameters
				if self.ramp_fit.weighting == 'fixed':
					self.ramp_fit.fixed_exponent = meta.default_ramp_fit_fixed_exponent
				elif self.ramp_fit.weighting == 'custom':
					self.ramp_fit.custom_snr_bounds = meta.default_ramp_fit_custom_snr_bounds
					self.ramp_fit.custom_exponents = meta.default_ramp_fit_custom_exponents
			with fits.open(filename, mode='update') as hdulist:
				if hdulist[0].header['INSTRUME']=='NIRCam':
				# jwst 1.3.3 breaks unless NDITHPTS and NRIMDTPT are integers rather than the strings that they are in the old simulated NIRCam data, code from Stage 2 calibration
					hdulist[0].header['NDITHPTS'] = 1
					hdulist[0].header['NRIMDTPT'] = 1
			# Run Stage 1
			self(filename)

		# Calculate total run time
		total = (time.time() - t0) / 60.
		log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

		# Save results
		if not meta.testing_S1:
			log.writelog('Saving Metadata')
			me.saveevent(meta, meta.workdir + 'S1_' + meta.eventlabel + "_Meta_Save", save=[])

		return meta
