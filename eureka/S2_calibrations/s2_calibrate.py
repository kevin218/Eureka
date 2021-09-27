#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline

import os
import numpy as np
from astropy.io import fits
import jwst
from jwst import datamodels, assign_wcs, extract_2d, srctype, photom, flatfield

# class MetaClass:
#     def __init__(self):
#         # initialize Univ
#         # Univ.__init__(self)
#         # self.initpars(ecf)
#         # self.foo = 2
#         return
# 
#
# class DataClass:
#     def __init__(self):
#         # initialize Univ
#         # Univ.__init__(self)
#         # self.initpars(ecf)
#         # self.foo = 2
#         return


def calibrate_JWST(filename, do_assignwcs=True, do_extract2d=True, do_srctype=True, do_flatfield=False, do_photom=True, delete_files=True):
	'''
	Processes a Stage 1, *_rateints.fits to an equivalent of a Stage 2 *_calints.fits file, except the full subarray will be extracted instead of a trimmed 2D image.
	Options to turn off and on certain steps are available, but are primarily for debugging
	purposes - there is no guarantee that Eureka will work if you skip some of the steps.

	Parameters
	----------
	filename			: Single filename to read
	do_assignwcs		: Boolean to perform the Assign WCS pipeline step
	do_extract2d		: Boolean to perform the Extract 2D pipeline step
	do_srctype			: Boolean to perform the Source Type pipeline step
	do_flatfield		: Boolean to perform the Flat Field pipeline step
	do_photom			: Boolean to perform the Photometric correction pipeline step
	delete_files		: Boolean to delete intermediate files produced by the steps.

	Returns
	-------
	current_file		: Array of data frames

	History
	-------
	Written by Aarynn Carter / Eva-Maria Ahrer  June 2021
	'''

	# Make sure we have the full directory location of the file
	current_file = os.path.abspath(filename)

	# Get the directory the data is saved in, and a prefix for the file itself.
	file_dir = '/'.join(current_file.split('/')[:-1])
	file_prefix = current_file.split('/')[-1].split('_rateints.fits')[0]

	# Grab location of config files
	configs = jwst.__file__.split('__init__.py')[0] + 'pipeline/'

	# Assign World Coordinate System - this is the step where we need to modify values to ensure the full subarray is extracted.
	if do_assignwcs:
		#Read in the *rateints.fits file so we know what grating/filter we are using.
		with fits.open(current_file) as hdulist:
			grating = hdulist[0].header['GRATING']
			filt = hdulist[0].header['FILTER']

		# Set extraction values based on the grating/filter.
		if grating == 'PRISM':
			slit_y_low, slit_y_high = -1, 50 #Controls the cross-dispersion extraction
			wav_start, wav_end = 6e-08, 6e-06   #Control the dispersion extraction - DOES NOT WORK CURRENTLY
		else:
			raise ValueError("I don't understand how to adjust the extraction aperture for this grating/filter yet!")

		# Modify the existing file to broaden the dispersion extraction - DOES NOT WORK CURRENTLY
		with datamodels.open(filename) as m:
			m.meta.wcsinfo.waverange_start = wav_start
			m.meta.wcsinfo.waverange_end = wav_end
			m.save(filename)

		# Run the step, note that the cross-dispersion extraction is input here.
		stepname = 'assign_wcs'
		curr_result = assign_wcs.AssignWcsStep.call(current_file, config_file=configs+stepname+'.cfg', slit_y_low=slit_y_low, slit_y_high=slit_y_high, output_dir=file_dir, output_file=file_prefix+'_{}.fits'.format(stepname))
		# Update the current file we are working with.
		current_file = file_dir + '/' + file_prefix+'_{}.fits'.format(stepname)

	# Extract 2D spectrum, also does wavelength calibration
	if do_extract2d:
		stepname = 'extract_2d'
		result = extract_2d.Extract2dStep.call(curr_result, config_file=configs+stepname+'.cfg', output_dir=file_dir, output_file=file_prefix+'_{}.fits'.format(stepname))
		current_file = file_dir + '/' + file_prefix+'_{}.fits'.format(stepname)

	# Identify source type
	if do_srctype:
		stepname = 'srctype'
		result = srctype.SourceTypeStep.call(current_file, config_file=configs+stepname+'.cfg', output_dir=file_dir, output_file=file_prefix+'_{}.fits'.format(stepname))
		current_file = file_dir + '/' + file_prefix+'_{}.fits'.format(stepname)

	# Perform flat field correction
	# ***NOTE*** At the time the NIRSpec ERS Hackathon simulated data was created, this step did not work correctly and is by default turned off.
	if do_flatfield:
		stepname = 'flat_field'
		result = flatfield.FlatFieldStep.call(current_file, config_file=configs+stepname+'.cfg', output_dir=file_dir, output_file=file_prefix+'_{}.fits'.format(stepname))
		current_file = file_dir + '/' + file_prefix+'_{}.fits'.format(stepname)

	# Perform photometric correction to pixel count values.
	if do_photom:
		stepname = 'photom'
		result = photom.PhotomStep.call(current_file, config_file=configs+stepname+'.cfg', output_dir=file_dir, output_file=file_prefix+'_{}.fits'.format(stepname))
		current_file = file_dir + '/' + file_prefix+'_{}.fits'.format(stepname)

	# Delete any intermediate files that were produced to keep things clean.
	if delete_files:
		to_remove = ['_assign_wcs.fits', '_extract_2d.fits', '_srctype.fits', '_flat_field.fits']
		if do_photom:
			to_remove = to_remove
		elif do_flatfield:
			to_remove = to_remove[:-1]
		elif do_srctype:
			to_remove = to_remove[:-2]
		elif do_extract2d:
			to_remove = to_remove[:-3]
		else:
			to_remove = []

		for file_suffix in to_remove:
			del_file = file_dir + '/' + file_prefix + file_suffix
			if os.path.exists(del_file):
				os.remove(del_file)

	return current_file
