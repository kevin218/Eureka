# NIRSpec specific rountines go here
import os
import numpy as np
from astropy.io import fits
import jwst
from jwst import datamodels
from jwst import assign_wcs, extract_2d, srctype, photom, flatfield

def process_to_stage2(filename, do_assignwcs=True, do_extract2d=True, do_srctype=True, do_flatfield=False, do_photom=True, delete_files=True, output_dir=None):
    ''' 
    Processes a Stage 1, *_rateints.fits to an equivalent of a Stage 2 *_calints.fits file, except the full subarray with be extracted instead of a trimmed 2D image.
    Options to turn off and on certain steps are available, but are primarily for debugging
    purposes - there is no guarantee that Eureka will work if you skip some of the steps. 

    ''' 
    rootname = filename.split('/')[-1].split('_rateints.fits')[0]
    output_rootname = output_dir + rootname
    current_file = filename

    # Grab location of configs file
    configs = jwst.__file__.split('__init__.py')[0] + 'pipeline/'

    # Assign World Coordinate System - this is the step where we need to modify values to ensure the full subarray is extracted. 
    if do_assignwcs:
        #Read in the *rateints.fits file so we know what grating/filter we are using. 
        with fits.open(filename) as hdulist:
            grating = hdulist[0].header['GRATING']
            filt = hdulist[0].header['FILTER']

        # Set extraction values based on the grating/filter.
        if grating == 'PRISM':
            slit_y_low, slit_y_high = -1, 50 #Controls the cross-dispersion extraction
            wav_start, wav_end = 6e-08, 6e-06   #Control the dispersion extraction
        else:
            raise ValueError("I don't understand how to adjust the extraction aperture for this grating/filter yet!") 

        # Modify the existing file to broaden the dispersion extraction. 
        with datamodels.open(filename) as m:
            m.meta.wcsinfo.waverange_start = wav_start
            m.meta.wcsinfo.waverange_end = wav_end
            m.save(filename)

        # Run the step, note that the cross-dispersion extraction is input here. 
        stepname = 'assign_wcs'
        curr_result = assign_wcs.AssignWcsStep.call(current_file, config_file=configs+stepname+'.cfg', slit_y_low=slit_y_low, slit_y_high=slit_y_high, output_file=filename.replace('rateints.fits', '{}.fits'.format(stepname)), output_dir=output_dir)
        current_file = output_rootname+'_{}.fits'.format(stepname)

    # Extract 2D spectrum, also does wavelength calibration
    if do_extract2d:
        result = extract_2d.Extract2dStep.call(curr_result, config_file=configs+'extract_2d.cfg', output_file=filename, output_dir=output_dir)
        current_file = output_rootname+'_extract_2d.fits'

    # Identify source type
    if do_srctype:
        result = srctype.SourceTypeStep.call(current_file, config_file=configs+'srctype.cfg', output_file=filename, output_dir=output_dir)
        current_file = output_rootname+'_srctype.fits'

    # Perform flat field correction
    # ***NOTE*** At the time the NIRSpec ERS Hackathon simulated data was created, this step did not work correctly and is by default turned off. 
    if do_flatfield:
        result = flatfield.FlatFieldStep.call(current_file, config_file=configs+'flat_field.cfg', output_file=filename, output_dir=output_dir)
        current_file = output_rootname+'_flat_field.fits'

    # Perform photometric correction to pixel count values.
    if do_photom:
        result = photom.PhotomStep.call(current_file, config_file=configs+'photom.cfg', output_file=filename, output_dir=output_dir)
        current_file = output_rootname+'_photom.fits'
        
        
    # Delete any intermediate files that were produced to keep things clean.
    if delete_files:
        to_remove = ['_assign_wcs.fits', '_extract_2d.fits', '_srctype.fits', '_flat_field.fits']
        if do_photom: 
            to_remove = to_remove
        elif do_flatfield:
            to_remove = to_remove[:-1]
            print(to_remove)
        elif do_srctype:
            to_remove = to_remove[:-2]
        elif do_extact2d:
            to_remove = to_remove[:-3]
        else:
            to_remove = []

        for suffix in to_remove:
            del_file = output_rootname+suffix
            if os.path.exists(del_file):
                os.remove(del_file)

    return current_file

def read(filename, data, do_assignwcs=True, do_extract2d=True, do_srctype=True, do_flatfield=False, do_photom=True, delete_files=True, output_dir=None, stage_1_process = True):
    '''
    Reads single FITS file from JWST's NIRSpec instrument.

    Parameters
    ----------
    filename          : Single filename to read, should be the Stage 1, *_rateints.fits file. 
    data              : data object in which the fits data will stored
    returnHdr         : Set True to return header files
    
    Returns
    -------
    data            : Array of data frames
    err             : Array of uncertainty frames
    hdr             : List of header files
    master_hdr      : List of master header files
    
    History
    -------
    Written by Kevin Stevenson          November 2012
    Updated for NIRSpec by Aarynn Carter/Eva-Maria Ahrer  June 2021
    '''

    assert isinstance(filename, str)
    if stage_1_process:
        # Before extracting the information ourselves, we must process the data to a Stage 2 data product.
        # We do not start with Stage 2 as an excessive number of pixels are trimmed in the default pipeline. 
        stage2_filename = process_to_stage2(filename, do_assignwcs=do_assignwcs, do_extract2d=do_extract2d, do_srctype=do_srctype, do_flatfield=do_flatfield, do_photom=do_photom, delete_files=delete_files,output_dir=output_dir)
    else:
        stage2_filename = filename
    
    # Now we can start working with the data.
    hdulist           = fits.open(stage2_filename)
    data.mhdr         = hdulist[0].header
    data.shdr         = hdulist['SCI',1].header

    data.intstart     = 1
    data.intend     = data.mhdr['NINTS']

    data.data         = hdulist['SCI',1].data
    data.err         = hdulist['ERR',1].data
    data.dq         = hdulist['DQ',1].data
    data.wave         = hdulist['WAVELENGTH',1].data
    data.v0         = hdulist['VAR_RNOISE',1].data
    data.int_times    = hdulist['INT_TIMES',1].data[data.intstart-1:data.intend]

    return data

############## delete this, just to test if it works
homedir = '/storage/astro2/phrgmk/Workshops/ERS_JWST_Hackathon/JWST-Sim/NIRSpec/Stage_1_testing/'
output_dir = '/storage/astro2/phrgmk/Workshops/ERS_JWST_Hackathon/JWST-Sim/NIRSpec/Stage_1_testing/output/'
import eureka.S3_data_reduction.s3_reduce as s3  
#try if this works well and which output it produces 
read(homedir + 'ERS_DATASIMv2_NIRSpec_PRISM_rateints.fits',s3.Data(),output_dir=output_dir)