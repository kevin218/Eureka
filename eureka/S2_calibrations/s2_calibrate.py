#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline


# Proposed Steps
# --------------
# 1.  Read in Stage 1 data products
# 2.  Change default trimming if needed
# 3.  Run the JWST pipeline with any requested modifications
# 4.  Save Stage 2 data products
# 5.  Produce plots


import os, sys, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline
from jwst.pipeline.calwebb_image2 import Image2Pipeline
from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import readECF as rd


class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return

def calibrateJWST(eventlabel):
    '''Reduces rateints spectrum or image files ouput from Stage 1 of the JWST pipeline into calints and x1dints.

    This function does the preparation for running the STScI's JWST pipeline and decides whether to run the
    Spec2Pipeline or Image2Pipeline.

    Parameters
    ----------
    eventlabel: str
        Unique label for this dataset

    Returns
    -------
    meta:   MetaClass
        The metadata object

    Notes
    ------
    History:

    - 03 Nov 2021 Taylor Bell
        Initial version
    '''

    t0 = time.time()

    # Initialize metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Load Eureka! control file and store values in Event object
    ecffile = 'S2_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Create directories for Stage 2 processing outputs
    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir
    run = util.makedirectory(meta, 'S2')
    meta.outputdir = util.pathdirectory(meta, 'S2', run)

    # Output S2 log file
    meta.s2_logname = meta.outputdir + 'S2_' + meta.eventlabel + ".log"
    log = logedit.Logedit(meta.s2_logname)
    log.writelog("\nStarting Stage 2 Reduction")

    # Copy ecf
    log.writelog('Copying S2 control file')
    shutil.copy(ecffile, meta.outputdir)

    # Create list of file segments
    meta = util.readfiles(meta)
    meta.num_data_files = len(meta.segment_list)
    if meta.num_data_files==0:
        rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
        if rootdir[-1]!='/':
            rootdir += '/'
        raise AssertionError(f'Unable to find any "{meta.suffix}.fits" files in the inputdir: \n"{rootdir}"!')
    else:
        log.writelog(f'\nFound {meta.num_data_files} data file(s) ending in {meta.suffix}.fits')

    # If testing, only run the last file
    if meta.testing_S2:
        istart = meta.num_data_files - 1
    else:
        istart = 0

    # Figure out which pipeline we need to use (spectra or images)
    with fits.open(meta.segment_list[0]) as hdulist:
        # Figure out which instrument we are using
        exp_type = hdulist[0].header['EXP_TYPE']
    if 'image' in exp_type.lower():
        # EXP_TYPE header is either MIR_IMAGE, NRC_IMAGE, NRC_TSIMAGE, NIS_IMAGE, or NRS_IMAGING
        pipeline = EurekaImage2Pipeline()
    else:
        # EXP_TYPE doesn't say image, so it should be a spectrum (or someone is putting weird files into Eureka!)
        pipeline = EurekaSpec2Pipeline()

    # Run the pipeline on each file sequentially
    for m in range(istart, meta.num_data_files):
        # Report progress
        log.writelog(f'Starting file {m + 1} of {meta.num_data_files}')
        filename = meta.segment_list[m]

        with fits.open(filename, mode='update') as hdulist:
            if hdulist[0].header['INSTRUME']=='NIRCam':
                # jwst 1.3.3 breaks unless NDITHPTS and NRIMDTPT are integers rather than the strings that they are in the old simulated NIRCam data
                hdulist[0].header['NDITHPTS'] = 1
                hdulist[0].header['NRIMDTPT'] = 1

        pipeline.run_eurekaS2(filename, meta, log)

    # Calculate total run time
    total = (time.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    # Save results
    if not meta.testing_S2:
        log.writelog('Saving Metadata')
        me.saveevent(meta, meta.outputdir + 'S2_' + meta.eventlabel + "_Meta_Save", save=[])

    return meta


class EurekaSpec2Pipeline(Spec2Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_spec2.Spec2Pipeline.

    This wrapper class can allow non-standard changes to Stage 2 steps for Eureka!.

    Notes
    ------
    History:

    - October 2021 Taylor Bell
        Initial version
    '''

    def run_eurekaS2(self, filename, meta, log):
        '''Reduces rateints spectrum files ouput from Stage 1 of the JWST pipeline into calints and x1dints.

        Parameters
        ----------
        filename:   str
            A string pointing to the rateint or rateints file to be operated on.
        meta:   MetaClass
            The metadata object
        log:    logedit.Logedit
            The open log in which notes from this step can be added.

        Returns
        -------
        None

        Notes
        ------
        History:

        - June 2021 Eva-Maria Ahrer and Aarynn Carter
            Code fragments written
        - October 2021 Taylor Bell
            Significantly overhauled code formatting
        - 03 Nov 2021 Taylor Bell
            Fragmented code to allow reuse of code between spectral and image analysis.
        '''
        if meta.slit_y_low != None:
            #Controls the cross-dispersion extraction
            self.assign_wcs.slit_y_low = meta.slit_y_low

        if meta.slit_y_high != None:
            #Controls the cross-dispersion extraction
            self.assign_wcs.slit_y_high = meta.slit_y_high

        if meta.waverange_start != None:
            #Control the dispersion extraction - FIX: Does not actually change dispersion direction extraction
            log.writelog('Editing (in place) the waverange in the input file')
            with datamodels.open(filename) as m:
                m.meta.wcsinfo.waverange_start = meta.waverange_start
                m.save(filename)

        if meta.waverange_end != None:
            #Control the dispersion extraction - FIX: Does not actually change dispersion direction extraction
            if meta.waverange_start == None:
                # Only log this once
                log.writelog('Editing (in place) the waverange in the input file')
            with datamodels.open(filename) as m:
                m.meta.wcsinfo.waverange_end = meta.waverange_end
                m.save(filename)
        
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
        self.resample_spec.skip = meta.skip_resample
        self.cube_build.skip = meta.skip_cube_build
        self.extract_1d.skip = meta.skip_extract_1d
        # Save outputs if requested to the folder specified in the ecf
        self.save_results = (not meta.testing_S2)
        self.output_dir = meta.outputdir
        # This needs to be reset to None to permit the pipeline to be run on multiple files
        self.suffix = None

        # Call the main Spec2Pipeline function (defined in the parent class)
        log.writelog('Running the Spec2Pipeline\n')
        # Must call the pipeline in this way to ensure the skip booleans are respected
        self(filename)

        # Produce some summary plots if requested
        if not meta.testing_S2 and not self.extract_1d.skip:
            log.writelog('\nGenerating x1dints figure')
            m = np.where(meta.segment_list==filename)[0][0]+1
            max_m = meta.num_data_files
            fig_number = '11'+str(m).zfill(np.max([int(np.floor(np.log10(max_m))+1),2]))
            fname = 'fig{}_'.format(fig_number)+'_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints'
            x1d_fname = '_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints'
            with datamodels.open(meta.outputdir+x1d_fname+'.fits') as sp1d:
                plt.figure(int(fig_number), figsize=[15,5])
                plt.clf()
                
                for i in range(len(sp1d.spec)):
                    plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'], sp1d.spec[i].spec_table['FLUX'])

                plt.title('Time Series Observation: Extracted spectra')
                plt.xlabel('Wavelength (micron)')
                plt.ylabel('Flux')
                plt.savefig(meta.outputdir+'figs/'+fname+'.png', bbox_inches='tight', dpi=300)
                if meta.hide_plots:
                    plt.close()
                else:
                    plt.pause(0.2)

        return


class EurekaImage2Pipeline(Image2Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_image2.Image2Pipeline.

    This wrapper class can allow non-standard changes to Stage 2 steps for Eureka!.

    Notes
    ------
    History:

    - October 2021 Taylor Bell
        Initial version
    '''

    def run_eurekaS2(self, filename, meta, log):
        '''Reduces rateints image files ouput from Stage 1 of the JWST pipeline into calints.

        Parameters
        ----------
        filename:   str
            A string pointing to the rateint or rateints file to be operated on.
        meta:   MetaClass
            The metadata object
        log:    logedit.Logedit
            The open log in which notes from this step can be added.

        Returns
        -------
        None

        Notes
        ------
        History:

        - 03 Nov 2021 Taylor Bell
            Initial version
        '''

        # Skip steps according to input ecf file
        self.bkg_subtract.skip = meta.skip_bkg_subtract
        self.flat_field.skip = meta.skip_flat_field
        self.photom.skip = meta.skip_photom
        self.resample.skip = meta.skip_resample
        # Save outputs if requested to the folder specified in the ecf
        self.save_results = (not meta.testing_S2)
        self.output_dir = meta.outputdir
        # This needs to be reset to None to permit the pipeline to be run on multiple files
        self.suffix = None

        # Call the main Image2Pipeline function (defined in the parent class)
        log.writelog('Running the Image2Pipeline\n')
        # Must call the pipeline in this way to ensure the skip booleans are respected
        self(filename)

        return
