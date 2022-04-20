#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline


# Proposed Steps
# --------------
# 1.  Read in Stage 1 data products
# 2.  Change default trimming if needed
# 3.  Run the JWST pipeline with any requested modifications
# 4.  Save Stage 2 data products
# 5.  Produce plots


import os, shutil, time, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline
from jwst.pipeline.calwebb_image2 import Image2Pipeline
from ..lib import logedit, util
from ..lib import sort_nicely as sn
from ..lib import manageevent as me
from ..lib import readECF
from ..lib.plots import figure_filetype

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return

def calibrateJWST(eventlabel, ecf_path='./', s1_meta=None):
    '''Reduces rateints spectrum or image files ouput from Stage 1 of the JWST pipeline into calints and x1dints.

    This function does the preparation for running the STScI's JWST pipeline and decides whether to run the
    Spec2Pipeline or Image2Pipeline.

    Parameters
    ----------
    eventlabel: str
        Unique label for this dataset
    ecf_path:   str
        The absolute or relative path to where ecfs are stored
    s1_meta:    MetaClass
        The metadata object from Eureka!'s S1 step (if running S1 and S2 sequentially).

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

    # Load Eureka! control file and store values in Event object
    ecffile = 'S2_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
    meta.eventlabel = eventlabel

    if s1_meta == None:
        #load savefile
        s1_meta = read_s1_meta(meta)

    if s1_meta != None:
        meta = load_general_s1_meta_info(meta, ecf_path, s1_meta)
    else:
        # Attempt to find subdirectory containing S1 FITS files
        meta = find_s1_files(meta)

        # Create directories for Stage 2 processing outputs
        meta.inputdir_raw = meta.inputdir
        meta.outputdir_raw = meta.outputdir
        meta.inputdir = os.path.join(meta.topdir, *meta.inputdir_raw.split(os.sep))
        meta.outputdir = os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))
    
    run = util.makedirectory(meta, 'S2')
    meta.outputdir = util.pathdirectory(meta, 'S2', run)

    # Output S2 log file
    meta.s2_logname = meta.outputdir + 'S2_' + meta.eventlabel + ".log"
    if s1_meta != None:
        log = logedit.Logedit(meta.s2_logname, read=s1_meta.s1_logname)
    else:
        log = logedit.Logedit(meta.s2_logname)
    log.writelog("\nStarting Stage 2 Reduction")
    log.writelog(f"Input directory: {meta.inputdir}")
    log.writelog(f"Output directory: {meta.outputdir}")

    # Copy ecf
    log.writelog('Copying S2 control file')
    meta.copy_ecf()

    # Create list of file segments
    meta = util.readfiles(meta)
    meta.num_data_files = len(meta.segment_list)
    if meta.num_data_files==0:
        raise AssertionError(f'Unable to find any "{meta.suffix}.fits" files in the inputdir: \n"{meta.inputdir}"!\n'+
                             f'You likely need to change the inputdir in {meta.filename} to point to the folder containing the "{meta.suffix}.fits" files.')
    else:
        log.writelog(f'\nFound {meta.num_data_files} data file(s) ending in {meta.suffix}.fits')

    # If testing, only run the last file
    if meta.testing_S2:
        istart = meta.num_data_files - 1
    else:
        istart = 0

    # Figure out which pipeline we need to use (spectra or images)
    with fits.open(meta.segment_list[0]) as hdulist:
        # Figure out which observatory and observation mode we are using
        telescope = hdulist[0].header['TELESCOP']
    if telescope=='JWST':
        exp_type = hdulist[0].header['EXP_TYPE']
        if 'image' in exp_type.lower():
            # EXP_TYPE header is either MIR_IMAGE, NRC_IMAGE, NRC_TSIMAGE, NIS_IMAGE, or NRS_IMAGING
            pipeline = EurekaImage2Pipeline()
        else:
            # EXP_TYPE doesn't say image, so it should be a spectrum (or someone is putting weird files into Eureka!)
            pipeline = EurekaSpec2Pipeline()
    elif telescope=='HST':
        log.writelog('There is no Stage 2 for HST - skipping.')
        shutil.rmtree(os.path.join(meta.topdir, *meta.outputdir_raw.split(os.sep))) # Clean up temporary folder
        meta.outputdir = meta.inputdir
        return meta
    else:
        raise AssertionError(f'Telescope "{telescope}" detected in FITS header is not JWST or HST and is unsupported!')

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

    log.closelog()
    
    return meta


def read_s1_meta(meta):
    '''Loads in an S1 meta file.

    Parameters
    ----------
    meta:    MetaClass
        The new meta object for the current S2 processing.

    Returns
    -------
    s1_meta:   MetaClass
        The S1 metadata object.

    Notes
    -------
    History:

    - April 15, 2022 Taylor Bell
        Initial version.
    '''
    # Search for the S1 output metadata in the inputdir provided in
    # First just check the specific inputdir folder
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    fnames = glob.glob(rootdir+'S1_'+meta.eventlabel+'*_Meta_Save.dat')
    if len(fnames)==0:
        # There were no metadata files in that folder, so let's see if there are in children folders
        fnames = glob.glob(rootdir+'**/S1_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)

    if len(fnames)>=1:
        # get the folder with the latest modified time
        fname = max(fnames, key=os.path.getmtime)
    
    if len(fnames)==0:
        # There may be no rateints files in the inputdir or any of its children directories - raise an error and give a helpful message
        print('WARNING: Unable to find an output metadata file from Eureka!\'s S1 step '
             +'in the inputdir: \n"{}"!\n'.format(meta.inputdir)
             +'Assuming this S1 data was produced by the JWST pipeline instead.')
        return None
    elif len(fnames)>1:
        # There may be multiple runs - use the most recent but warn the user
        print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                +'Using the metadata file: \n{}\n'.format(fname)
                +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                +'of the run you intended, please provide a more precise folder for the metadata file.')

    fname = fname[:-4] # Strip off the .dat ending

    s1_meta = me.loadevent(fname)

    # Code to not break backwards compatibility with old MetaClass save files but also use the new MetaClass going forwards
    s1_meta = readECF.MetaClass(**s1_meta.__dict__)

    return s1_meta

def find_s1_files(meta):
    '''Locates S1 output FITS files if not running S1 and S2 sequentially.

    Parameters
    ----------
    meta:    MetaClass
        The new meta object for the current S2 processing.

    Returns
    -------
    meta:   MetaClass
        The meta object with the updated inputdir pointing to the location of
        the input files to use.

    Notes
    -------
    History:

    - April 15, 2022 Taylor Bell
        Initial version.
    '''
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    fnames = glob.glob(rootdir+'*'+meta.suffix + '.fits')
    if len(fnames)==0:
        # There were no rateints files in that folder, so let's see if there are in children folders
        fnames = glob.glob(rootdir+'**/*'+meta.suffix + '.fits', recursive=True)
        fnames = sn.sort_nicely(fnames)

    if len(fnames)==0:
        # If the code can't find any of the reqested files, raise an error and give a helpful message
        raise AssertionError(f'Unable to find any "{meta.suffix}.fits" files in the inputdir: \n"{meta.inputdir}"!\n'+
                             f'You likely need to change the inputdir in {meta.filename} to point to the folder containing the "{meta.suffix}.fits" files.')
    
    folders = np.unique([os.sep.join(fname.split(os.sep)[:-1]) for fname in fnames])
    if len(folders)>=1:
        # get the file with the latest modified time
        folder = max(folders, key=os.path.getmtime)
    
    if len(folders)>1:
        # There may be multiple runs - use the most recent but warn the user
        print(f'WARNING: There are multiple folders containing "{meta.suffix}.fits" files in your inputdir: \n"{meta.inputdir}"\n'
             +f'Using the files in: \n{folder}\n'
              +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
              +'of the run you intended, please provide a more precise folder for the metadata file.')

    meta.inputdir = folder[len(meta.topdir):]
    if meta.inputdir[-1] != '/':
        meta.inputdir += '/'

    return meta

def load_general_s1_meta_info(meta, ecf_path, s1_meta):
    '''Loads in the S1 meta save file and adds in attributes from the S2 ECF.

    Parameters
    ----------
    meta:    MetaClass
        The new meta object for the current S2 processing.
    ecf_path:
        The absolute path to where the S2 ECF is stored.

    Returns
    -------
    meta:   MetaClass
        The S1 metadata object with attributes added by S2.

    Notes
    -------
    History:

    - April 15, 2022 Taylor Bell
        Initial version.
    '''
    # Need to remove the topdir from the outputdir
    s1_outputdir = s1_meta.outputdir[len(meta.topdir):]
    if s1_outputdir[0]=='/':
        s1_outputdir = s1_outputdir[1:]
    if s1_outputdir[-1]!='/':
        s1_outputdir += '/'
    s1_topdir = s1_meta.topdir
    
    # Load S2 Eureka! control file and store values in the S1 metadata object
    ecffile = 'S2_' + meta.eventlabel + '.ecf'
    meta = s1_meta
    meta.read(ecf_path, ecffile)

    # Overwrite the inputdir with the exact output directory from S1
    meta.inputdir = os.path.join(s1_topdir, s1_outputdir)
    meta.old_datetime = meta.datetime # Capture the date that the S1 data was made (to figure out it's foldername)
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = s1_outputdir
    meta.outputdir_raw = meta.outputdir

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
            fig_number = str(m).zfill(int(np.floor(np.log10(max_m))+1))
            fname = f'fig2101_file{fig_number}_x1dints'
            x1d_fname = '_'.join(filename.split('/')[-1].split('_')[:-1])+'_x1dints'
            with datamodels.open(meta.outputdir+x1d_fname+'.fits') as sp1d:
                plt.figure(2101, figsize=[15,5])
                plt.clf()
                
                for i in range(len(sp1d.spec)):
                    plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'], sp1d.spec[i].spec_table['FLUX'])

                plt.title('Time Series Observation: Extracted spectra')
                plt.xlabel('Wavelength (micron)')
                plt.ylabel('Flux')
                plt.savefig(meta.outputdir+'figs/'+fname+figure_filetype, bbox_inches='tight', dpi=300)
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
