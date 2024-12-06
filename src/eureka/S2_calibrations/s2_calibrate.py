#! /usr/bin/env python

# Eureka! Stage 2 calibration pipeline

# Proposed Steps
# --------------
# 1.  Read in Stage 1 data products
# 2.  Change default trimming if needed
# 3.  Run the JWST pipeline with any requested modifications
# 4.  Save Stage 2 data products
# 5.  Produce plots

import os
import time as time_pkg
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from functools import partial
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline
from jwst.pipeline.calwebb_image2 import Image2Pipeline
import jwst.assign_wcs.nirspec

from .s2_meta import S2MetaClass
from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import plots


def calibrateJWST(eventlabel, ecf_path=None, s1_meta=None, input_meta=None):
    '''Reduces rateints spectrum or image files ouput from Stage 1 of the JWST
    pipeline into calints and x1dints.

    This function does the preparation for running the STScI's JWST pipeline
    and decides whether to run the Spec2Pipeline or Image2Pipeline.

    Parameters
    ----------
    eventlabel : str
        Unique label for this dataset.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored. Defaults
        to None which resolves to './'.
    s1_meta : eureka.lib.readECF.MetaClass; optional
        The metadata object from Eureka!'s S1 step (if running S1 and S2
        sequentially). Defaults to None.
    input_meta : eureka.lib.readECF.MetaClass; optional
        An optional input metadata object, so you can manually edit the meta
        object without having to edit the ECF file.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    t0 = time_pkg.time()

    s1_meta = deepcopy(s1_meta)
    input_meta = deepcopy(input_meta)

    if input_meta is None:
        meta = S2MetaClass(folder=ecf_path, eventlabel=eventlabel)
    else:
        meta = S2MetaClass(**input_meta.__dict__)

    if s1_meta is None:
        # Locate the old MetaClass savefile, and load new ECF into
        # that old MetaClass
        s1_meta, meta.inputdir, meta.inputdir_raw = \
            me.findevent(meta, 'S1', allowFail=True)
    else:
        # Running these stages sequentially, so can safely assume
        # that the path hasn't changed
        meta.inputdir = s1_meta.outputdir
        meta.inputdir_raw = meta.inputdir[len(meta.topdir):]

    if s1_meta is None:
        # Attempt to find subdirectory containing S1 FITS files
        meta = util.find_fits(meta)
    else:
        meta = S2MetaClass(**me.mergeevents(meta, s1_meta).__dict__)

    # Create list of file segments
    meta = util.readfiles(meta)

    # First apply any instrument-specific defaults
    if meta.inst == 'miri':
        meta.set_MIRI_defaults()
    elif meta.inst == 'nircam':
        meta.set_NIRCam_defaults()
    elif meta.inst == 'nirspec':
        meta.set_NIRSpec_defaults()
    elif meta.inst == 'niriss':
        meta.set_NIRISS_defaults()
    # Then apply instrument-agnostic defaults
    meta.set_defaults()

    run = util.makedirectory(meta, 'S2')
    meta.outputdir = util.pathdirectory(meta, 'S2', run)

    # Output S2 log file
    meta.s2_logname = meta.outputdir + 'S2_' + meta.eventlabel + ".log"
    if s1_meta is not None:
        log = logedit.Logedit(meta.s2_logname, read=s1_meta.s1_logname)
    else:
        log = logedit.Logedit(meta.s2_logname)
    log.writelog("\nStarting Stage 2 Reduction")
    log.writelog(f"Eureka! Version: {meta.version}", mute=True)
    log.writelog(f"Input directory: {meta.inputdir}")
    log.writelog(f'  Found {meta.num_data_files} data file(s) ending '
                 f'in {meta.suffix}.fits', mute=(not meta.verbose))
    log.writelog(f"Output directory: {meta.outputdir}")

    # Copy ecf
    log.writelog('Copying S2 control file')
    meta.copy_ecf()

    log.writelog(f"CRDS Context pmap: {meta.pmap}", mute=True)

    # If testing, only run the last file
    if meta.testing_S2:
        istart = meta.num_data_files - 1
    else:
        istart = 0

    if meta.photometry:
        # EXP_TYPE header is either MIR_IMAGE, NRC_IMAGE, NRC_TSIMAGE,
        # NIS_IMAGE, or NRS_IMAGING
        pipeline = EurekaImage2Pipeline()
    else:
        # EXP_TYPE doesn't say image, so it should be a spectrum
        # (or someone is putting weird files into Eureka!)
        pipeline = EurekaSpec2Pipeline()

        if (meta.inst == 'nirspec' and
            (meta.waverange_start is not None or
             meta.waverange_end is not None)):
            # By default pipeline can trim the dispersion axis,
            # override the function that does this with specific
            # wavelength range that you want to trim to.
            jwst.assign_wcs.nirspec.nrs_wcs_set_input = \
                partial(jwst.assign_wcs.nirspec.nrs_wcs_set_input,
                        wavelength_range=[meta.waverange_start,
                                          meta.waverange_end])

    # Run the pipeline on each file sequentially
    for m in range(istart, meta.num_data_files):
        # Report progress
        log.writelog(f'Starting file {m + 1} of {meta.num_data_files}')
        filename = meta.segment_list[m]

        with fits.open(filename, mode='update') as hdulist:
            if hdulist[0].header['INSTRUME'] == 'NIRCam':
                # jwst 1.3.3 breaks unless NDITHPTS and NRIMDTPT are integers
                # rather than the strings that they are in the old simulated
                # NIRCam data
                hdulist[0].header['NDITHPTS'] = 1
                hdulist[0].header['NRIMDTPT'] = 1

        pipeline.run_eurekaS2(filename, meta, log)

    # make citations for current stage
    util.make_citations(meta, 2)

    # Save results
    if not meta.testing_S2:
        log.writelog('Saving Metadata')
        me.saveevent(meta, meta.outputdir+'S2_'+meta.eventlabel+"_Meta_Save",
                     save=[])

    # Calculate total run time
    total = (time_pkg.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    log.closelog()

    return meta


class EurekaSpec2Pipeline(Spec2Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_spec2.Spec2Pipeline.

    This wrapper class allows non-standard changes to Stage 2 for Eureka!.

    Notes
    -----
    History:

    - October 2021 Taylor Bell
        Initial version
    '''

    def run_eurekaS2(self, filename, meta, log):
        '''Reduces rateints spectrum files ouput from Stage 1 of the JWST
        pipeline into calints and x1dints.

        Parameters
        ----------
        filename : str
            A string pointing to the rateint or rateints file to process.
        meta : eureka.lib.readECF.MetaClass
            The metadata object.
        log : logedit.Logedit
            The open log in which notes from this step can be added.

        Notes
        -----
        History:

        - June 2021 Eva-Maria Ahrer and Aarynn Carter
            Code fragments written
        - October 2021 Taylor Bell
            Significantly overhauled code formatting
        - 03 Nov 2021 Taylor Bell
            Fragmented code to allow reuse of code between spectral and image
            analysis.
        '''

        if meta.inst == 'nirspec' and meta.slit_y_low is not None:
            #  NIRSpec subarray lower bound in cross-dispersion direction
            self.assign_wcs.slit_y_low = meta.slit_y_low

        if meta.inst == 'nirspec' and meta.slit_y_high is not None:
            #  NIRSpec subarray upper bound in cross-dispersion direction
            self.assign_wcs.slit_y_high = meta.slit_y_high

        if meta.inst == 'nircam' and meta.tsgrism_extract_height is not None:
            # NIRCam grism subarray height in cross-dispersion direction
            self.extract_2d.tsgrism_extract_height = \
                meta.tsgrism_extract_height

        # Skip steps according to input ecf file
        self.msa_flagging.skip = meta.skip_msaflagopen
        if hasattr(self, 'nsclean'):
            # Allowing backwards compatibility with older jwst versions
            self.nsclean.skip = meta.skip_nsclean
        self.imprint_subtract.skip = meta.skip_imprint
        self.bkg_subtract.skip = meta.skip_bkg_subtract
        self.extract_2d.skip = meta.skip_extract_2d
        self.srctype.skip = meta.skip_srctype
        self.master_background_mos.skip = meta.skip_master_background
        self.wavecorr.skip = meta.skip_wavecorr
        self.straylight.skip = meta.skip_straylight
        self.flat_field.skip = meta.skip_flat_field
        self.fringe.skip = meta.skip_fringe
        self.pathloss.skip = meta.skip_pathloss
        self.barshadow.skip = meta.skip_barshadow
        self.wfss_contam.skip = meta.skip_wfss_contam
        self.photom.skip = meta.skip_photom
        self.residual_fringe.skip = meta.skip_residual_fringe
        self.resample_spec.skip = meta.skip_resample
        self.cube_build.skip = meta.skip_cube_build
        self.extract_1d.skip = meta.skip_extract_1d

        # Save outputs if requested to the folder specified in the ecf
        self.save_results = (not meta.testing_S2)
        self.output_dir = meta.outputdir
        # This needs to be reset to None to permit the pipeline to be run on
        # multiple files
        self.suffix = None

        # Call the main Spec2Pipeline function (defined in the parent class)
        log.writelog('Running the Spec2Pipeline\n')
        # Must call the pipeline in this way to ensure the skip booleans are
        # respected
        self(filename)

        # Produce some summary plots if requested
        if not meta.testing_S2 and not self.extract_1d.skip:
            log.writelog('\nGenerating x1dints figure')
            m = np.where(meta.segment_list == filename)[0][0]+1
            max_m = meta.num_data_files
            fig_number = str(m).zfill(int(np.floor(np.log10(max_m))+1))
            fname = f'fig2101_file{fig_number}_x1dints'
            x1d_fname = ('_'.join(filename.split(os.sep)[-1].split('_')[:-1]) +
                         '_x1dints')
            with datamodels.open(meta.outputdir+x1d_fname+'.fits') as sp1d:
                plt.figure(2101, figsize=[15, 5])
                plt.clf()

                for i in range(len(sp1d.spec)):
                    plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'],
                             sp1d.spec[i].spec_table['FLUX'])

                plt.title('Time Series Observation: Extracted spectra')
                plt.xlabel('Wavelength (micron)')
                plt.ylabel('Flux')
                plt.savefig(meta.outputdir+'figs'+os.sep+fname +
                            plots.figure_filetype,
                            bbox_inches='tight', dpi=300)
                if meta.hide_plots:
                    plt.close()
                else:
                    plt.pause(0.2)

        return


class EurekaImage2Pipeline(Image2Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_image2.Image2Pipeline.

    This wrapper class allows non-standard changes to Stage 2 for Eureka!.

    Notes
    -----
    History:

    - October 2021 Taylor Bell
        Initial version
    '''

    def run_eurekaS2(self, filename, meta, log):
        '''Reduces rateints image files ouput from Stage 1 of the JWST
        pipeline into calints.

        Parameters
        ----------
        filename : str
            A string pointing to the rateint or rateints file to process.
        meta : MetaClass
            The metadata object.
        log : logedit.Logedit
            The open log in which notes from this step can be added.

        Notes
        -----
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
        # This needs to be reset to None to permit the pipeline to be run on
        # multiple files
        self.suffix = None

        # Call the main Image2Pipeline function (defined in the parent class)
        log.writelog('Running the Image2Pipeline\n')
        # Must call the pipeline in this way to ensure the skip booleans are
        # respected
        self(filename)

        return
