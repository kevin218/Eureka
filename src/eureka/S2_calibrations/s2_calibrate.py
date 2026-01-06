import os
import inspect
import time as time_pkg
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from functools import partial
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline
from jwst.pipeline.calwebb_image2 import Image2Pipeline
from jwst.assign_wcs import nirspec as nrs

from .s2_meta import S2MetaClass
from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import plots

_orig_nrs_wcs_set_input = nrs.nrs_wcs_set_input


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

    meta.run_s2 = util.makedirectory(meta, 'S2')
    meta.outputdir = util.pathdirectory(meta, 'S2', meta.run_s2)

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

        # By default pipeline can trim the dispersion axis,
        # override the function that does this with specific
        # wavelength range that you want to trim to.
        patched_nrs = False
        if (meta.inst == 'nirspec' and meta.waverange_start is not None
                and meta.waverange_end is not None):
            nrs.nrs_wcs_set_input = partial(
                _nrs_set_input_override,
                wavelength_range=(meta.waverange_start, meta.waverange_end),
                slit_y_low=meta.slit_y_low, slit_y_high=meta.slit_y_high)
            patched_nrs = True

    try:
        # Run the pipeline on each file sequentially
        for m in range(istart, meta.num_data_files):
            # Report progress
            meta.m = m
            filename = meta.segment_list[m]
            log.writelog(f'Starting file {m + 1} of {meta.num_data_files}')

            need_update = False
            with fits.open(filename) as hdulist:
                if (hdulist[0].header['INSTRUME'] == 'NIRCam'
                        and isinstance(hdulist[0].header['NDITHPTS'], str)):
                    need_update = True

                meta.intstart = hdulist[0].header['INTSTART']-1
                meta.intend = hdulist[0].header['INTEND']
                meta.n_int = meta.intend-meta.intstart

            if need_update:
                with fits.open(filename, mode='update') as hdulist:
                    # If the NDITHPTS header is a string, then it is an old
                    # simulated file and we need to change it to an integer
                    hdulist[0].header['NDITHPTS'] = int(
                        hdulist[0].header['NDITHPTS'])
                    hdulist[0].header['NRIMDTPT'] = int(
                        hdulist[0].header['NRIMDTPT'])

            pipeline.run_eurekaS2(filename, meta, log)
    finally:
        # Restore original nrs_wcs_set_input after processing all files
        if 'patched_nrs' in locals() and patched_nrs:
            nrs.nrs_wcs_set_input = _orig_nrs_wcs_set_input

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


def _nrs_set_input_override(input_model, slit_name, wavelength_range=None,
                            slit_y_low=None, slit_y_high=None):
    """Return a WCS object for a specific slit, slice or shutter.

    Overrides the original jwst.assign_wcs.nirspec.nrs_wcs_set_input to
    accept wavelength range and slit y-limits to avoid the newly forced
    cropping behavior in jwst>=1.20.1.

    Parameters
    ----------
    input_model : JwstDataModel
        A datamodel that contains a WCS object for the all open slitlets in
        an observation.
    slit_name : int or str
        Slit.name of an open slit.
    wavelength_range : tuple of float
        Wavelength range for the combination of filter and grating.
    slit_y_low, slit_y_high : float
        The lower and upper bounds of the slit. Optional.

    Returns
    -------
    wcsobj : `~gwcs.wcs.WCS`
        WCS object for this slit.
    """
    slit_wcs = _orig_nrs_wcs_set_input(input_model, slit_name)
    # Build bbox over requested wavelength range, using cross-disp limits
    # if provided; otherwise from slit geometry.
    transform = slit_wcs.get_transform("detector", "slit_frame")
    # Try to get slit geometry from WCS (fixed-slit TSO path)
    slit = None
    if "gwa" in input_model.meta.wcs.available_frames:
        g2s = input_model.meta.wcs.get_transform("gwa", "slit_frame")
        slits = getattr(g2s, "slits", [])
        slit = next((s for s in slits if s.name == slit_name), None)
    if slit_y_low is not None:
        ylo = slit_y_low
    elif slit is not None:
        ylo = slit.ymin
    else:
        ylo = -0.55
    if slit_y_high is not None:
        yhi = slit_y_high
    elif slit is not None:
        yhi = slit.ymax
    else:
        yhi = 0.55

    sig = inspect.signature(nrs.compute_bounding_box)
    if 'slit_name' in sig.parameters:
        bb = nrs.compute_bounding_box(
            transform, slit_name=None, wavelength_range=wavelength_range,
            slit_ymin=ylo, slit_ymax=yhi)
    else:
        # Minimal fallback for older signature in 1.18.0 without slit_name arg
        bb = nrs.compute_bounding_box(
            transform, wavelength_range=wavelength_range,
            slit_ymin=ylo, slit_ymax=yhi)
    slit_wcs.bounding_box = bb
    return slit_wcs


class EurekaSpec2Pipeline(Spec2Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_spec2.Spec2Pipeline.

    This wrapper class allows non-standard changes to Stage 2 for Eureka!.
    '''

    @plots.apply_style
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
        self.run(filename)

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
                fig = plt.figure(2101)
                fig.set_size_inches(15, 5, forward=True)
                fig.clf()

                for i in range(len(sp1d.spec)):
                    plt.plot(sp1d.spec[i].spec_table['WAVELENGTH'],
                             sp1d.spec[i].spec_table['FLUX'])

                plt.title('Time Series Observation: Extracted spectra')
                plt.xlabel('Wavelength (micron)')
                plt.ylabel('Flux')
                plt.savefig(meta.outputdir+'figs'+os.sep+fname +
                            plots.get_filetype(),
                            bbox_inches='tight', dpi=300)
                if meta.hide_plots:
                    plt.close()
                else:
                    plt.pause(0.2)

        return


class EurekaImage2Pipeline(Image2Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_image2.Image2Pipeline.

    This wrapper class allows non-standard changes to Stage 2 for Eureka!.
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
        self.run(filename)

        return
