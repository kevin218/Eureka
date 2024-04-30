import os
import time as time_pkg
import numpy as np
from copy import deepcopy
from astropy.io import fits

from jwst.pipeline.calwebb_detector1 import Detector1Pipeline
import crds

from .ramp_fitting import Eureka_RampFitStep
from .superbias import Eureka_SuperBiasStep

from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import readECF
from ..version import version


def rampfitJWST(eventlabel, ecf_path=None, input_meta=None):
    """Process a Stage 0, _uncal.fits file to Stage 1 _rate.fits and
    _rateints.fits files.

    Steps taken to perform this processing can follow the default JWST
    pipeline, or alternative methods.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored. Defaults to
        None which resolves to './'.
    input_meta : eureka.lib.readECF.MetaClass; optional
        An optional input metadata object, so you can manually edit the meta
        object without having to edit the ECF file.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - October 2021 Taylor Bell
        Code fragments
    - October 2021 Aarynn Carter and Eva-Maria Ahrer
        Initial version
    - February 2022 Aarynn Carter and Eva-Maria Ahrer
        Updated for JWST version 1.3.3, code restructure
    """
    t0 = time_pkg.time()

    input_meta = deepcopy(input_meta)

    if input_meta is None:
        # Load Eureka! control file and store values in Event object
        ecffile = 'S1_' + eventlabel + '.ecf'
        meta = readECF.MetaClass(ecf_path, ecffile)
    else:
        meta = input_meta

    # If a specific CRDS context is entered in the ECF, apply it.
    # Otherwise, log and fix the default CRDS context to make sure it doesn't
    # change between different segments.
    if not hasattr(meta, 'pmap') or meta.pmap is None:
        # Get just the numerical value
        meta.pmap = crds.get_context_name('jwst')[5:-5]
    os.environ['CRDS_CONTEXT'] = f'jwst_{meta.pmap}.pmap'

    meta.version = version
    meta.eventlabel = eventlabel
    meta.datetime = time_pkg.strftime('%Y-%m-%d')

    # Create directories for Stage 1 processing outputs
    run = util.makedirectory(meta, 'S1')
    meta.outputdir = util.pathdirectory(meta, 'S1', run)
    # Make a separate folder for plot outputs
    if not os.path.exists(meta.outputdir+'figs'):
        os.makedirs(meta.outputdir+'figs')

    # Output S2 log file
    meta.s1_logname = meta.outputdir + 'S1_' + meta.eventlabel + ".log"
    log = logedit.Logedit(meta.s1_logname)
    log.writelog("\nStarting Stage 1 Processing")
    log.writelog(f"Eureka! Version: {meta.version}", mute=True)
    log.writelog(f"CRDS Context pmap: {meta.pmap}", mute=True)
    log.writelog(f"Input directory: {meta.inputdir}")
    log.writelog(f"Output directory: {meta.outputdir}")

    # Copy ecf
    log.writelog('Copying S1 control file')
    meta.copy_ecf()

    # Create list of file segments
    meta = util.readfiles(meta, log)

    # If testing, only run the last file
    if meta.testing_S1:
        istart = meta.num_data_files - 1
    else:
        istart = 0

    for m in range(istart, meta.num_data_files):
        # Report progress
        filename = meta.segment_list[m]
        log.writelog(f'Starting file {m + 1} of {meta.num_data_files}: ' +
                     filename.split(os.sep)[-1])

        with fits.open(filename, mode='update') as hdulist:
            # record instrument information in meta object for citations
            if not hasattr(meta, 'inst'):
                meta.inst = hdulist[0].header["INSTRUME"].lower()

            # jwst 1.3.3 breaks unless NDITHPTS/NRIMDTPT are integers rather
            # than the strings that they are in the old simulated NIRCam data
            if hdulist[0].header['INSTRUME'] == 'NIRCAM':
                hdulist[0].header['NDITHPTS'] = 1
                hdulist[0].header['NRIMDTPT'] = 1

            meta.m = m
            meta.intstart = hdulist[0].header['INTSTART']-1
            meta.intend = hdulist[0].header['INTEND']
            meta.n_int = meta.intend-meta.intstart
            EurekaS1Pipeline().run_eurekaS1(filename, meta, log)

    # Calculate total run time
    total = (time_pkg.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    # make citations for current stage
    util.make_citations(meta, 1)

    # Save results
    if not meta.testing_S1:
        log.writelog('Saving Metadata')
        me.saveevent(meta, meta.outputdir+'S1_'+meta.eventlabel+"_Meta_Save",
                     save=[])

    log.closelog()

    return meta


class EurekaS1Pipeline(Detector1Pipeline):
    '''A wrapper class for jwst.pipeline.calwebb_detector1.Detector1Pipeline

    This wrapper class allows non-standard changes to Stage 1 for Eureka!.

    Notes
    -----
    History:

    - October 2021 Aarynn Carter /  Eva-Maria Ahrer
        Initial version
    - February 2022 Aarynn Carter /  Eva-Maria Ahrer
        Updated for JWST version 1.3.3, code restructure
    '''

    def run_eurekaS1(self, filename, meta, log):
        '''Reduces uncal files from STScI into rateints files.

        Parameters
        ----------
        filename : str
            A string pointing to the uncal file to be operated on.
        meta : eureka.lib.readECF.MetaClass
            The metadata object.
        log : logedit.Logedit
            The open log in which notes from this step can be added.

        Notes
        -----
        History:

        - October 2021 Aarynn Carter /  Eva-Maria Ahrer
            Initial version
        - February 2022 Aarynn Carter /  Eva-Maria Ahrer
            Updated for JWST version 1.3.3, code restructure
        '''
        # Define superbias offset procedure
        self.superbias = Eureka_SuperBiasStep()
        self.superbias.s1_meta = meta
        self.superbias.s1_log = log

        # Figure out which instrument we're working on
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
        if hasattr(meta, 'custom_linearity') and meta.custom_linearity:
            self.linearity.override_linearity = meta.linearity_file
        self.dark_current.skip = meta.skip_dark_current
        self.jump.skip = meta.skip_jump
        self.jump.maximum_cores = meta.maximum_cores
        if (hasattr(meta, 'jump_rejection_threshold') and
                isinstance(meta.jump_rejection_threshold, float)):
            self.jump.rejection_threshold = meta.jump_rejection_threshold
        else:
            log.writelog("\nJump rejection threshold is not" +
                         "defined or not a float, default is used (4.0)")
        self.gain_scale.skip = meta.skip_gain_scale

        # Instrument Specific Steps
        if instrument in ['NIRCAM', 'NIRISS', 'NIRSPEC']:
            self.persistence.skip = meta.skip_persistence
            self.superbias.skip = meta.skip_superbias
            if hasattr(meta, 'custom_bias') and meta.custom_bias:
                self.superbias.override_superbias = meta.superbias_file
        elif instrument in ['MIRI']:
            if (hasattr(meta, 'remove_390hz')
                    and meta.remove_390hz):
                # Need to apply these steps later to be able to remove 390 Hz
                self.firstframe.skip = True
                self.lastframe.skip = True
            else:
                self.firstframe.skip = meta.skip_firstframe
                self.lastframe.skip = meta.skip_lastframe
            if hasattr(meta, 'skip_reset'):
                self.reset.skip = meta.skip_reset
            if hasattr(meta, 'skip_rscd'):
                self.rscd.skip = meta.skip_rscd
            if hasattr(meta, 'skip_emicorr'):
                self.emicorr.skip = meta.skip_emicorr

        # Define ramp fitting procedure
        self.ramp_fit = Eureka_RampFitStep()
        self.ramp_fit.algorithm = meta.ramp_fit_algorithm
        self.ramp_fit.maximum_cores = meta.maximum_cores
        self.ramp_fit.skip = meta.skip_ramp_fitting
        self.ramp_fit.s1_meta = meta
        self.ramp_fit.s1_log = log

        # Default ramp fitting settings
        if self.ramp_fit.algorithm == 'default':
            self.ramp_fit.weighting = meta.default_ramp_fit_weighting
            # Some weighting methods need additional parameters
            if self.ramp_fit.weighting == 'fixed':
                self.ramp_fit.fixed_exponent = \
                    meta.default_ramp_fit_fixed_exponent
            elif self.ramp_fit.weighting == 'custom':
                self.ramp_fit.custom_snr_bounds = \
                    meta.default_ramp_fit_custom_snr_bounds
                self.ramp_fit.custom_exponents = \
                    meta.default_ramp_fit_custom_exponents

        # Run Stage 1
        self(filename)

        return
