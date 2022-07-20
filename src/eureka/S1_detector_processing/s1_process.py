import os
import time as time_pkg
import numpy as np
from astropy.io import fits

from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

from eureka.S1_detector_processing.ramp_fitting import Eureka_RampFitStep

from ..lib import logedit, util
from ..lib import manageevent as me
from ..lib import readECF


def rampfitJWST(eventlabel, ecf_path=None):
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

    # Load Eureka! control file and store values in Event object
    ecffile = 'S1_' + eventlabel + '.ecf'
    meta = readECF.MetaClass(ecf_path, ecffile)
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
            # jwst 1.3.3 breaks unless NDITHPTS/NRIMDTPT are integers rather
            # than the strings that they are in the old simulated NIRCam data
            if hdulist[0].header['INSTRUME'] == 'NIRCAM':
                hdulist[0].header['NDITHPTS'] = 1
                hdulist[0].header['NRIMDTPT'] = 1

            EurekaS1Pipeline().run_eurekaS1(filename, meta, log)

    # Calculate total run time
    total = (time_pkg.time() - t0) / 60.
    log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))

    # Save results
    if not meta.testing_S1:
        log.writelog('Saving Metadata')
        me.saveevent(meta, meta.outputdir+'S1_'+meta.eventlabel+"_Meta_Save",
                     save=[])

    return meta


class EurekaS1Pipeline(Detector1Pipeline):
    '''A wrapper class for the jwst.pipeline.calwebb_detector1.Detector1Pipeline

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
        # Run the pipeline
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
