# pylint: disable=attribute-defined-outside-init
from ..lib.readECF import MetaClass


class S1MetaClass(MetaClass):
    '''A class to hold Eureka! S1 metadata.

    This class loads a Stage 1 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-03 Taylor J Bell
        Made specific S1 class based on MetaClass
    '''

    def __init__(self, folder=None, file=None, eventlabel=None, **kwargs):
        '''Initialize the MetaClass object.

        Parameters
        ----------
        folder : str; optional
            The folder containing an ECF file to be read in. Defaults to None
            which resolves to './'.
        file : str; optional
            The ECF filename to be read in. Defaults to None which first tries
            to find the filename using eventlabel and stage, and if that fails
            results in an empty MetaClass object.
        eventlabel : str; optional
            The unique identifier for these data.
        **kwargs : dict
            Any additional parameters to be loaded into the MetaClass after
            the ECF has been read in

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version.
        '''
        super().__init__(folder, file, eventlabel, stage=1, **kwargs)

    def set_defaults(self):
        '''Set Stage 1 specific defaults for generic instruments.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for any instrument.
        '''
        # Data file suffix
        self.suffix = getattr(self, 'suffix', 'uncal')

        # Control parallelization
        # (Options are 'none', quarter', 'half', 'all', or any integer)
        self.maximum_cores = getattr(self, 'maximum_cores', 'half')

        # Control ramp fitting method
        self.ramp_fit_algorithm = getattr(self, 'ramp_fit_algorithm',
                                          'default')

        # Pipeline steps
        self.skip_group_scale = getattr(self, 'skip_group_scale', False)
        self.skip_dq_init = getattr(self, 'skip_dq_init', False)
        self.skip_saturation = getattr(self, 'skip_saturation', False)
        self.skip_ipc = getattr(self, 'skip_ipc', True)
        self.skip_refpix = getattr(self, 'skip_refpix', False)
        self.skip_linearity = getattr(self, 'skip_linearity', False)
        self.skip_dark_current = getattr(self, 'skip_dark_current', False)
        self.skip_jump = getattr(self, 'skip_jump', False)
        self.skip_ramp_fitting = getattr(self, 'skip_ramp_fitting', False)
        self.skip_gain_scale = getattr(self, 'skip_gain_scale', False)

        # CR sigma rejection threshold
        self.jump_rejection_threshold = getattr(self,
                                                'jump_rejection_threshold', 4.)
        try:
            self.jump_rejection_threshold = \
                float(self.jump_rejection_threshold)
        except ValueError:
            print("\nmeta.jump_rejection_threshold cannot be type-casted to "
                  "a float. Defaulting to 4.0")
            self.jump_rejection_threshold = 4.0

        # CR algorithm threshold
        self.minimum_sigclip_groups = getattr(self, 'minimum_sigclip_groups',
                                              100)

        # Custom linearity reference file
        self.custom_linearity = getattr(self, 'custom_linearity', False)
        if self.custom_linearity:
            # Force this to be specified if custom_linearity is True
            self.linearity_file = getattr(self, 'linearity_file')

        self.custom_mask = getattr(self, 'custom_mask', False)
        if self.custom_mask:
            # Force this to be specified if custom_mask is True
            self.mask_file = getattr(self, 'mask_file')

        # Custom bias when using NIRSpec G395H
        # Options: [mean, group_level, smooth, None].
        # If not None, requires masktrace=True
        self.bias_correction = getattr(self, 'bias_correction', None)
        if self.bias_correction is not None:
            # Group number options: [1, 2, ..., each]
            self.bias_group = getattr(self, 'bias_group', 1)
            if self.bias_correction == 'smooth':
                # Force this to be specified if bias_correction is 'smooth'
                self.bias_smooth_length = getattr(self, 'bias_smooth_length')
        self.custom_bias = getattr(self, 'custom_bias', False)
        if self.custom_bias:
            # Force this to be specified if custom_bias is True
            self.superbias_file = getattr(self, 'superbias_file')

        # Manually mask groups
        self.mask_groups = getattr(self, 'mask_groups', False)

        # Saturation
        # Wheter to update the saturation flags more aggressively?
        self.update_sat_flags = getattr(self, 'update_sat_flags', False)
        if self.update_sat_flags:
            # Expand saturation flags to previous group
            self.expand_prev_group = getattr(self, 'expand_prev_group', False)
            # Force this to be specified if dq_sat_mode is 'defined'.
            # Options: [percentile, min, defined]
            self.dq_sat_mode = getattr(self, 'dq_sat_mode')
            if self.dq_sat_mode == 'percentile':
                # Force this to be specified if dq_sat_mode is 'defined'.
                # The percentile of the entire time series to use to define the
                # saturation mask (50=median)
                self.dq_sat_percentile = getattr(self, 'dq_sat_percentile')
            elif self.dq_sat_mode == 'defined':
                # Force this to be specified if dq_sat_mode is 'defined'
                self.dq_sat_columns = getattr(self, 'dq_sat_columns')

        # Mask curved traces
        self.masktrace = getattr(self, 'masktrace', False)
        if self.masktrace:
            # Override bg_y1 and bg_y2 if masking the trace
            print('  Overriding meta.bg_y1 and meta.bg_y2 since you set '
                  'meta.masktrace to True, and the bg_y1 and bg_y2 parameters '
                  'are not needed when masking the trace.')
            self.bg_y1 = 17
            self.bg_y2 = 16
            # Force these to be specified if masking the trace
            self.window_len = getattr(self, 'window_len')
            self.expand_mask = getattr(self, 'expand_mask')
            self.ignore_low = getattr(self, 'ignore_low')
            self.ignore_hi = getattr(self, 'ignore_hi')
        elif self.bias_correction:
            raise AssertionError(
                f'meta.bias_correction has been set to {self.bias_correction} '
                'which requires meta.masktrace=True. Meanwhile, meta.masktrace'
                ' has been set to False. Please update bias_correction or '
                'masktrace in your Stage 1 ECF.')

        # Manual reference pixel correction for NIRSpec PRISM when not
        # subtracting BG
        self.refpix_corr = getattr(self, 'refpix_corr', False)
        if self.refpix_corr:
            if self.grouplevel_bg:
                print('WARNING: Performing GLBS and reference pixel correction'
                      ' is redundant and not recommended.')
            # Force these to be specified if refpix_corr is True
            self.npix_top = getattr(self, 'npix_top')
            self.npix_bot = getattr(self, 'npix_bot')

        # Background subtraction
        self.grouplevel_bg = getattr(self, 'grouplevel_bg', False)
        if self.grouplevel_bg:
            self.ncpu = getattr(self, 'ncpu', 4)
            # Force this to be specified if grouplevel_bg is True
            self.bg_y1 = getattr(self, 'bg_y1')
            # Force this to be specified if grouplevel_bg is True
            self.bg_y2 = getattr(self, 'bg_y2')
            self.bg_deg = getattr(self, 'bg_deg', 0)
            # Options: std (Standard Deviation),
            # median (Median Absolute Deviation), or
            # mean (Mean Absolute Deviation)
            self.bg_method = getattr(self, 'bg_method', 'median')
            self.p3thresh = getattr(self, 'p3thresh', 3)
            # Row-by-row BG subtraction (only useful for NIRCam)
            self.bg_row_by_row = getattr(self, 'bg_row_by_row', False)
            self.orders = getattr(self, 'orders', None)
            self.src_ypos = getattr(self, 'src_ypos', 15)
        # bg_x1 and bg_x2 also need to be defined if meta.masktrace is True
        # Left edge of exclusion region for row-by-row BG subtraction
        self.bg_x1 = getattr(self, 'bg_x1', None)
        # Right edge of exclusion region for row-by-row BG subtraction
        self.bg_x2 = getattr(self, 'bg_x2', None)

        # Diagnostics
        self.isplots_S1 = getattr(self, 'isplots_S1', 1)
        self.nplots = getattr(self, 'nplots', 5)
        self.hide_plots = getattr(self, 'hide_plots', True)
        self.testing_S1 = getattr(self, 'testing_S1', False)
        self.verbose = getattr(self, 'verbose', True)

        # Project directory
        # Must be provided in the ECF
        self.topdir = getattr(self, 'topdir')

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage0')
        self.outputdir = getattr(self, 'outputdir', 'Stage1')

        #####

        # "Default" ramp fitting settings
        # Options are "default", "fixed", "interpolated", "flat", or "custom"
        self.default_ramp_fit_weighting = 'default'
        if self.default_ramp_fit_weighting == 'fixed':
            # Force this to be specified if fixed weighting
            self.default_ramp_fit_fixed_exponent = getattr(
                self, 'default_ramp_fit_fixed_exponent', 10)
        elif self.default_ramp_fit_weighting == 'custom':
            # Force these to be specified if custom weighting
            self.default_ramp_fit_custom_snr_bounds = getattr(
                self, 'default_ramp_fit_custom_snr_bounds',
                [5, 10, 20, 50, 100])
            self.default_ramp_fit_custom_exponents = getattr(
                self, 'default_ramp_fit_custom_exponents',
                [0.4, 1, 3, 6, 10])

    def set_MIRI_defaults(self):
        '''Set Stage 1 specific defaults for MIRI.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for MIRI.
        '''
        # MIRI-specific pipeline stages

        # jwst skips by default for MIRI TSO, but should likely be set False
        # (that may depend on the dataset though).
        self.skip_firstframe = getattr(self, 'skip_firstframe', True)
        # jwst skips by default for MIRI TSO, but should likely be set False
        # (that may depend on the dataset though).
        self.skip_lastframe = getattr(self, 'skip_lastframe', True)
        self.skip_reset = getattr(self, 'skip_reset', False)
        # jwst skips by default for MIRI TSO.
        self.skip_rscd = getattr(self, 'skip_rscd', True)
        self.skip_emicorr = getattr(self, 'skip_emicorr', True)

        # Remove the 390 Hz periodic noise in MIRI/LRS SLITLESSPRISM
        # group-level data?
        self.remove_390hz = getattr(self, 'remove_390hz', False)
        if self.remove_390hz:
            # Default to True if remove_390hz
            self.grouplevel_bg = getattr(self, 'grouplevel_bg', True)
        else:
            # Default to False if not remove_390hz
            self.grouplevel_bg = getattr(self, 'grouplevel_bg', False)

    def set_NIR_defaults(self):
        '''Set Stage 1 specific defaults for NIR-instruments.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for NIR-instruments.
        '''
        # NIR-specific pipeline stages
        self.skip_superbias = getattr(self, 'skip_superbias', False)
        # Skipped by default for Near-IR TSO.
        self.skip_persistence = getattr(self, 'skip_persistence', True)

        # Placeholder for Eureka_RampFitStep
        self.remove_390hz = getattr(self, 'remove_390hz', False)
        if self.remove_390hz:
            raise AssertionError('remove_390hz cannot be set to True for NIR '
                                 'instruments!')
