from ..lib.readECF import MetaClass

class S1MetaClass(MetaClass):
    '''A class to hold Eureka! S1 metadata.

    This class loads a Stage 1 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-03-05 Taylor J Bell
        Made specific S1 class based on MetaClass
    '''

    def set_defaults(self):
        '''Set Stage 1 specific defaults for generic instruments.

        Notes
        -----
        History:

        - 2024-03-05 Taylor J Bell
            Initial version setting defaults for any instrument.
        '''
        # Control parallelization
        # (Options are 'none', quarter', 'half', 'all', or any integer)
        self.maximum_cores = getattr(self, 'maximum_cores', 'half')

        # Control ramp fitting method
        self.ramp_fit_algorithm = getattr(self, 'ramp_fit_algorithm', 'default')

        # Pipeline stages
        self.skip_group_scale = getattr(self, 'skip_group_scale', False)
        self.skip_dq_init = getattr(self, 'skip_dq_init', False)
        self.skip_saturation = getattr(self, 'skip_saturation', False)
        self.skip_ipc = getattr(self, 'skip_ipc', True)  # Skipped by default for all instruments.
        self.skip_refpix = getattr(self, 'skip_refpix', False)
        self.skip_linearity = getattr(self, 'skip_linearity', False)
        self.skip_dark_current = getattr(self, 'skip_dark_current', False)
        self.skip_jump = getattr(self, 'skip_jump', False)
        self.skip_ramp_fitting = getattr(self, 'skip_ramp_fitting', False)
        self.skip_gain_scale = getattr(self, 'skip_gain_scale', False)

        # CR sigma rejection threshold
        self.jump_rejection_threshold = getattr(self, 'jump_rejection_threshold', 4.0)

        # Custom linearity reference file
        self.custom_linearity = getattr(self, 'custom_linearity', False)
        if self.custom_linearity:
            self.linearity_file = getattr(self, 'linearity_file')  # Force this to be specified if custom_linearity is True

        # Custom bias when using NIRSpec G395H
        # Options: [mean, group_level, smooth, None].
        # If not None, requires masktrace=True
        self.bias_correction = getattr(self, 'bias_correction', None)
        if self.bias_correction is not None:
            self.bias_group = getattr(self, 'bias_group', 1)  # Group number options: [1, 2, ..., each]
            if self.bias_correction == 'smooth':
                self.bias_smooth_length = getattr(self, 'bias_smooth_length')  # Force this to be specified if bias_correction is 'smooth'
            self.custom_bias = getattr(self, 'custom_bias', False)
            if self.custom_bias:
                self.superbias_file = getattr(self, 'superbias_file')  # Force this to be specified if custom_bias is True

        # Manually mask groups
        self.mask_groups = getattr(self, 'mask_groups', False)

        # Saturation
        self.update_sat_flags = getattr(self, 'update_sat_flags', False)  # Wheter to update the saturation flags more aggressively
        if self.update_sat_flags:
            self.expand_prev_group = getattr(self, 'expand_prev_group', False)  # Expand saturation flags to previous group
            self.dq_sat_mode = getattr(self, 'dq_sat_mode')  # Force this to be specified if dq_sat_mode is 'defined'. Options: [percentile, min, defined]
            if self.dq_sat_mode == 'percentile':
                self.dq_sat_percentile = getattr(self, 'dq_sat_percentile')  # Force this to be specified if dq_sat_mode is 'defined'. The percentile of the entire time series to use to define the saturation mask (50=median)
            elif self.dq_sat_mode == 'defined':
                self.dq_sat_columns = getattr(self, 'dq_sat_columns')  # Force this to be specified if dq_sat_mode is 'defined'

        # Mask curved traces
        self.masktrace = getattr(self, 'masktrace', False)
        if self.masktrace:
            self.window_len = getattr(self, 'window_len')  # Force this to be specified if masking the trace
            self.expand_mask = getattr(self, 'expand_mask')  # Force this to be specified if masking the trace
            self.ignore_low = getattr(self, 'ignore_low') # Force this to be specified if masking the trace
            self.ignore_hi = getattr(self, 'ignore_hi')  # Force this to be specified if masking the trace
        elif self.bias_correction:
            raise AssertionError(f'meta.bias_correction has been set to {self.bias_correction} which requires meta.masktrace=True. Meanwhile, meta.masktrace has been set to False. Please update bias_correction or masktrace in your Stage 1 ECF.')

        # Manual reference pixel correction for NIRSpec PRISM when not subtracting BG
        self.refpix_corr = getattr(self, 'refpix_corr', False)
        if self.refpix_corr:
            self.npix_top = getattr(self, 'npix_top')  # Force this to be specified if refpix_corr is True
            self.npix_bot = getattr(self, 'npix_bot')  # Force this to be specified if refpix_corr is True

        # Background subtraction
        self.grouplevel_bg = getattr(self, 'grouplevel_bg', False)
        if self.grouplevel_bg:
            self.ncpu = getattr(self, 'ncpu', 4)
            self.bg_y1 = getattr(self, 'bg_y1')  # Force this to be specified if grouplevel_bg is True
            self.bg_y2 = getattr(self, 'bg_y2')  # Force this to be specified if grouplevel_bg is True
            self.bg_deg = getattr(self, 'bg_deg', 0)
            self.bg_method = getattr(self, 'bg_method', 'mean')  # Options: std (Standard Deviation), median (Median Absolute Deviation), mean (Mean Absolute Deviation)
            self.p3thresh = getattr(self, 'p3thresh', 5)
            self.bg_disp = getattr(self, 'bg_disp', False)  # Row-by-row BG subtraction (only useful for NIRCam)
            self.bg_x1 = getattr(self, 'bg_x1', None)  # Left edge of exclusion region for row-by-row BG subtraction
            self.bg_x2 = getattr(self, 'bg_x2', None)  # Right edge of exclusion region for row-by-row BG subtraction

        # Diagnostics
        self.isplots_S1 = getattr(self, 'isplots_S1', 1)
        self.nplots = getattr(self, 'nplots', 5)
        self.hide_plots = getattr(self, 'hide_plots', False)
        self.testing_S1 = getattr(self, 'testing_S1', False)
        self.verbose = getattr(self, 'verbose', True)

        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Uncalibrated')
        self.outputdir = getattr(self, 'outputdir', 'Stage1')

        #####

        # "Default" ramp fitting settings
        self.default_ramp_fit_weighting = 'default'  # Options are "default", "fixed", "interpolated", "flat", or "custom"
        if self.default_ramp_fit_weighting == 'fixed':
            self.default_ramp_fit_fixed_exponent = getattr(self, 'default_ramp_fit_fixed_exponent')  # Force this to be specified if fixed weighting
        elif self.default_ramp_fit_weighting == 'fixed':
            self.default_ramp_fit_custom_snr_bounds = getattr(self, 'default_ramp_fit_custom_snr_bounds')  # Force this to be specified if fixed weighting
            self.default_ramp_fit_custom_exponents = getattr(self, 'default_ramp_fit_custom_exponents')  # Force this to be specified if fixed weighting

    def set_MIRI_defaults(self):
        '''Set Stage 1 specific defaults for MIRI.

        Notes
        -----
        History:

        - 2024-03-05 Taylor J Bell
            Initial version setting defaults for MIRI.
        '''
        # MIRI-specific pipeline stages
        self.skip_firstframe = getattr(self, 'skip_firstframe', True)  # jwst skips by default for MIRI TSO, but should likely be set False (that may depend on the dataset though).
        self.skip_lastframe = getattr(self, 'skip_lastframe', True)  # jwst skips by default for MIRI TSO, but should likely be set False (that may depend on the dataset though).
        self.skip_reset = getattr(self, 'skip_reset', False)
        self.skip_rscd = getattr(self, 'skip_rscd', True)  # jwst skips by default for MIRI TSO.

        # Remove the 390 Hz periodic noise in MIRI/LRS SLITLESSPRISM group-level data?
        self.remove_390hz = getattr(self, 'remove_390hz', False)
        if self.remove_390hz:
            self.grouplevel_bg = getattr(self, 'grouplevel_bg', True)  # Default to True if remove_390hz
        else:
            self.grouplevel_bg = getattr(self, 'grouplevel_bg', False)  # Default to False if not remove_390hz
    
    def set_NIR_defaults(self):
        '''Set Stage 1 specific defaults for NIR-instruments.

        Notes
        -----
        History:

        - 2024-03-05 Taylor J Bell
            Initial version setting defaults for NIR-instruments.
        '''
        # NIR-specific pipeline stages
        self.skip_superbias = getattr(self, 'skip_superbias', False)
        self.skip_persistence = getattr(self, 'skip_persistence', True)  # Skipped by default for Near-IR TSO.

        # Placeholder for Eureka_RampFitStep
        self.remove_390hz = getattr(self, 'remove_390hz', False)
        if self.remove_390hz:
            raise AssertionError('remove_390hz cannot be set to True for NIR instruments!')
