from ..lib.readECF import MetaClass

class S3MetaClass(MetaClass):
    '''A class to hold Eureka! S3 metadata.

    This class loads a Stage 3 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-03 Taylor J Bell
        Made specific S3 class based on MetaClass
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
        super.__init__(folder, file, eventlabel, stage=3, **kwargs)

    def set_defaults(self):
        '''Set Stage 3 specific defaults for generic instruments.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for any instrument.
        '''
        # Data file suffix
        self.suffix = getattr(self, 'suffix', 'calints')

        self.ncpu = hasattr(self, 'ncpu', 4)
        self.nfiles = hasattr(self, 'nfiles', 1000)  # By default, try to load a bunch of files if permitted by max_memory
        self.max_memory = hasattr(self, 'max_memory', 0.5)
        self.indep_batches = hasattr(self, 'indep_batches', False)
        self.calibrated_spectra = getattr(self, 'calibrated_spectra', False)

        # Reference files or values
        self.gain = getattr(self, 'gain', None)
        self.gainfile = getattr(meta, 'gainfile', None)
        self.photfile = getattr(self, 'photfile', None)
        self.time_file = getattr(self, 'time_file', None)

        # Subarray region of interest
        self.src_pos_type = getattr(self, 'ywindow')  # Require this to be set in the ECF
        self.record_ypos = getattr(self, 'xwindow')  # Require this to be set in the ECF
        self.src_pos_type = getattr(self, 'src_pos_type', 'gaussian')
        self.record_ypos = getattr(self, 'record_ypos', True)
        self.dqmask = getattr(self, 'dqmask', True)
        self.manmask = getattr(self, 'manmask', None)
        self.expand = getattr(self, 'expand', 1)
        self.isrotate = getattr(self, 'isrotate', 0)  # FINDME: I think bg_disp, bg_dir, and isrotate all control the same thing...

        # Outlier rejection along time axis
        self.ff_outlier = getattr(self, 'ff_outlier', False)
        self.bg_thresh = getattr(self, 'bg_thresh')  # Require this parameter to be set
        self.use_estsig = getattr(self, 'use_estsig', False)

        # Diagnostics
        self.isplots_S3 = getattr(self, 'isplots_S3', 3)
        self.nplots = getattr(self, 'nplots', None)
        self.vmin = getattr(self, 'vmin', 0.97)
        self.vmax = getattr(self, 'vmax', 1.03)
        self.time_axis = getattr(self, 'time_axis', 'y')
        if self.time_axis not in ['y', 'x']:
            print("WARNING: meta.time_axis is not one of ['y', 'x']!"
                  " Using 'y' by default.")
            self.time_axis = 'y'
        self.testing_S3 = getattr(self, 'testing_S3', False)
        self.hide_plots = getattr(self, 'hide_plots', False)
        self.save_output = getattr(self, 'save_output', True)
        self.save_fluxdata = getattr(self, 'save_fluxdata', True)
        self.verbose = getattr(self, 'verbose', True)
        
        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage2')
        self.outputdir = getattr(self, 'outputdir', 'Stage3')

    def set_spectral_defaults(self):
        '''Set Stage 3 specific defaults for generic spectroscopic data.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for any spectroscopic data.
        '''
        # Spectral extraction parameters
        self.spec_hw = getattr(self, 'spec_hw')  # Require this parameter to be set
        self.fittype = getattr(self, 'fittype', 'meddata')
        self.median_thresh = getattr(self, 'median_thresh', 5)
        if self.fittype in ['meddata', 'smooth']:
            self.window_len = getattr(self, 'window_len')  # Require this parameter to be set if relevant
        elif self.fittype == 'poly':
            self.prof_deg = getattr(self, 'prof_deg')  # Require this parameter to be set if relevant
        if self.fittype in ['smooth', 'gauss', 'poly']:
            self.p5thresh = getattr(self, 'p5thresh')  # Require this parameter to be set if relevant
        else:
            self.p5thresh = getattr(self, 'p5thresh', None)  # Set it to None if not relevant
        self.p7thresh = getattr(self, 'p7thresh')  # Require this parameter to be set

        # Curvature correction
        self.curvature = getattr(self, 'curvature', None)  # By default, don't correct curvature

        # Background parameters
        self.bg_hw = getattr(self, 'bg_hw')  # Require this parameter to be set
        self.bg_deg = getattr(self, 'bg_deg', 0)
        self.bg_disp = getattr(self, 'bg_disp', False)  # FINDME: I think bg_disp, bg_dir, and isrotate all control the same thing...
        self.bg_dir = getattr(self, 'bg_dir', 'CxC')  # FINDME: I think bg_disp, bg_dir, and isrotate all control the same thing...
        self.bg_x1 = getattr(self, 'bg_x1', None)
        self.bg_x2 = getattr(self, 'bg_x2', None)
        self.bg_method = getattr(self, 'bg_method', 'mean')
        self.p3thresh = getattr(self, 'p3thresh', 5)

    def set_photometric_defaults(self):
        '''Set Stage 3 specific defaults for generic photometric data.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for any photometric data.
        '''
        self.expand = getattr(self, 'expand', 1)
        if self.expand > 1:
            # Super sampling not supported for photometry
            # This is here just in case someone tries to super sample
            print("Super sampling not supported for photometry."
                  "Setting meta.expand to 1.")
            self.expand = 1

        self.flag_bg = getattr(self, 'flag_bg', True)
        self.interp_method = getattr(self, 'interp_method', 'cubic')
        self.ctr_guess = getattr(self, 'ctr_guess', None)
        self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 5)
        self.oneoverf_corr = getattr(self, 'oneoverf_corr', None)
        self.centroid_method = getattr(self, 'centroid_method', 'fgc')
        self.skip_apphot_bg = getattr(self, 'skip_apphot_bg', False)
        self.photap = getattr(self, 'photap')  # Require this parameter to be set
        self.skyin = getattr(self, 'skyin')  # Require this parameter to be set
        self.skywidth = getattr(self, 'skywidth')  # Require this parameter to be set
        if self.centroid_method == 'mgmc':
            self.centroid_tech = getattr(self, 'centroid_tech', 'com')
            self.gauss_frame = getattr(self, 'gauss_frame', 15)  # For MIRI
            self.gauss_frame = getattr(self, 'gauss_frame', 100)  # For NIRCam

    def set_MIRI_defaults(self):
        '''Set Stage 3 specific defaults for MIRI.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for MIRI.
        '''
        self.isrotate = getattr(self, 'isrotate', 2)

        self.set_spectral_defaults()

    def set_NIRCam_defaults(self):
        '''Set Stage 3 specific defaults for NIRCam.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for NIRCam.
        '''
        self.poly_wavelength = getattr(self, 'poly_wavelength', False)
        self.curvature = getattr(self, 'curvature', True)

        self.set_spectral_defaults()

    def set_NIRSpec_defaults(self):
        '''Set Stage 3 specific defaults for NIRSpec.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for NIRSpec.
        '''
        self.curvature = getattr(self, 'curvature', True)

        self.set_spectral_defaults()

    def set_NIRISS_defaults(self):
        '''Set Stage 3 specific defaults for NIRISS.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for NIRISS.
        '''
        self.curvature = getattr(self, 'curvature', True)

        self.set_spectral_defaults()

    def set_WFC3_defaults(self):
        '''Set Stage 3 specific defaults for WFC3.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for WFC3.
        '''
        if not hasattr(self, 'iref'):
            # Should we just set iref to [2, 3] by default?
            raise AttributeError(
                'You must set the meta.iref parameter in your ECF for WFC3 '
                'observations. The recommended setting is [2, 3].'
            )
        self.horizonsfile = getattr(self, 'horizonsfile', None)
        if self.horizonsfile is not None:
            self.hst_cal = getattr(self, 'hst_cal')  # Require this parameter to be set if relevant
        self.leapdir = getattr(self, 'leapdir', 'leapdir')
        self.flatfile = getattr(self, 'flatfile', None)

        self.isrotate = getattr(self, 'isrotate', 2)

        self.set_spectral_defaults()

    def set_NIRCam_Photometry_defaults(self):
        '''Set Stage 3 specific defaults for NIRCam Photometry.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for NIRCam Photometry.
        '''
        self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 5)
        self.oneoverf_corr = getattr(self, 'oneoverf_corr', 'median')
        self.oneoverf_dist = getattr(self, 'oneoverf_dist', 350)

        self.set_photometric_defaults()

    def set_MIRI_Photometry_defaults(self):
        '''Set Stage 3 specific defaults for MIRI Photometry.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for MIRI Photometry.
        '''
        self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 10)
        self.oneoverf_corr = getattr(self, 'oneoverf_corr', None)
        if self.oneoverf_corr is not None:
            raise AssertionError('Cannot apply the oneoverf_corr step to MIRI data.')

        self.set_photometric_defaults()

    def setup_aperture_radii(self):
        '''
        '''
        # check for range of spectral apertures
        if self.photometry:
            if isinstance(self.photap, list):
                self.spec_hw_range = np.arange(self.photap[0],
                                               self.photap[1]+self.photap[2],
                                               self.photap[2])
            else:
                self.spec_hw_range = np.array([self.photap])
            # Super sampling not supported for photometry
        else:
            if isinstance(self.spec_hw, list):
                self.spec_hw_range = np.arange(self.spec_hw[0],
                                               self.spec_hw[1]+self.spec_hw[2],
                                               self.spec_hw[2])
            else:
                self.spec_hw_range = np.array([self.spec_hw])
            # Increase relevant self parameter values
            self.spec_hw_range *= self.expand

        # check for range of background apertures
        if hasattr(self, 'bg_hw'):
            if isinstance(self.bg_hw, list):
                self.bg_hw_range = np.arange(self.bg_hw[0],
                                            self.bg_hw[1]+self.bg_hw[2],
                                            self.bg_hw[2])
            else:
                self.bg_hw_range = np.array([self.bg_hw])
            self.bg_hw_range *= self.expand
        elif hasattr(self, 'skyin') and hasattr(self, 'skywidth'):
            # E.g., if skyin = 90 and skywidth = 60, then the
            # directory will use "bg90_150"
            if not isinstance(self.skyin, list):
                self.skyin = [self.skyin]
            else:
                self.skyin = range(self.skyin[0],
                                self.skyin[1]+self.skyin[2],
                                self.skyin[2])
            if not isinstance(self.skywidth, list):
                self.skywidth = [self.skywidth]
            else:
                self.skywidth = range(self.skywidth[0],
                                    self.skywidth[1]+self.skywidth[2],
                                    self.skywidth[2])
            self.bg_hw_range = [f'{skyin}_{skyin+skywidth}'
                                for skyin in self.skyin
                                for skywidth in self.skywidth]
