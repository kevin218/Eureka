import numpy as np
from jwst.datamodels import JwstDataModel

from ..lib.readECF import MetaClass


class S3MetaClass(MetaClass):
    '''A class to hold Eureka! S3 metadata.

    This class loads a Stage 3 Eureka! Control File (ecf) and lets you
    query the parameters and values.
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
        '''
        # Remove the stage from kwargs if present
        if 'stage' in kwargs:
            kwargs.pop('stage')

        super().__init__(folder, file, eventlabel, stage=3, **kwargs)

        # Set a default data file suffix
        self.suffix = getattr(self, 'suffix', 'calints')

    def set_defaults(self):
        '''Set Stage 3 specific defaults for generic instruments.
        '''
        # Make sure the inst, filt, and src_ypos attributes are at
        # least initialized
        self.inst = getattr(self, 'inst', None)
        self.filter = getattr(self, 'filter', None)
        self.src_ypos = getattr(self, 'src_ypos', None)

        self.ncpu = getattr(self, 'ncpu', 4)
        # By default, try to load a bunch of files if permitted by max_memory
        self.nfiles = getattr(self, 'nfiles', 1000)
        self.max_memory = getattr(self, 'max_memory', 0.5)
        self.indep_batches = getattr(self, 'indep_batches', False)
        self.calibrated_spectra = getattr(self, 'calibrated_spectra', False)

        # Reference files or values
        self.gain = getattr(self, 'gain', None)
        self.gainfile = getattr(self, 'gainfile', None)
        self.photfile = getattr(self, 'photfile', None)
        self.time_file = getattr(self, 'time_file', None)

        # Subarray region of interest
        # Require these to be set in the ECF
        self.ywindow = getattr(self, 'ywindow', [None, None])
        self.xwindow = getattr(self, 'xwindow', [None, None])
        if self.xwindow is None:
            self.xwindow = [None, None]
        if self.ywindow is None:
            self.ywindow = [None, None]

        # Auto-populate the xwindow and ywindow if they are None
        if self.xwindow[0] is None:
            self.xwindow[0] = 0
        if self.xwindow[1] is None:
            with JwstDataModel(self.segment_list[0]) as model:
                self.xwindow[1] = model.data.shape[2]-1
        if self.ywindow[0] is None:
            self.ywindow[0] = 0
        if self.ywindow[1] is None:
            with JwstDataModel(self.segment_list[0]) as model:
                self.ywindow[1] = model.data.shape[1]-1

        self.src_pos_type = getattr(self, 'src_pos_type', 'gaussian')
        self.record_ypos = getattr(self, 'record_ypos', True)
        self.dqmask = getattr(self, 'dqmask', True)
        self.manmask = getattr(self, 'manmask', None)
        self.expand = getattr(self, 'expand', 1)
        if self.expand != int(self.expand) or self.expand < 1:
            raise ValueError('meta.expand must be an integer >= 1, but got '
                             f'{self.expand}')
        else:
            self.expand = int(self.expand)

        # Outlier rejection along time axis
        self.ff_outlier = getattr(self, 'ff_outlier', False)
        self.use_estsig = getattr(self, 'use_estsig', False)
        # Require this parameter to be set
        self.bg_thresh = getattr(self, 'bg_thresh')

        # Diagnostics
        self.isplots_S3 = getattr(self, 'isplots_S3', 3)
        self.nplots = getattr(self, 'nplots', 5)
        self.vmin = getattr(self, 'vmin', 0.97)
        self.vmax = getattr(self, 'vmax', 1.03)
        self.time_axis = getattr(self, 'time_axis', 'y')
        if self.time_axis not in ['y', 'x']:
            print("WARNING: meta.time_axis is not one of ['y', 'x']!"
                  " Using 'y' by default.")
            self.time_axis = 'y'
        self.testing_S3 = getattr(self, 'testing_S3', False)
        self.hide_plots = getattr(self, 'hide_plots', True)
        self.save_output = getattr(self, 'save_output', True)
        self.save_fluxdata = getattr(self, 'save_fluxdata', False)
        self.verbose = getattr(self, 'verbose', True)

        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage2')
        self.outputdir = getattr(self, 'outputdir', 'Stage3')

    def set_spectral_defaults(self):
        '''Set Stage 3 specific defaults for generic spectroscopic data.
        '''
        # Spectral extraction parameters
        # Require this parameter to be set
        self.spec_hw = getattr(self, 'spec_hw')
        self.fittype = getattr(self, 'fittype', 'meddata')
        self.median_thresh = getattr(self, 'median_thresh', 5)
        if self.fittype in ['meddata', 'smooth']:
            # Require this parameter to be set if relevant
            self.window_len = getattr(self, 'window_len')
        if self.fittype == 'poly':
            # Require this parameter to be set if relevant
            self.prof_deg = getattr(self, 'prof_deg')
        else:
            # Set it to None if not relevant
            self.prof_deg = self.prof_deg = None
        if self.fittype in ['smooth', 'gauss', 'poly']:
            # Require this parameter to be set if relevant
            self.p5thresh = getattr(self, 'p5thresh')
        else:
            # Set it to None if not relevant
            self.p5thresh = None
        # Require this parameter to be set
        self.p7thresh = getattr(self, 'p7thresh')

        # Curvature correction
        # By default, don't correct curvature
        self.curvature = getattr(self, 'curvature', None)

        # Background parameters
        self.skip_bg = getattr(self, 'skip_bg', False)
        self.bg_hw = getattr(self, 'bg_hw')  # Require this parameter to be set
        self.bg_deg = getattr(self, 'bg_deg', 0)
        self.bg_row_by_row = getattr(self, 'bg_row_by_row', False)
        self.bg_x1 = getattr(self, 'bg_x1', None)
        self.bg_x2 = getattr(self, 'bg_x2', None)
        self.bg_method = getattr(self, 'bg_method', 'mean')
        self.p3thresh = getattr(self, 'p3thresh', 5)
        self.orders = getattr(self, 'orders', None)

    def set_photometric_defaults(self):
        '''Set Stage 3 specific defaults for generic photometric data.
        '''
        self.expand = getattr(self, 'expand', 1)
        if self.expand > 1:
            # FINDME: We should soon be able to support expand != 1
            # for photometry
            # (only relevant for POET method of aperture photometry)
            print("Super sampling is not currently supported for photometry. "
                  "Setting meta.expand to 1.")
            self.expand = 1

        self.ff_outlier = getattr(self, 'ff_outlier', False)

        # Require window_len to be set to 0 to avoid smoothing in
        # optspex.get_clean
        self.window_len = getattr(self, 'window_len', 0)
        if self.window_len != 0:
            print("Warning: meta.window_len is not 0 which is not permitted "
                  "for photometric data! Setting it to 0.")
            self.window_len = 0

        # Centroiding parameters
        self.centroid_method = getattr(self, 'centroid_method', 'fgc')
        if self.centroid_method == 'mgmc':
            self.centroid_tech = getattr(self, 'centroid_tech', 'com')
            self.gauss_frame = getattr(self, 'gauss_frame', 15)
        self.ctr_guess = getattr(self, 'ctr_guess', None)
        self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 5)

        self.oneoverf_corr = getattr(self, 'oneoverf_corr', None)

        # Photometric extraction parameters
        self.phot_method = getattr(self, 'phot_method', 'poet')
        self.aperture_shape = getattr(self, 'aperture_shape', 'circle')
        if self.phot_method in ['photutils', 'optimal']:
            self.aperture_edge = getattr(self, 'aperture_edge', 'center')
            if self.aperture_edge not in ['center', 'exact']:
                raise ValueError('aperture_edge must be "center" or "exact", '
                                 f'but got {self.aperture_edge}')
            if self.aperture_shape not in ['circle', 'ellipse', 'rectangle']:
                raise ValueError('aperture_shape must be "circle", "ellipse", '
                                 'or "rectangle", but got '
                                 f'{self.aperture_shape}')
            if self.aperture_shape != 'circle':
                self.photap_b = getattr(self, 'photap_b', self.photap)
                self.photap_theta = getattr(self, 'photap_theta', 0)
            else:
                self.photap_b = self.photap
                self.photap_theta = 0
        elif self.phot_method == 'poet':
            self.aperture_edge = getattr(self, 'aperture_edge', 'center')
            if self.aperture_edge != 'center':
                raise ValueError('aperture_edge must be "center" for poet, '
                                 f'but got {self.aperture_edge}')
            if self.aperture_shape not in ['circle', 'hexagon']:
                raise ValueError('aperture_shape must be "circle" or '
                                 f'"hexagon", but got {self.aperture_shape}')
            self.betahw = getattr(self, 'betahw', None)
            if self.betahw is not None and self.betahw <= 0:
                raise ValueError(f'betahw must be > 0, but got {self.betahw}')
            elif self.betahw is not None and self.betahw != int(self.betahw):
                raise ValueError(f'betahw must be an integer, but got '
                                 f'{self.betahw}')
            elif self.betahw is not None:
                self.betahw = int(self.betahw)
        self.moving_centroid = getattr(self, 'moving_centroid', False)
        self.interp_method = getattr(self, 'interp_method', 'cubic')
        self.skip_apphot_bg = getattr(self, 'skip_apphot_bg', False)
        # Require these parameters to be set
        self.photap = getattr(self, 'photap')
        self.skyin = getattr(self, 'skyin')
        self.skywidth = getattr(self, 'skywidth')
        self.minskyfrac = getattr(self, 'minskyfrac', 0.1)
        if not self.skip_apphot_bg and not (0 < self.minskyfrac <= 1):
            raise ValueError(f'skyfrac is {self.minskyfrac} but must be in '
                             'range (0,1]')

        self.bg_row_by_row = getattr(self, 'bg_row_by_row', False)
        self.bg_x1 = getattr(self, 'bg_x1', None)
        self.bg_x2 = getattr(self, 'bg_x2', None)
        self.bg_method = getattr(self, 'bg_method', 'median')
        if (not self.skip_apphot_bg and
                self.bg_method not in ['mean', 'median']):
            raise ValueError(f'bg_method must be "mean" or "median", but got '
                             f'{self.bg_method}')
        self.p3thresh = getattr(self, 'p3thresh', 5)

    def set_MIRI_defaults(self):
        '''Set Stage 3 specific defaults for MIRI.
        '''
        self.set_spectral_defaults()
        self.isrotate = 2
        self.bg_dir = 'CxC'
        self.bg_row_by_row = False

    def set_NIRCam_defaults(self):
        '''Set Stage 3 specific defaults for NIRCam.
        '''
        self.poly_wavelength = getattr(self, 'poly_wavelength', False)
        self.wave_pixel_offset = getattr(self, 'wave_pixel_offset', None)
        self.curvature = getattr(self, 'curvature', True)

        self.set_spectral_defaults()

    def set_NIRSpec_defaults(self):
        '''Set Stage 3 specific defaults for NIRSpec.
        '''
        self.curvature = getattr(self, 'curvature', True)
        # When calibrated_spectra is True, flux values above the cutoff
        # will be set to zero.
        self.cutoff = getattr(self, 'cutoff', 1e-4)

        self.set_spectral_defaults()

    def set_NIRISS_defaults(self):
        '''Set Stage 3 specific defaults for NIRISS.
        '''
        self.curvature = getattr(self, 'curvature', True)
        self.src_ypos = getattr(self, 'src_ypos', [35, 80])
        self.orders = getattr(self, 'orders', [1, 2])
        self.all_orders = getattr(self, 'all_orders', [1, 2])
        self.record_ypos = getattr(self, 'record_ypos', False)
        self.trace_offset = getattr(self, 'trace_offset', None)

        self.set_spectral_defaults()

    def set_WFC3_defaults(self):
        '''Set Stage 3 specific defaults for WFC3.
        '''
        self.iref = getattr(self, 'iref', [2, 3])
        self.horizonsfile = getattr(self, 'horizonsfile', None)
        if self.horizonsfile is not None:
            # Require this parameter to be set if relevant
            self.hst_cal = getattr(self, 'hst_cal')
        self.leapdir = getattr(self, 'leapdir', 'leapdir')
        self.flatfile = getattr(self, 'flatfile', None)
        # Applying DQ mask doesn't seem to work for WFC3
        self.dqmask = getattr(self, 'dqmask', False)

        self.set_spectral_defaults()

    def set_NIRCam_Photometry_defaults(self):
        '''Set Stage 3 specific defaults for NIRCam Photometry.
        '''
        self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 5)
        self.oneoverf_corr = getattr(self, 'oneoverf_corr', 'median')
        self.oneoverf_dist = getattr(self, 'oneoverf_dist', 350)

        self.centroid_method = getattr(self, 'centroid_method', 'fgc')
        if self.centroid_method == 'mgmc':
            self.gauss_frame = getattr(self, 'gauss_frame', 100)

        self.set_photometric_defaults()

    def set_MIRI_Photometry_defaults(self):
        '''Set Stage 3 specific defaults for MIRI Photometry.
        '''
        self.xwindow = getattr(self, 'xwindow', [None, None])
        self.ywindow = getattr(self, 'ywindow', [None, None])
        if self.xwindow is None:
            self.xwindow = [None, None]
        if self.ywindow is None:
            self.ywindow = [None, None]

        # Auto-populate the xwindow and ywindow if any elements are None
        if None in [*self.xwindow, *self.ywindow]:
            # If any of the xwindow or ywindow elements are None, then
            # set them to default values based on the specified
            # subarray size. By default use a 151x151 subarray centered on the
            # centroid
            self.subarray_halfwidth = getattr(self, 'subarray_halfwidth', 75)
            if not isinstance(self.subarray_halfwidth, (int, np.int16,
                                                        np.int32, np.int64)):
                print("Warning: meta.subarray_halfwidth is not an integer! "
                      f"Rounding the input value of {self.subarray_halfwidth} "
                      "to the nearest integer.")
                self.subarray_halfwidth = round(self.subarray_halfwidth)
            if self.subarray_halfwidth < 0:
                raise ValueError(f'meta.subarray_halfwidth must be >= 0, but '
                                 f'got {self.subarray_halfwidth}')

            # Get the centroid position and the subarray size
            with JwstDataModel(self.segment_list[0]) as model:
                guess = [model.meta.wcsinfo.crpix1,
                         model.meta.wcsinfo.crpix2]
                ysize, xsize = model.data.shape[1:]

            if self.xwindow[0] is None:
                self.xwindow[0] = max(0, round(guess[0] -
                                               self.subarray_halfwidth))
            if self.xwindow[1] is None:
                self.xwindow[1] = min(xsize, round(guess[0] +
                                                   self.subarray_halfwidth+1))
            if self.ywindow[0] is None:
                self.ywindow[0] = max(0, round(guess[1] -
                                               self.subarray_halfwidth))
            if self.ywindow[1] is None:
                self.ywindow[1] = min(ysize, round(guess[1] +
                                                   self.subarray_halfwidth+1))

        self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 10)
        self.oneoverf_corr = getattr(self, 'oneoverf_corr', None)
        if self.oneoverf_corr is not None:
            raise AssertionError('Cannot apply the oneoverf_corr step to '
                                 'MIRI data.')

        self.centroid_method = getattr(self, 'centroid_method', 'fgc')
        if self.centroid_method == 'mgmc':
            self.gauss_frame = getattr(self, 'gauss_frame', 15)

        self.set_photometric_defaults()

    def setup_aperture_radii(self):
        '''A function to set up the spectral and background aperture radii.
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
        if hasattr(self, 'skyin') and hasattr(self, 'skywidth'):
            # E.g., if skyin = 90 and skywidth = 60, then the
            # directory will use "bg90_150"
            if not isinstance(self.skyin, list):
                self.skyin = [self.skyin]
            else:
                self.skyin = np.arange(self.skyin[0],
                                       self.skyin[1]+self.skyin[2],
                                       self.skyin[2])
            if not isinstance(self.skywidth, list):
                self.skywidth = [self.skywidth]
            else:
                self.skywidth = np.arange(self.skywidth[0],
                                          self.skywidth[1]+self.skywidth[2],
                                          self.skywidth[2])

            self.bg_hw_range = []
            for skyin in self.skyin:
                for skywidth in self.skywidth:
                    skyout = skyin + skywidth
                    if isinstance(skyout, float):
                        # Avoid annoying floating point precision issues
                        # by rounding to the same number of decimal places
                        # as the input values
                        ndecimals_skyin = len(str(float(skyin)).split('.')[1])
                        ndecimals_skywidth = len(str(float(skywidth)
                                                     ).split('.')[1])
                        ndecimals_skyout = max(ndecimals_skyin,
                                               ndecimals_skywidth)
                        skyout = np.round(skyout, ndecimals_skyout)
                    self.bg_hw_range.append(f'{skyin}_{skyout}')
        elif hasattr(self, 'bg_hw'):
            if isinstance(self.bg_hw, list):
                self.bg_hw_range = np.arange(self.bg_hw[0],
                                             self.bg_hw[1]+self.bg_hw[2],
                                             self.bg_hw[2])
            else:
                self.bg_hw_range = np.array([self.bg_hw])
            self.bg_hw_range *= self.expand
