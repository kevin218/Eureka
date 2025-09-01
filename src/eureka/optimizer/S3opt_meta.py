import numpy as np
from jwst.datamodels import JwstDataModel

from ..lib.readECF import MetaClass


class S3optMetaClass(MetaClass):
    '''A class to hold Eureka! Optimizer metadata.

    This class loads an Optimizer Eureka! Control File (ecf) and lets you
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

        super().__init__(folder, file, eventlabel, stage='3opt', **kwargs)

        # Set a default data file suffix
        self.suffix = getattr(self, 'suffix', 'calints')

    def set_defaults(self):
        '''Set Optimizer specific defaults for generic instruments.
        '''
        # Make sure the inst, filt, and src_ypos attributes are at
        # least initialized
        # self.inst = getattr(self, 'inst', None)
        # self.filter = getattr(self, 'filter', None)
        # self.src_ypos = getattr(self, 'src_ypos', None)

        # self.ncpu = getattr(self, 'ncpu', 4)
        # # By default, try to load a bunch of files if permitted by max_memory
        # self.nfiles = getattr(self, 'nfiles', 1000)
        # self.max_memory = getattr(self, 'max_memory', 0.5)
        # self.indep_batches = getattr(self, 'indep_batches', False)
        # self.calibrated_spectra = getattr(self, 'calibrated_spectra', False)

        # # Reference files or values
        # self.gain = getattr(self, 'gain', None)
        # self.gainfile = getattr(self, 'gainfile', None)
        # self.photfile = getattr(self, 'photfile', None)
        # self.time_file = getattr(self, 'time_file', None)

        # # Subarray region of interest
        # # Require these to be set in the ECF
        # self.ywindow = getattr(self, 'ywindow', [None, None])
        # self.xwindow = getattr(self, 'xwindow', [None, None])
        # if self.xwindow is None:
        #     self.xwindow = [None, None]
        # if self.ywindow is None:
        #     self.ywindow = [None, None]

        # # Auto-populate the xwindow and ywindow if they are None
        # if self.xwindow[0] is None:
        #     self.xwindow[0] = 0
        # if self.xwindow[1] is None:
        #     with JwstDataModel(self.segment_list[0]) as model:
        #         self.xwindow[1] = model.data.shape[2]-1
        # if self.ywindow[0] is None:
        #     self.ywindow[0] = 0
        # if self.ywindow[1] is None:
        #     with JwstDataModel(self.segment_list[0]) as model:
        #         self.ywindow[1] = model.data.shape[1]-1

        # self.src_pos_type = getattr(self, 'src_pos_type', 'gaussian')
        # self.record_ypos = getattr(self, 'record_ypos', True)
        # self.dqmask = getattr(self, 'dqmask', True)
        # self.manmask = getattr(self, 'manmask', None)
        # self.expand = getattr(self, 'expand', 1)
        # if self.expand != int(self.expand) or self.expand < 1:
        #     raise ValueError('meta.expand must be an integer >= 1, but got '
        #                      f'{self.expand}')
        # else:
        #     self.expand = int(self.expand)

        # # Outlier rejection along time axis
        # self.ff_outlier = getattr(self, 'ff_outlier', False)
        # self.use_estsig = getattr(self, 'use_estsig', False)
        # # Require this parameter to be set
        # self.bg_thresh = getattr(self, 'bg_thresh')

        # # Diagnostics
        # self.isplots_S3 = getattr(self, 'isplots_S3', 3)
        # self.nplots = getattr(self, 'nplots', 5)
        # self.vmin = getattr(self, 'vmin', 0.97)
        # self.vmax = getattr(self, 'vmax', 1.03)
        # self.time_axis = getattr(self, 'time_axis', 'y')
        # if self.time_axis not in ['y', 'x']:
        #     print("WARNING: meta.time_axis is not one of ['y', 'x']!"
        #           " Using 'y' by default.")
        #     self.time_axis = 'y'
        # self.testing_S3 = getattr(self, 'testing_S3', False)
        # self.hide_plots = getattr(self, 'hide_plots', True)
        # self.save_output = getattr(self, 'save_output', True)
        # self.save_fluxdata = getattr(self, 'save_fluxdata', False)
        # self.verbose = getattr(self, 'verbose', True)

        # # Project directory
        # self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage2')
        self.outputdir = getattr(self, 'outputdir', 'Optimizer')

    def set_spectral_defaults(self):
        '''Set Optimizer specific defaults for generic spectroscopic data.
        '''
        # Spectral extraction parameters
        # Require this parameter to be set
        self.bounds_spec_hw = getattr(self, 'bounds_spec_hw', [1, 15])


    def set_photometric_defaults(self):
        '''Set Optimizer specific defaults for generic photometric data.
        '''
        # self.expand = getattr(self, 'expand', 1)
        # if self.expand > 1:
        #     # FINDME: We should soon be able to support expand != 1
        #     # for photometry
        #     # (only relevant for POET method of aperture photometry)
        #     print("Super sampling is not currently supported for photometry. "
        #           "Setting meta.expand to 1.")
        #     self.expand = 1

        # self.ff_outlier = getattr(self, 'ff_outlier', False)
        return


    def set_MIRI_defaults(self):
        '''Set Optimizer specific defaults for MIRI.
        '''
        self.set_spectral_defaults()
        # self.isrotate = 2
        # self.bg_dir = 'CxC'
        # self.bg_row_by_row = False

    def set_NIRCam_defaults(self):
        '''Set Optimizer specific defaults for NIRCam.
        '''
        # self.poly_wavelength = getattr(self, 'poly_wavelength', False)
        # self.wave_pixel_offset = getattr(self, 'wave_pixel_offset', None)
        # self.curvature = getattr(self, 'curvature', True)

        self.set_spectral_defaults()

    def set_NIRSpec_defaults(self):
        '''Set Optimizer specific defaults for NIRSpec.
        '''
        # self.curvature = getattr(self, 'curvature', True)
        # # When calibrated_spectra is True, flux values above the cutoff
        # # will be set to zero.
        # self.cutoff = getattr(self, 'cutoff', 1e-4)

        self.set_spectral_defaults()

    def set_NIRISS_defaults(self):
        '''Set Optimizer specific defaults for NIRISS.
        '''
        # self.curvature = getattr(self, 'curvature', True)
        # self.src_ypos = getattr(self, 'src_ypos', [35, 80])
        # self.orders = getattr(self, 'orders', [1, 2])
        # self.all_orders = getattr(self, 'all_orders', [1, 2])
        # self.record_ypos = getattr(self, 'record_ypos', False)
        # self.trace_offset = getattr(self, 'trace_offset', None)

        self.set_spectral_defaults()

    def set_WFC3_defaults(self):
        '''Set Optimizer specific defaults for WFC3.
        '''
        # self.iref = getattr(self, 'iref', [2, 3])
        # self.horizonsfile = getattr(self, 'horizonsfile', None)
        # if self.horizonsfile is not None:
        #     # Require this parameter to be set if relevant
        #     self.hst_cal = getattr(self, 'hst_cal')
        # self.leapdir = getattr(self, 'leapdir', 'leapdir')
        # self.flatfile = getattr(self, 'flatfile', None)
        # # Applying DQ mask doesn't seem to work for WFC3
        # self.dqmask = getattr(self, 'dqmask', False)

        self.set_spectral_defaults()

    def set_NIRCam_Photometry_defaults(self):
        '''Set Optimizer specific defaults for NIRCam Photometry.
        '''
        # self.ctr_cutout_size = getattr(self, 'ctr_cutout_size', 5)
        # self.oneoverf_corr = getattr(self, 'oneoverf_corr', 'median')
        # self.oneoverf_dist = getattr(self, 'oneoverf_dist', 350)

        # self.centroid_method = getattr(self, 'centroid_method', 'fgc')
        # if self.centroid_method == 'mgmc':
        #     self.gauss_frame = getattr(self, 'gauss_frame', 100)

        self.set_photometric_defaults()

    def set_MIRI_Photometry_defaults(self):
        '''Set Optimizer specific defaults for MIRI Photometry.
        '''

        self.set_photometric_defaults()
