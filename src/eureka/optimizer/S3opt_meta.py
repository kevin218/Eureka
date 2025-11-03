import numpy as np
from jwst.datamodels import JwstDataModel

from ..lib.readECF import MetaClass
from ..lib import util


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

        # Set optimization flag
        self.isopt_S1 = getattr(self, 'isopt_S1', False)
        self.isopt_S3 = getattr(self, 'isopt_S3', True)

        # Create list of file segments
        self = util.readfiles(self)

        # First apply any instrument-specific defaults
        if self.photometry:
            if self.inst == 'miri':
                self.set_MIRI_Photometry_defaults()
            elif self.inst == 'nircam':
                self.set_NIRCam_Photometry_defaults()
        else:
            if self.inst == 'miri':
                self.set_MIRI_defaults()
            elif self.inst == 'nircam':
                self.set_NIRCam_defaults()
            elif self.inst == 'nirspec':
                self.set_NIRSpec_defaults()
            elif self.inst == 'niriss':
                self.set_NIRISS_defaults()
            elif self.inst == 'wfc3':
                self.set_WFC3_defaults()
        # Then apply instrument-agnostic defaults
        self.set_defaults()


    def set_defaults(self):
        '''
        Set Optimizer specific defaults for generic instruments.
        '''
        defaults = {
            "inputdir": 'Stage2',
            "outputdir": 'Optimizer',
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))


    def set_spectral_defaults(self):
        '''
        Set Optimizer specific defaults for generic spectroscopic data.
        '''
        # Spectral extraction parameters
        defaults = {
            # Stage 3
            "bounds_dqmask": [True, False],  # not a range
            "bounds_bg_thresh": np.arange(3, 5.5, 0.5),  # 3 → 5, step 0.5
            "bounds_bg_method": ["std", "mean", "median"],
            "bounds_bg_deg": range(0, 2),  # 0 → 1
            "bounds_p3thresh": range(3, 8),  # 3 → 7
            "bounds_median_thresh": range(3, 10),  # 3 → 9
            "bounds_window_len": range(1, 22, 2),  # 1 → 21, odd only
            "bounds_p7thresh": range(5, 61, 5),  # 5 → 60, step 5
            # Stage 4
            "bounds_sigma": range(3, 8),  # 3 → 7
            "bounds_box_width": range(11, 52, 10),  # 11 → 51, step 10
            "bounds_mad_sigma": range(4, 8),  # 4 → 7
            "bounds_mad_box_width": range(11, 52, 10),  # 11 → 51, step 10
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))


    def set_photometric_defaults(self):
        '''
        Set Optimizer specific defaults for generic photometric data.
        '''
        return


    def set_MIRI_defaults(self):
        '''
        Set Optimizer specific defaults for MIRI.
        '''
        defaults = {
            "bounds_bg_hw": range(5, 16),
            "bounds_spec_hw": range(1, 10),
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))
        self.bounds_spec_hw__bg_hw = getattr(self, 'bounds_spec_hw__bg_hw',
                                             [self.bounds_bg_hw,
                                              self.bounds_spec_hw])

        self.set_spectral_defaults()


    def set_NIRCam_defaults(self):
        '''
        Set Optimizer specific defaults for NIRCam.
        '''
        defaults = {
            "bounds_bg_hw": range(5, 16),
            "bounds_spec_hw": range(1, 10),
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))
        self.bounds_spec_hw__bg_hw = getattr(self, 'bounds_spec_hw__bg_hw',
                                             [self.bounds_bg_hw,
                                              self.bounds_spec_hw])

        self.set_spectral_defaults()


    def set_NIRSpec_defaults(self):
        '''
        Set Optimizer specific defaults for NIRSpec.
        '''
        defaults = {
            "bounds_bg_hw": range(5, 16),
            "bounds_spec_hw": range(1, 10),
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))
        self.bounds_spec_hw__bg_hw = getattr(self, 'bounds_spec_hw__bg_hw',
                                             [self.bounds_bg_hw,
                                              self.bounds_spec_hw])

        self.set_spectral_defaults()


    def set_NIRISS_defaults(self):
        '''
        Set Optimizer specific defaults for NIRISS.
        '''
        defaults = {
            "bounds_bg_hw": range(15, 20),
            "bounds_spec_hw": range(17, 25),
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))

        self.set_spectral_defaults()


    def set_WFC3_defaults(self):
        '''
        Set Optimizer specific defaults for WFC3.
        '''
        self.set_spectral_defaults()


    def set_NIRCam_Photometry_defaults(self):
        '''
        Set Optimizer specific defaults for NIRCam Photometry.
        '''
        self.set_photometric_defaults()


    def set_MIRI_Photometry_defaults(self):
        '''
        Set Optimizer specific defaults for MIRI Photometry.
        '''

        self.set_photometric_defaults()
