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
        defaults = {
            "inputdir": 'Stage2',
            "outputdir": 'Optimizer',
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))


    def set_spectral_defaults(self):
        '''Set Optimizer specific defaults for generic spectroscopic data.
        '''
        # Spectral extraction parameters
        defaults = {
            "bounds_dqmask": [True, False],  # not a range
            "bounds_bg_thresh": range(3, 8),  # 3 → 7
            "bounds_bg_method": ["std", "mean", "median"],
            "bounds_p3thresh": range(3, 8),  # 3 → 7
            "bounds_median_thresh": range(3, 8),  # 3 → 7
            "bounds_window_len": range(1, 22, 2),  # 1 → 21, odd only
            "bounds_p7thresh": range(5, 96, 5),  # 5 → 95 step 5
            # "bounds_expand": range(1, 6),  # uncomment if needed, 1 → 5
            "bounds_sigma": range(3, 8),  # 3 → 7
            "bounds_box_width": range(10, 51, 10),  # 10, 20, 30, 40, 50
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))


    def set_photometric_defaults(self):
        '''Set Optimizer specific defaults for generic photometric data.
        '''
        return


    def set_MIRI_defaults(self):
        '''Set Optimizer specific defaults for MIRI.
        '''
        defaults = {
            "bounds_bg_hw": range(5, 16),
            "bounds_spec_hw": range(1, 10),
            "bounds_bg_deg": range(-1, 2),  # -1 → 1
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))
        self.bounds_bg_hw__spec_hw = getattr(self, 'bounds_bg_hw__spec_hw',
                                             [self.bounds_bg_hw,
                                              self.bounds_spec_hw])

        self.set_spectral_defaults()


    def set_NIRCam_defaults(self):
        '''Set Optimizer specific defaults for NIRCam.
        '''
        defaults = {
            "bounds_bg_hw": range(5, 16),
            "bounds_spec_hw": range(1, 10),
            "bounds_bg_deg": range(-1, 2),  # -1 → 1
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))
        self.bounds_bg_hw__spec_hw = getattr(self, 'bounds_bg_hw__spec_hw',
                                             [self.bounds_bg_hw,
                                              self.bounds_spec_hw])

        self.set_spectral_defaults()

    def set_NIRSpec_defaults(self):
        '''Set Optimizer specific defaults for NIRSpec.
        '''
        defaults = {
            "bounds_bg_hw": range(5, 16),
            "bounds_spec_hw": range(1, 10),
            "bounds_bg_deg": range(-1, 2),  # -1 → 1
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))
        self.bounds_bg_hw__spec_hw = getattr(self, 'bounds_bg_hw__spec_hw',
                                             [self.bounds_bg_hw,
                                              self.bounds_spec_hw])

        self.set_spectral_defaults()

    def set_NIRISS_defaults(self):
        '''Set Optimizer specific defaults for NIRISS.
        '''
        defaults = {
            "bounds_bg_hw": range(15, 20),
            "bounds_spec_hw": range(17, 25),
            "bounds_bg_deg": range(0, 2),  # 0 → 1
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))

        self.set_spectral_defaults()

    def set_WFC3_defaults(self):
        '''Set Optimizer specific defaults for WFC3.
        '''
        self.set_spectral_defaults()

    def set_NIRCam_Photometry_defaults(self):
        '''Set Optimizer specific defaults for NIRCam Photometry.
        '''
        self.set_photometric_defaults()

    def set_MIRI_Photometry_defaults(self):
        '''Set Optimizer specific defaults for MIRI Photometry.
        '''

        self.set_photometric_defaults()
