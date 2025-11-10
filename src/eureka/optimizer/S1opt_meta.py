from ..lib.readECF import MetaClass
from ..lib import util


class S1optMetaClass(MetaClass):
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

        super().__init__(folder, file, eventlabel, stage='1opt', **kwargs)

        # Set default optimizer and target fitness score
        self.optimizer = getattr(self, 'optimizer', 'parametric')
        self.target_fitness = getattr(self, 'target_fitness', 0.0)

        # Set a default data file suffix
        self.suffix = getattr(self, 'suffix', 'uncal')

        # Set optimization flag
        self.isopt_S1 = getattr(self, 'isopt_S1', True)
        self.isopt_S3 = getattr(self, 'isopt_S3', False)

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
            "inputdir": 'Stage0',
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
            # Stage 1
            "bounds_jump_rejection_threshold": range(4, 15),    # 4 → 14
            "bounds_bg_method": ["std", "mean", "median"],
            "bounds_bg_deg": range(0, 2),  # 0 → 1
            "bounds_p3thresh": range(3, 8),  # 3 → 7
            "bounds_window_len": range(1, 22, 2),  # 1 → 21, odd only
            "bounds_p7thresh": range(5, 61, 5),  # 5 → 60, step 5
            "bounds_expand_mask": range(5, 12),  # 5 → 11
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))

    def set_NIRSpec_defaults(self):
        '''
        Set Optimizer specific defaults for NIRSpec.
        '''
        # Fitness scaling factors
        self.scaling_MAD_spec = getattr(self, 'scaling_MAD_spec', 0.01)
        self.scaling_MAD_white = getattr(self, 'scaling_MAD_white', 1.0)

        defaults = {
            "bounds_bg_hw": range(5, 16),
        }

        for key, default in defaults.items():
            setattr(self, key, getattr(self, key, default))

        self.set_spectral_defaults()
