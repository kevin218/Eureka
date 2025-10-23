import os
import logging
logger = logging.getLogger(__name__)

try:
    from .version import __version__
except ModuleNotFoundError:
    from setuptools_scm import get_version
    __version__ = get_version(root=f'..{os.sep}..{os.sep}',
                              relative_to=__file__)

from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

from . import lib
try:
    import jwst
    success = True
except ModuleNotFoundError:
    logger.warning("The package jwst has not been installed. As a result, "
                   "Eureka!'s Stages 1 and 2 will not work.")
    success = False
if success:
    from . import S1_detector_processing
    from . import S2_calibrations
from . import S3_data_reduction
from . import S4_generate_lightcurves
from . import S4cal_StellarSpectra
from . import S5_lightcurve_fitting
from . import S6_planet_spectra

__all__ = ["lib", "S3_data_reduction", "S4_generate_lightcurves",
           "S4cal_StellarSpectra", "S5_lightcurve_fitting",
           "S6_planet_spectra"]

if success:
    __all__.extend(["S1_detector_processing", "S2_calibrations"])
