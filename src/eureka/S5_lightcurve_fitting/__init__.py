"""
Package to fit models to light curve data
"""
# Do some checks here to avoid spamming these warnings
try:
    from . import differentiable_models
except ModuleNotFoundError:
    # Don't require that the pymc3, starry, and theano packages be installed
    # but also don't raise a warning here to avoid excessive spam
    pass

from . import fitters
from . import gradient_fitters
from . import lightcurve
from . import likelihood
from . import limb_darkening_fit
from . import modelgrid
from . import models
from . import plots_s5
from . import s5_fit
from . import s5_meta
from . import simulations
from . import utils
