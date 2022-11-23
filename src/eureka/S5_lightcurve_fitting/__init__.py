"""
Package to fit models to light curve data
"""
# Do some checks here to avoid spamming these warnings
try:
    import theano
    import pymc3
    import starry
    import pymc3_ext
    import arviz
    success = True
except ImportError:
    success = False
    print("Could not import starry and/or pymc3 related packages. "
          "Functionality may be limited.")
if success:
    from . import differentiable_models

from . import fitters
from . import gradient_fitters
from . import lightcurve
from . import likelihood
from . import limb_darkening_fit
from . import modelgrid
from . import models
from . import plots_s5
from . import s5_fit
from . import simulations
from . import utils
