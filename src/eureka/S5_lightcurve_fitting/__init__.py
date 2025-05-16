"""
Package to fit models to light curve data
"""
# Do some checks here to avoid spamming these warnings
try:
    import os
    import multiprocessing

    # Must come before any import that touches JAX
    ncpu = multiprocessing.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"

    import jax
    jax.config.update("jax_enable_x64", True)
    from . import jax_models
    from . import jax_lightcurve
except ModuleNotFoundError:
    # Don't require that the jax and jaxoplanet packages be installed
    # but also don't raise a warning here to avoid excessive spam
    pass

from . import fitters
# from . import gradient_fitters
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
