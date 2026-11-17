"""Configure the shared JAX backend before importing Stage 5 models."""
import multiprocessing
import os

# JAX is required by the model implementations used by both fitter families.
# XLA reads this option when initializing the backend.
ncpu = multiprocessing.cpu_count()
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
