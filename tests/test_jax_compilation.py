"""Executable compilation and derivative-accuracy contracts for Stage 5.

These checks explicitly lower, compile, and execute value-and-gradient pairs.
Finite differences test the derivatives independently of autodiff, at smooth
points. Step locations and Monte Carlo spot-occultation boundaries are not
smooth parameters; their suitability for HMC is not established here. The
symmetric Lorentzian's signed-distance exponent stays fixed at an even
integer, since arbitrary real powers of negative distances are undefined.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.table import Table
from jax.flatten_util import ravel_pytree
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

from eureka.S5_lightcurve_fitting import jax_models as jm
from eureka.S5_lightcurve_fitting.jax_models.JaxModel import \
    make_numpyro_composite_model

from .test_jax_models import SYSTEMATICS, composite, config, orbit, parameters


@pytest.fixture(autouse=True)
def enable_compilation():
    """Compilation coverage must work even if JAX_DISABLE_JIT was set."""
    with jax.disable_jit(False):
        yield


def assert_compiled_gradient(function, initial):
    """Check a compiled executable at two points without Python retracing."""
    initial = jnp.asarray(initial, dtype=jnp.float64)
    traces = []

    def value_and_grad(theta):
        traces.append(None)  # Records tracing, never retains traced values.
        return jax.value_and_grad(function)(theta)

    with jax.checking_leaks():
        executable = jax.jit(value_and_grad).lower(initial).compile()
        for theta in [initial, initial + 0.02 * jnp.maximum(
                jnp.abs(initial), 1e-3)]:
            value, gradient = jax.block_until_ready(executable(theta))
            assert np.isfinite(value)
            assert np.all(np.isfinite(gradient))
            assert np.linalg.norm(gradient) > 1e-10
            np.testing.assert_allclose(value, function(theta), rtol=1e-11)

            differences = []
            for i, coordinate in enumerate(np.asarray(theta)):
                step = 1e-6 * max(1., abs(coordinate))
                offset = jnp.zeros_like(theta).at[i].set(step)
                plus = executable(theta + offset)[0]
                minus = executable(theta - offset)[0]
                differences.append((plus - minus) / (2. * step))
            np.testing.assert_allclose(
                gradient, differences, rtol=3e-5, atol=2e-7)

        # Repeated execution with changed parameters uses the compiled graph.
        assert len(traces) == 1
    return executable


def check_model_gradient(model, values):
    """Avoid cancellation of centered model terms in the scalar objective."""
    names = tuple(values)
    # Do not warm up eval before compilation: exercise its first trace too.
    weights = jnp.linspace(1., 2., sum(model.nints))

    def objective(theta):
        params = dict(zip(names, theta))
        return jnp.vdot(weights, model.eval(param_dict=params))

    assert_compiled_gradient(objective, list(values.values()))


@pytest.mark.parametrize("name,values,free", SYSTEMATICS)
def test_systematic_compiles_with_accurate_gradients(name, values, free):
    # Offset the grid so oscillator cutoffs and asymmetric peaks are not
    # evaluated precisely on their piecewise boundaries.
    time = np.linspace(-0.05, 0.05, 11) + 0.00137
    model = getattr(jm, name)(**config(parameters(**values), time=time))
    smooth = {key: value for key, value in values.items()
              if not key.startswith('steptime')}
    if name == 'LorentzianModel' and 'lor_hwhm' in values:
        smooth.pop('lor_power', None)
    check_model_gradient(model, smooth)


@pytest.mark.parametrize("axis", ["xpos", "ypos", "xwidth", "ywidth"])
def test_centroid_compiles_with_accurate_gradients(axis):
    model = jm.CentroidModel(**config(
        parameters(**{axis: 0.2}), axis=axis,
        centroid=np.linspace(0.1, 0.3, 11)))
    check_model_gradient(model, {axis: 0.2})


def test_common_mode_compiles_with_accurate_gradients(tmp_path):
    path = tmp_path / 'common.ecsv'
    Table({'common': np.linspace(0.9, 1.1, 11)}).write(
        path, format='ascii.ecsv')
    meta = SimpleNamespace(common_mode_file=str(path),
                           common_mode_name='common', verbose=False)
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None)
    model = jm.CommonModeModel(meta, log, **config(
        parameters(cm1=0.02, cm2=0.01)))
    check_model_gradient(model, {'cm1': 0.02, 'cm2': 0.01})


@pytest.mark.parametrize('name,key,value', [
    ('JaxoplanetModel', 'rp', 0.1),
    ('HarmonicaTransitModel', 'rp', 0.1),
    ('JaxoplanetStarryModel', 'fp', 0.002),
    ('FleckTransitModel', 'spotcon', 0.5),
])
def test_orbital_model_compiles_with_accurate_gradients(name, key, value):
    time = np.linspace(-0.043, 0.047, 13)
    pars = orbit(spotrot=1e12, spotstari=90., spotrad=0.05,
                 spotlat=20., spotlon=10., spotcon=0.5)
    model = getattr(jm, name)(**config(
        pars, time=time, nints=[len(time)], ydeg=2))
    check_model_gradient(model, {key: value})


@pytest.mark.parametrize('name,values', [
    ('SinusoidPhaseCurveModel', {'AmpCos1': 0.1, 'AmpSin1': 0.03}),
    ('QuasiLambertianPhaseCurve', {'quasi_gamma': 2., 'quasi_offset': 10.}),
])
def test_phasecurve_compiles_with_accurate_gradients(name, values):
    model = getattr(jm, name)(**config(
        orbit(**values), time=np.linspace(0.2, 0.8, 11)))
    check_model_gradient(model, values)


@pytest.mark.parametrize('gp', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_hmc_potential_compiles_with_accurate_gradients(gp, masked):
    model, time, flux, unc = composite(gp=gp, masked=masked)
    model.parameters.scatter_mult = 1., 'free', 0.5, 2., 'U'
    model.freenames = model.freenames + ['scatter_mult']
    values = {name: getattr(model.parameters, name).value
              for name in model.freenames}
    values['c0'] = 1.003
    function = make_numpyro_composite_model(model, time, flux, unc)
    info = initialize_model(
        jax.random.PRNGKey(17), function, model_args=(time, flux, unc),
        init_strategy=init_to_value(values=values))
    theta, unravel = ravel_pytree(info.param_info.z)

    # This includes the support transforms and log-Jacobian terms used by
    # NUTS, rather than just the log density in constrained parameter space.
    assert_compiled_gradient(
        lambda vector: info.potential_fn(unravel(vector)), theta)


def test_compiled_hmc_gradient_matches_analytic_gaussian_posterior():
    model, time, flux, unc = composite()
    function = make_numpyro_composite_model(model, time, flux, unc)
    info = initialize_model(
        jax.random.PRNGKey(17), function, model_args=(time, flux, unc),
        init_strategy=init_to_value(values={'c0': 1.003}))
    executable = assert_compiled_gradient(
        lambda vector: info.potential_fn({'c0': vector[0]}), [0.4])

    for z in [0.4, 0.6]:
        sigmoid = 1. / (1. + np.exp(-z))
        c0 = 0.8 + 0.4 * sigmoid
        dc0_dz = 0.4 * sigmoid * (1. - sigmoid)
        residual = c0 + 0.02 * time - flux
        # Gaussian likelihood gradient plus the Uniform-support transform's
        # negative log-Jacobian derivative. This is independent of autodiff.
        expected = np.sum(residual / unc**2) * dc0_dz + 2. * sigmoid - 1.
        np.testing.assert_allclose(
            executable(jnp.array([z]))[1][0], expected, rtol=1e-11)
