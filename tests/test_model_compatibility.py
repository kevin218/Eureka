"""Public OO import paths share JAX kernels and still support mixed fits."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eureka.S5_lightcurve_fitting import jax_models as jm, models
from eureka.S5_lightcurve_fitting.jax_models.JaxModel import (
    make_numpyro_composite_model,
)
from .test_jax_models import SYSTEMATICS, config, orbit, parameters


@pytest.mark.parametrize("name,values,free", SYSTEMATICS)
def test_public_model_exports_and_evaluation(name, values, free):
    public_class = getattr(models, name)
    assert public_class is getattr(jm, name)

    model = public_class(**config(parameters(**values)))
    canonical = getattr(jm, name)(**config(parameters(**values)))
    np.testing.assert_allclose(
        model.eval(channel=0), canonical.eval(channel=0)
    )
    np.testing.assert_allclose(
        jax.jit(lambda v: model.eval(channel=0, param_dict={free: v}))(
            values[free]
        ),
        canonical.eval(),
    )
    assert np.isfinite(
        jax.grad(
            lambda v: jnp.sum(model.eval(channel=0, param_dict={free: v}))
        )(values[free])
    )
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is public_class
    np.testing.assert_allclose(restored.eval(channel=0), model.eval(channel=0))
    model.freenames = [free]
    model.update([1.1 * values[free]])
    np.testing.assert_allclose(
        model.eval(channel=0),
        canonical.eval(param_dict={free: 1.1 * values[free]}),
    )


@pytest.mark.parametrize(
    "name", ["SinusoidPhaseCurveModel", "QuasiLambertianPhaseCurve"]
)
def test_public_phasecurve_exports_and_nested_jax(name):
    public_class = getattr(models, name)
    assert public_class is getattr(jm, name)
    cfg = config(
        orbit(
            AmpCos1=0.1,
            quasi_gamma=2.0,
            fp_pl1=0.003,
            AmpCos1_pl1=0.2,
            quasi_gamma_pl1=3.0,
        ),
        num_planets=2,
        name="custom name",
    )
    phase = public_class(**cfg)
    canonical = getattr(jm, name)(**cfg)
    np.testing.assert_allclose(
        phase.eval(channel=0, pid=1), canonical.eval(channel=0, pid=1)
    )
    astro = jm.AstroModel([phase], **cfg)
    expected = (
        1.0 + 0.002 * canonical.eval(pid=0) + 0.003 * canonical.eval(pid=1)
    )
    np.testing.assert_allclose(jax.jit(lambda: astro.eval())(), expected)


@pytest.mark.parametrize("axis", ["xpos", "ypos", "xwidth", "ywidth"])
def test_public_centroid_evaluation(axis):
    data = np.linspace(0.1, 0.3, 11)
    model = models.CentroidModel(
        **config(parameters(**{axis: 0.2}), axis=axis, centroid=data)
    )
    np.testing.assert_allclose(
        model.eval(channel=0), 1.0 + 0.2 * (data - data.mean())
    )


def test_public_common_mode_evaluation(tmp_path):
    from astropy.table import Table
    from types import SimpleNamespace
    from eureka.lib.logedit import Logedit

    series = np.linspace(1.0, 1.1, 11)
    path = tmp_path / "common.ecsv"
    Table({"common": series}).write(path, format="ascii.ecsv")
    meta = SimpleNamespace(
        common_mode_file=str(path), common_mode_name="common", verbose=False
    )
    model = models.CommonModeModel(
        meta,
        Logedit(str(tmp_path / "fit.log")),
        **config(parameters(cm1=0.1, cm2=0.2)),
    )
    x = series - series.mean()
    np.testing.assert_allclose(
        model.eval(channel=0), 1.0 + 0.1 * x + 0.2 * x**2
    )
    assert type(model) is jm.CommonModeModel


@pytest.mark.parametrize("jax_first", [True, False])
def test_mixed_public_models_work_but_are_rejected_by_hmc(jax_first):
    pars = orbit(c0=1.01, c1=0.02)
    cfg = config(pars)
    transit = models.BatmanTransitModel(**cfg)
    trend = models.PolynomialModel(**cfg)
    expected = np.asarray(transit.eval()) * np.asarray(trend.eval())
    mixed = trend * transit if jax_first else transit * trend
    mixed.time = cfg["time"]
    np.testing.assert_allclose(mixed.eval(), expected)
    assert not isinstance(mixed, jm.CompositeJaxModel)
    with pytest.raises(TypeError, match="not JAX-compatible"):
        make_numpyro_composite_model(mixed)


@pytest.mark.parametrize(
    "kernel_name", ["Matern32", "ExpSquared", "RationalQuadratic", "Exp"]
)
def test_tinygp_kernel_is_shared_by_both_fitters(kernel_name):
    from types import SimpleNamespace
    import tinygp
    from eureka.S5_lightcurve_fitting.jax_models.GPModel import (
        build_tinygp_kernel,
    )

    pars = parameters(A=-12.0, m=-2.0)
    time = np.linspace(0.0, 1.0, 11)
    lc = SimpleNamespace(
        flux=np.ones(11), unc=np.full(11, 0.001), unc_fit=np.full(11, 0.001)
    )
    legacy = models.GPModel(
        [kernel_name],
        ["time"],
        lc,
        gp_code_name="tinygp",
        **config(pars, time=time),
    )
    kernel = legacy.get_kernel(kernel_name, 0)
    kwargs = {"alpha": 1.0} if kernel_name == "RationalQuadratic" else {}
    reference = np.exp(-12.0) * getattr(tinygp.kernels, kernel_name)(
        scale=np.exp(-4.0), **kwargs
    )
    np.testing.assert_allclose(
        kernel.evaluate(0.0, 0.02), reference.evaluate(0.0, 0.02)
    )
    result = jax.jit(
        lambda amp: build_tinygp_kernel(kernel_name, amp, -2.0).evaluate(
            0.0, 0.02
        )
    )(-12.0)
    np.testing.assert_allclose(result, reference.evaluate(0.0, 0.02))


def test_mixed_astrophysical_adapter_with_shared_phasecurve():
    pars = orbit(AmpCos1=0.1, AmpSin1=0.02)
    cfg = config(pars, time=np.linspace(0.2, 0.8, 31), compute_ltt=False)
    transit = models.BatmanTransitModel(**cfg)
    eclipse = models.BatmanEclipseModel(**cfg)
    phase = models.SinusoidPhaseCurveModel(**cfg)
    model = models.AstroModel([transit, eclipse, phase], **cfg)
    expected = np.asarray(transit.eval()) + np.asarray(
        eclipse.eval()
    ) * np.asarray(phase.eval(channel=0, pid=0))
    np.testing.assert_allclose(model.eval(), expected)
