"""Stage 5 regression coverage for pure JAX kernels and their OO adapters."""

from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from eureka.lib.readEPF import Parameters
from eureka.S5_lightcurve_fitting import jax_models as jm, models
from eureka.S5_lightcurve_fitting.jax_models.JaxModel import (
    build_scatter_array_jax,
    make_numpyro_composite_model,
)
from eureka.S5_lightcurve_fitting.jax_models.AstroModel import (
    compute_astroparams,
)


def parameters(**values):
    pars = Parameters()
    for key, value in values.items():
        setattr(
            pars,
            key,
            (value, "independent" if isinstance(value, str) else "fixed"),
        )
    return pars


def config(pars, **extra):
    result = dict(
        parameters=pars,
        time=np.linspace(-0.05, 0.05, 11),
        nints=[11],
        freenames=[],
        num_planets=1,
        paramtitles=list(pars.dict),
    )
    result.update(extra)
    return result


def orbit(**extra):
    result = dict(
        rp=0.1,
        per=1.0,
        t0=0.0,
        inc=87.0,
        a=8.0,
        ecc=0.0,
        w=90.0,
        Rs=1.0,
        limb_dark="quadratic",
        u1=0.2,
        u2=0.1,
        fp=0.002,
    )
    result.update(extra)
    return parameters(**result)


SYSTEMATICS = [
    ("PolynomialModel", dict(c0=1.02, c1=0.02, c2=0.001), "c1"),
    ("ExpRampModel", dict(r0=0.02, r1=2.0), "r0"),
    ("ExpRampModel", dict(r0=0.02, r1=2.0, r2=-0.01, r3=4.0), "r2"),
    (
        "HSTRampModel",
        dict(h0=0.01, h1=2.0, h2=0.01, h3=0.02, h4=0.04, h5=0.001),
        "h0",
    ),
    ("StepModel", dict(step0=0.01, steptime0=0.0), "step0"),
    (
        "DampedOscillatorModel",
        dict(
            osc_amp=0.02,
            osc_amp_decay=0.1,
            osc_per=0.04,
            osc_per_decay=0.1,
            osc_t0=-0.03,
            osc_t1=0.01,
        ),
        "osc_amp",
    ),
    (
        "LorentzianModel",
        dict(lor_amp=0.02, lor_hwhm=0.02, lor_t0=0.001, lor_power=2.0),
        "lor_amp",
    ),
    (
        "LorentzianModel",
        dict(lor_amp=0.02, lor_hwhm_lhs=0.02, lor_hwhm_rhs=0.03, lor_t0=0.001),
        "lor_amp",
    ),
    (
        "LorentzianModel",
        dict(
            lor_amp_lhs=0.02,
            lor_amp_rhs=0.01,
            lor_hwhm_lhs=0.02,
            lor_hwhm_rhs=0.03,
            lor_t0=0.001,
        ),
        "lor_amp_lhs",
    ),
]


def systematic_expectation(name, values, time):
    """Independent NumPy expectations for the fixed regression cases."""
    v = values
    local = time - time[0]
    if name == "PolynomialModel":
        t = time - np.mean(time)
        return np.polynomial.polynomial.polyval(t, [1.02, 0.02, 0.001])
    if name == "ExpRampModel":
        return (
            1.0
            + v["r0"] * np.exp(-v["r1"] * local)
            + v.get("r2", 0.0) * np.exp(-v.get("r3", 0.0) * local)
        )
    if name == "HSTRampModel":
        t = (local - v["h5"]) % v["h4"]
        return 1.0 + 0.01 * np.exp(-2.0 * t) + 0.01 * t + 0.02 * t**2
    if name == "StepModel":
        return 1.0 + 0.01 * (local >= 0.0)
    if name == "DampedOscillatorModel":
        amp = 0.02 * np.exp(-0.1 * (time + 0.03))
        period = 0.04 * np.exp(-0.1 * (time + 0.03))
        signal = 1.0 + amp * np.sin(2.0 * np.pi * (time - 0.01) / period)
        return np.where(time < -0.03, 1.0, signal)
    if "lor_hwhm" in v:
        return 1.0 + 0.02 / (1.0 + (2.0 * (time - 0.001) / 0.02) ** 2)
    width = np.where(time <= 0.001, 0.02, 0.03)
    shape = 1.0 / (1.0 + ((time - 0.001) / width) ** 2)
    if "lor_amp" in v:
        return 1.0 + 0.02 * shape
    return np.where(time <= 0.001, 1.0 + 0.02 * shape, 1.01 + 0.01 * shape)


@pytest.mark.parametrize("name,values,free", SYSTEMATICS)
@pytest.mark.parametrize("multwhite", [False, True])
def test_systematic_jit_grad_and_analytic_values(
    name, values, free, multwhite
):
    pars = parameters(**values)
    times = np.linspace(-0.05, 0.05, 11)
    nints = [11, 7] if multwhite else [11, 11]
    time = np.r_[times, np.linspace(0.2, 0.3, 7)] if multwhite else times
    cfg = config(
        pars,
        time=time,
        nints=nints,
        multwhite=multwhite,
        nchannel_fitted=2,
        fitted_channels=[0, 1],
    )
    model = getattr(jm, name)(**cfg)
    times_by_channel = [time[:11], time[11:]] if multwhite else [time, time]
    expected = np.concatenate(
        [systematic_expectation(name, values, t) for t in times_by_channel]
    )
    assert model.nchannel_fitted == 2
    np.testing.assert_allclose(model.eval(), expected, atol=1e-12)
    original_time = np.asarray(model.time).copy()

    def evaluate(x, t):
        return model.eval(param_dict={free: x}, time=t)

    value = values[free]
    np.testing.assert_allclose(
        jax.jit(evaluate)(value, jnp.asarray(time)), model.eval(), atol=1e-12
    )
    grad = jax.grad(lambda x: jnp.sum(evaluate(x, jnp.asarray(time))))(value)
    assert np.isfinite(grad)
    np.testing.assert_array_equal(model.time, original_time)
    # Re-evaluate after tracing with changed data and parameters, detecting
    # leaked tracers and cached compiled constants on the wrapper.
    changed = jax.jit(evaluate)(value * 1.1, jnp.asarray(time + 0.003))
    np.testing.assert_allclose(changed, evaluate(value * 1.1, time + 0.003))
    for ch in (0, 1):
        start, stop = (0, 11) if ch == 0 else (11, sum(nints))
        np.testing.assert_allclose(
            model.eval(channel=ch), np.asarray(model.eval())[start:stop]
        )


def test_polynomial_sparse_and_wavelength_shared():
    pars = parameters(c0=1.0, c2=0.01, c0_ch1=2.0, c0_wl1=3.0)
    model = jm.PolynomialModel(
        **config(
            pars,
            nchannel_fitted=2,
            fitted_channels=[0, 1],
            wl_groups=[0, 1],
            nints=[11, 11],
        )
    )
    expected = np.r_[1.0 + 0.01 * model.time**2, 3.0 + 0.01 * model.time**2]
    np.testing.assert_allclose(model.eval(), expected)
    derivative = jax.grad(lambda v: jnp.sum(model.eval({"c0_wl1": v})))(3.0)
    assert derivative == 11


@pytest.mark.parametrize("channel", [0, 3])
def test_single_channel_scatter_and_shared_fallback(channel):
    unc = jnp.array([1.0, 2.0, 3.0]) * 1e-4

    def fn(p):
        return build_scatter_array_jax(
            unc, jnp.arange(3.0), [3], [channel], 1, False, p
        )

    np.testing.assert_allclose(jax.jit(fn)({"scatter_mult": 2.0}), 2.0 * unc)
    np.testing.assert_allclose(
        fn(
            {
                (
                    "scatter_ppm"
                    if channel == 0
                    else f"scatter_ppm_ch{channel}"
                ): 40.0
            }
        ),
        4e-5,
    )
    np.testing.assert_allclose(fn({}), unc)


@pytest.mark.parametrize("multwhite", [False, True])
def test_multichannel_scatter(multwhite):
    time = jnp.arange(5.0 if multwhite else 3.0)
    unc = jnp.arange(1.0, 6.0 if multwhite else 7.0) * 1e-4
    nints = [3, 2 if multwhite else 3]
    result = build_scatter_array_jax(
        unc,
        time,
        nints,
        [0, 1],
        2,
        multwhite,
        {"scatter_mult": 2.0, "scatter_mult_ch1": 3.0},
    )
    np.testing.assert_allclose(result, np.r_[2.0 * unc[:3], 3.0 * unc[3:]])


def test_numeric_configuration_is_preserved():
    pars = orbit(limb_dark="kipping2013", u1=0.25, u2=0.4)
    model = jm.JaxoplanetModel(**config(pars))
    astro = compute_astroparams(model._prepare_param_dict())
    np.testing.assert_allclose(astro["u"], [0.4, 0.1])
    np.testing.assert_allclose(
        model.eval(), jax.jit(lambda r: model.eval({"rp": r}))(0.1)
    )
    assert np.isfinite(jax.grad(lambda r: jnp.sum(model.eval({"rp": r})))(0.1))


def test_transit_multiple_planets_and_batman_parity():
    pars = orbit(rp_pl1=0.05, per_pl1=2.0, t0_pl1=0.02, a_pl1=12.0)
    cfg = config(pars, num_planets=2)
    transit = jm.JaxoplanetModel(**cfg)
    result = jax.jit(lambda r: transit.eval({"rp_pl1": r}))(0.05)
    assert result.shape == (11,)
    # The two planets' independent transit deficits add in jaxoplanet.
    first = jm.JaxoplanetModel(**config(orbit())).eval()
    second = jm.JaxoplanetModel(
        **config(orbit(rp=0.05, per=2.0, t0=0.02, a=12.0))
    ).eval()
    np.testing.assert_allclose(result, first + second - 1.0, atol=1e-10)
    batman = models.BatmanTransitModel(**config(orbit()))
    np.testing.assert_allclose(first, batman.eval(), atol=2e-7)


@pytest.mark.parametrize(
    "name,key",
    [
        ("SinusoidPhaseCurveModel", "AmpCos1"),
        ("QuasiLambertianPhaseCurve", "quasi_gamma"),
    ],
)
def test_phase_curves_jit_planet_selection(name, key):
    pars = orbit(
        AmpCos1=0.1,
        AmpCos1_pl1=0.2,
        quasi_gamma=2.0,
        quasi_gamma_pl1=3.0,
        rp_pl1=0.05,
        t0_pl1=0.02,
        per_pl1=2.0,
    )
    phase = getattr(jm, name)(**config(pars, num_planets=2))
    np.testing.assert_allclose(
        phase.eval(), phase.eval(pid=0) * phase.eval(pid=1)
    )
    assert np.isfinite(jax.grad(lambda v: jnp.sum(phase.eval({key: v})))(0.1))
    np.testing.assert_allclose(
        jax.jit(lambda v: phase.eval({key: v}))(0.2), phase.eval({key: 0.2})
    )


def composite(gp=False, masked=False):
    time = np.linspace(-0.05, 0.05, 11)
    flux = 1.01 + 0.02 * time
    unc = np.full(11, 0.001)
    if masked:
        time[1] = np.nan
        flux[3] = np.nan
        unc[4] = np.nan
    pars = parameters(c0=1.0, c1=0.02, scatter_mult=1.0)
    pars.c0 = 1.0, "free", 0.8, 1.2, "U"
    components = [jm.PolynomialModel(**config(pars, time=time))]
    if gp:
        pars.A = -12.0, "free", -20.0, -5.0, "U"
        pars.m = -2.0, "free", -5.0, 0.0, "U"
        lc = SimpleNamespace(flux=flux, unc=unc, unc_fit=unc)
        components.append(
            jm.GPModel(["Matern32"], ["time"], lc, **config(pars, time=time))
        )
    model = jm.CompositeJaxModel(
        components, **config(pars, time=time), flux=flux, lc_unc=unc
    )
    model.freenames = ["c0", "A", "m"] if gp else ["c0"]
    return model, time, flux, unc


@pytest.mark.parametrize("gp", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_numpyro_likelihood_jit_grad_is_pure(gp, masked):
    model, time, flux, unc = composite(gp, masked)
    before = deepcopy(model)
    numpyro_model = make_numpyro_composite_model(model, time, flux, unc)
    sampled = {"c0": 1.01}
    if gp:
        sampled.update(A=-12.0, m=-2.0)

    def density(values):
        return log_density(numpyro_model, (time, flux, unc), {}, values)[0]

    result = jax.jit(density)(sampled)
    assert np.isfinite(result)
    assert all(np.isfinite(v) for v in jax.grad(density)(sampled).values())
    np.testing.assert_array_equal(model.time, before.time)
    np.testing.assert_allclose(model.eval(), before.eval(), equal_nan=True)
    assert not hasattr(model, "scatter_array")
    assert model.parameters.c0.value == before.parameters.c0.value
    # A second transformation must not encounter tracers saved by the first.
    assert np.isfinite(jax.jit(density)({**sampled, "c0": 1.02}))
    if gp:
        assert density({**sampled, "A": -10.0}) != result


def test_mixed_object_oriented_composite():
    pars = orbit(c0=1.01, c1=0.02)
    cfg = config(pars)
    transit = models.BatmanTransitModel(**cfg)
    trend = jm.PolynomialModel(**cfg)
    mixed = models.CompositeModel([transit, trend], **cfg)
    np.testing.assert_allclose(
        mixed.eval(), transit.eval() * np.asarray(trend.eval())
    )
    new_time = np.linspace(-0.04, 0.04, 9)
    expected = models.BatmanTransitModel(**config(pars, time=new_time)).eval()
    expected *= np.asarray(trend.eval(time=new_time))
    np.testing.assert_allclose(mixed.interp(new_time, [9]), expected)
    assert len(mixed.time) == 11


@pytest.mark.parametrize(
    "name",
    [
        "JaxoplanetModel",
        "JaxoplanetStarryModel",
        "FleckTransitModel",
        "HarmonicaTransitModel",
    ],
)
def test_orbital_wrappers_jit_grad_and_astro_composition(name):
    pars = orbit(
        spotrot=1e12,
        spotstari=90.0,
        spotrad=0.05,
        spotlat=20.0,
        spotlon=10.0,
        spotcon=0.5,
    )
    cfg = config(pars, ydeg=2)
    model = getattr(jm, name)(**cfg)

    def evaluate(r):
        return model.eval({"rp": r})

    expected = model.eval()
    np.testing.assert_allclose(jax.jit(evaluate)(0.1), expected, atol=1e-10)
    assert np.isfinite(jax.grad(lambda r: jnp.sum(evaluate(r)))(0.1))
    astro = jm.AstroModel([model], **cfg)
    np.testing.assert_allclose(
        jax.jit(lambda r: astro.eval({"rp": r}))(0.1), expected, atol=1e-10
    )


def test_starry_zero_flux_is_differentiable_and_maps_affect_lightcurve():
    pars = orbit(fp=0.0, Y10=0.1, Y11=0.2, **{"Y1-1": -0.1})
    cfg = config(pars, time=np.linspace(0.2, 0.8, 21), ydeg=1)
    model = jm.JaxoplanetStarryModel(**cfg)
    evaluate = jax.jit(lambda fp: model.eval({"fp": fp}))
    assert np.all(np.isfinite(evaluate(0.0)))
    assert np.isfinite(jax.grad(lambda fp: jnp.sum(evaluate(fp)))(0.0))
    assert np.any(np.asarray(evaluate(0.002)) != 1.0)
    pieces = model.eval({"fp": 0.002}, piecewise=True)[0]
    np.testing.assert_allclose(evaluate(0.002), sum(pieces))


def test_astrophysical_phasecurve_two_planets():
    pars = orbit(
        fp_pl1=0.003,
        rp_pl1=0.05,
        per_pl1=2.0,
        t0_pl1=0.02,
        AmpCos1=0.1,
        AmpCos1_pl1=0.2,
    )
    cfg = config(pars, num_planets=2)
    phase = jm.SinusoidPhaseCurveModel(**cfg)
    astro = jm.AstroModel([phase], **cfg)
    expected = 1.0 + 0.002 * phase.eval(pid=0) + 0.003 * phase.eval(pid=1)
    np.testing.assert_allclose(astro.eval(), expected)
    np.testing.assert_allclose(
        jax.jit(lambda a: astro.eval({"AmpCos1": a}))(0.1), expected
    )


@pytest.mark.parametrize(
    "name", ["PolynomialModel", "ExpRampModel", "CentroidModel"]
)
def test_nonzero_channel_and_explicit_data(name):
    pars = parameters(
        c0=1.0,
        c1=0.01,
        r0=0.01,
        r1=2.0,
        xpos=0.01,
        c0_ch3=1.1,
        r0_ch3=0.02,
        xpos_ch3=0.02,
    )
    data = np.linspace(0.0, 0.1, 11)
    cfg = config(pars, fitted_channels=[3], centroid=data, axis="xpos")
    model = getattr(jm, name)(**cfg)
    assert model.nchannel_fitted == 1
    np.testing.assert_allclose(model.eval(), model.eval(channel=3))
    assert np.isfinite(
        jax.grad(lambda v: jnp.sum(model.eval({"c0_ch3": v})))(1.1)
    )


@pytest.mark.parametrize("fitter", ["jaxopt", "nuts"])
@pytest.mark.parametrize("gp", [False, True])
def test_fitters_save_and_reuse_model(tmp_path, fitter, gp):
    from eureka.S5_lightcurve_fitting import jax_fitters, jax_lightcurve
    from eureka.lib.logedit import Logedit

    model, time, flux, unc = composite(gp=gp)
    model.freenames = ["c0"]
    model.parameters.A = -12.0, "fixed"
    model.parameters.m = -2.0, "fixed"
    log = Logedit(str(tmp_path / "fit.log"))
    lc = jax_lightcurve.LightCurve(
        time,
        flux,
        0,
        1,
        log,
        [list(model.parameters.dict)],
        model.parameters,
        ["c0"],
        unc=unc,
        nints=[11],
    )
    meta = SimpleNamespace(
        old_fitparams=None,
        verbose=False,
        isplots_S5=0,
        run_myfuncs=["polynomial"],
        lsq_method="BFGS",
        lsq_tol=1e-8,
        lsq_maxiter=100,
        pixelsampling=False,
        nints=[11],
        outputdir=str(tmp_path) + "/",
        eventlabel="synthetic",
        expand=1,
        spec_hw=1,
        bg_hw=2,
        wave_low=np.array([1.0]),
        wave_hi=np.array([2.0]),
        multwhite=False,
        jaxopt_first=False,
        ncpu=1,
        chains=1,
        dense_mass=False,
        run_nburn=100,
        run_nsteps=100,
        run_nwalkers=4,
    )
    function = getattr(jax_fitters, fitter + "fitter")
    result = function(lc, model, meta, log)
    np.testing.assert_allclose(result.fit_params[0], 1.01, atol=0.002)
    assert np.isfinite(result.chi2red)
    assert list(tmp_path.glob(f"S5_{fitter}_fitparams*.csv"))
    assert list(tmp_path.glob("S5_*Table_Save*.txt"))
    if fitter == "nuts":
        assert list(tmp_path.glob("S5_nuts_samples*.h5"))
    # This failed in the old implementation after optimizer/sampler tracing.
    assert np.all(np.isfinite(result.eval(incl_GP=True)))
    assert result.physeval(interp=True)[0].shape == flux.shape


@pytest.mark.parametrize("ecc", [0.0, 0.2])
@pytest.mark.parametrize(
    "name", ["SinusoidPhaseCurveModel", "QuasiLambertianPhaseCurve"]
)
def test_phasecurve_keplerian_expectations_for_eccentric_orbits(name, ecc):
    pars = orbit(
        ecc=ecc,
        w=45.0,
        AmpCos1=0.1,
        AmpSin1=0.02,
        AmpCos2=0.01,
        AmpSin2=-0.02,
        quasi_gamma=2.5,
        quasi_offset=10.0,
    )
    cfg = config(
        pars, time=np.linspace(-0.05, 0.95, 31), force_positivity=True
    )
    model = getattr(jm, name)(**cfg)
    from eureka.S5_lightcurve_fitting.models.AstroModel import (
        PlanetParams,
        true_anomaly,
        get_ecl_midpt,
    )

    astro = PlanetParams(model)
    if ecc == 0.0:
        phi = 2.0 * np.pi * (cfg["time"] - get_ecl_midpt(astro)) / astro.per
    else:
        phi = true_anomaly(astro, cfg["time"]) + np.pi / 4.0 + np.pi / 2.0
    if name == "SinusoidPhaseCurveModel":
        expected = (
            1.0
            + 0.1 * (np.cos(phi) - 1.0)
            + 0.02 * np.sin(phi)
            + 0.01 * (np.cos(2.0 * phi) - 1.0)
            - 0.02 * np.sin(2.0 * phi)
        )
    else:
        expected = np.abs(np.cos((phi + np.deg2rad(10.0)) / 2.0)) ** 2.5
    np.testing.assert_allclose(model.eval(), expected, atol=2e-9)


def test_wavelength_alias_does_not_change_global_fallback():
    pars = parameters(c0=1.0, c0_wl1=3.0)
    model = jm.PolynomialModel(
        **config(
            pars,
            nchannel_fitted=2,
            fitted_channels=[0, 1],
            wl_groups=[1, 0],
            nints=[11, 11],
        )
    )
    np.testing.assert_allclose(
        model.eval(), np.r_[np.full(11, 3.0), np.ones(11)]
    )


def test_static_configuration_not_sampled_when_unused():
    model, time, flux, unc = composite()
    model.parameters.unused = 0.5, "free", 0.0, 1.0, "U"
    function = make_numpyro_composite_model(model)
    _, trace = log_density(function, (time, flux, unc), {}, {"c0": 1.01})
    assert "unused" not in trace


def test_log_uniform_bounds_use_natural_log_units():
    model, time, flux, unc = composite()
    model.parameters.c0 = 1.0, "free", np.log(0.8), np.log(1.2), "LU"
    function = make_numpyro_composite_model(model)
    _, trace = log_density(function, (time, flux, unc), {}, {"c0": 1.0})
    distribution = trace["c0"]["fn"]
    np.testing.assert_allclose(distribution.low, 0.8)
    np.testing.assert_allclose(distribution.high, 1.2)


@pytest.mark.parametrize("gp", [False, True])
def test_numpy_and_numpyro_likelihood_agree_with_missing_data(gp):
    from eureka.S5_lightcurve_fitting.likelihood import ln_like

    model, time, flux, unc = composite(gp, masked=True)
    names = model.freenames
    values = {key: getattr(model.parameters, key).value for key in names}
    function = make_numpyro_composite_model(model, time, flux, unc)
    _, trace = log_density(function, (time, flux, unc), {}, values)
    observed_log_prob = sum(
        jnp.sum(site["fn"].log_prob(site["value"]))
        for site in trace.values()
        if site["type"] == "sample" and site["is_observed"]
    )
    lc = SimpleNamespace(
        time=time,
        flux=flux,
        unc=unc,
        unc_fit=unc,
        nints=[11],
        multwhite=False,
        nchannel_fitted=1,
    )
    expected = ln_like(list(values.values()), lc, model, names)
    np.testing.assert_allclose(observed_log_prob, expected, atol=1e-10)


@pytest.mark.parametrize(
    "fitter,transit",
    [
        ("lsq", "jaxoplanet"),
        ("jaxopt", "jaxoplanet"),
        ("nuts", "jaxoplanet"),
        ("lsq", "batman_tr"),
    ],
)
def test_stage5_fit_channel_dispatch_and_plot(tmp_path, fitter, transit):
    from eureka.S5_lightcurve_fitting import s5_fit
    from eureka.S5_lightcurve_fitting.s5_meta import S5MetaClass
    from eureka.lib.logedit import Logedit
    from astropy.table import Table

    pars = orbit(c0=1.0, c1=0.02)
    pars.c0 = 1.0, "free", 0.9, 1.1, "U"
    time = np.linspace(-0.05, 0.05, 21)
    truth = jm.JaxoplanetModel(**config(pars, time=time)).eval()
    flux = np.ma.asarray(np.asarray(truth) * (1.01 + 0.02 * time))
    unc = np.ma.asarray(np.full(len(time), 0.001))
    meta = S5MetaClass(
        spec_hw=1,
        bg_hw=2,
        fit_par="",
        fit_method=[fitter],
        eventlabel="synthetic",
        run_myfuncs=[transit, "polynomial"],
        topdir=str(tmp_path),
        inputdir="",
        outputdir=str(tmp_path) + "/",
        sharedp=False,
        whitep=False,
        nints=[len(time)],
        wave_low=np.array([1.0]),
        wave_hi=np.array([2.0]),
        wave=np.array([1.5]),
        isplots_S5=1,
        hide_plots=True,
        verbose=False,
        ncpu=1,
        chains=1,
        run_nburn=100,
        run_nsteps=100,
        lsq_method="BFGS",
    )
    meta.set_defaults()
    (tmp_path / "figs").mkdir()
    log = Logedit(str(tmp_path / "stage5.log"))
    longnames, titles, names, pars = s5_fit.make_longparamlist(meta, pars, 1)
    s5_fit.fit_channel(
        meta,
        np.ma.asarray(time),
        flux,
        0,
        unc,
        "synthetic",
        pars,
        log,
        longnames,
        "BJD",
        titles,
        names,
        1,
        None,
        None,
        None,
        None,
        None,
    )
    filename = next(tmp_path.glob(f"S5_{fitter}_fitparams*.csv"))
    result = Table.read(filename)
    np.testing.assert_allclose(result["Mean"][0], 1.01, atol=0.002)
    assert list((tmp_path / "figs").glob("fig5101*"))


@pytest.mark.parametrize("multwhite", [False, True])
def test_common_mode_jit_and_analytic_values(tmp_path, multwhite):
    from astropy.table import Table
    from eureka.lib.logedit import Logedit

    time = np.linspace(0.0, 0.1, 11)
    if multwhite:
        time = np.r_[time, np.linspace(0.2, 0.3, 7)]
    common = np.sin(time)
    path = tmp_path / "common.ecsv"
    Table({"common": common}).write(path, format="ascii.ecsv")
    meta = SimpleNamespace(
        common_mode_file=str(path), common_mode_name="common", verbose=False
    )
    log = Logedit(str(tmp_path / "common.log"))
    pars = parameters(cm1=0.02, cm2=0.01, cm1_ch1=0.03)
    cfg = config(
        pars,
        time=time,
        multwhite=multwhite,
        nchannel_fitted=2,
        fitted_channels=[0, 1],
        nints=[11, 7 if multwhite else 11],
    )
    model = jm.CommonModeModel(meta, log, **cfg)
    centered = common - common.mean()
    x0, x1 = (
        (centered[:11], centered[11:]) if multwhite else (centered, centered)
    )
    expected = np.r_[
        1.0 + 0.02 * x0 + 0.01 * x0**2, 1.0 + 0.03 * x1 + 0.01 * x1**2
    ]
    np.testing.assert_allclose(model.eval(), expected)
    np.testing.assert_allclose(
        jax.jit(lambda v: model.eval(param_dict={"cm1": v}))(0.02),
        expected,
    )
    derivative = jax.grad(
        lambda v: jnp.sum(model.eval(param_dict={"cm1": v}))
    )(0.02)
    assert np.isfinite(derivative)


@pytest.mark.parametrize("jax_first", [True, False])
def test_mixed_model_multiplication(jax_first):
    pars = orbit(c0=1.01, c1=0.02)
    time = np.linspace(-0.05, 0.05, 11)
    cfg = config(pars)
    trend = jm.PolynomialModel(**cfg)
    transit = models.BatmanTransitModel(**cfg)
    expected = np.asarray(trend.eval()) * np.asarray(transit.eval())
    mixed = trend * transit if jax_first else transit * trend
    assert isinstance(mixed, models.CompositeModel)
    assert not isinstance(mixed, jm.CompositeJaxModel)
    mixed.time = time
    np.testing.assert_allclose(mixed.eval(), expected)


@pytest.mark.parametrize("gp", [False, True])
@pytest.mark.parametrize("multwhite", [False, True])
def test_multichannel_numpyro_and_gp_predictions(gp, multwhite):
    model, _, _, _ = composite(gp=gp)
    pars = model.parameters
    pars.c0_ch1 = 1.02, "free", 0.8, 1.2, "U"
    pars.scatter_mult = 1.2, "fixed"
    pars.scatter_mult_ch1 = 1.5, "fixed"
    t = np.linspace(-0.05, 0.05, 11)
    nints = [11, 7 if multwhite else 11]
    time = np.r_[t, np.linspace(0.2, 0.3, 7)] if multwhite else t
    flux = np.r_[
        1.01 + 0.02 * t,
        1.02 + 0.02 * (np.linspace(0.2, 0.3, 7) if multwhite else t),
    ]
    unc = np.full(len(flux), 0.001)
    cfg = config(
        pars,
        time=time,
        nints=nints,
        multwhite=multwhite,
        nchannel_fitted=2,
        fitted_channels=[0, 1],
    )
    components = [jm.PolynomialModel(**cfg)]
    if gp:
        lc = SimpleNamespace(flux=flux, unc=unc, unc_fit=unc)
        components.append(jm.GPModel(["Matern32"], ["time"], lc, **cfg))
    model = jm.CompositeJaxModel(components, **cfg, flux=flux, lc_unc=unc)
    names = ["c0", "c0_ch1", "A", "m"] if gp else ["c0", "c0_ch1"]
    model.freenames = names
    values = {key: getattr(pars, key).value for key in names}
    function = make_numpyro_composite_model(model, time, flux, unc)

    def density(parameters):
        return log_density(function, (time, flux, unc), {}, parameters)[0]

    assert np.isfinite(jax.jit(density)(values))
    assert all(np.isfinite(v) for v in jax.grad(density)(values).values())
    prediction = model.eval(incl_GP=True)
    assert prediction.shape == flux.shape
    np.testing.assert_allclose(
        model.eval(channel=1, incl_GP=True), prediction[nints[0]:]
    )


@pytest.mark.parametrize("gp", [False, True])
def test_masked_scatter_gradient_and_first_trace_purity(gp):
    model, time, flux, unc = composite(gp=gp, masked=True)
    model.parameters.scatter_mult = 1.0, "free", 0.5, 2.0, "U"
    model.freenames = model.freenames + ["scatter_mult"]
    values = {
        name: getattr(model.parameters, name).value for name in model.freenames
    }
    function = make_numpyro_composite_model(model, time, flux, unc)

    def density(parameters):
        return log_density(function, (time, flux, unc), {}, parameters)[0]

    with jax.checking_leaks():
        result = jax.jit(jax.value_and_grad(density))(values)
    assert np.isfinite(result[0])
    assert all(np.isfinite(v) for v in result[1].values())


@pytest.mark.parametrize("option", ["pixelsampling", "compute_ltt"])
def test_starry_unsupported_options_raise(option):
    with pytest.raises(NotImplementedError):
        jm.JaxoplanetStarryModel(**config(orbit(), **{option: True}))


def test_starry_starspot_plot(tmp_path):
    from eureka.S5_lightcurve_fitting.plots_s5 import plot_starry_star

    pars = orbit(spotrad=0.1, spotlat=20.0, spotlon=10.0, spotcon=0.5)
    star = jm.JaxoplanetStarryModel(**config(pars, ydeg=2))
    astro = jm.AstroModel([star], **config(pars))
    model = jm.CompositeJaxModel([astro], **config(pars))
    lc = SimpleNamespace(fitted_channels=[0], white=False, nchannel=1)
    meta = SimpleNamespace(outputdir=str(tmp_path) + "/", hide_plots=True)
    (tmp_path / "figs").mkdir()
    plot_starry_star(lc, model, meta, "jaxopt")
    assert list((tmp_path / "figs").glob("fig5308*"))


def test_sparse_ramp_and_high_step_indices():
    pars = parameters(r0_wl1=0.02, r3=0.1, step12=0.01, steptime12=0.03)
    cfg = config(pars, wl_groups=[1])
    ramp = jm.ExpRampModel(**cfg)
    np.testing.assert_allclose(ramp.eval(), 1.02)
    step = jm.StepModel(**cfg)
    expected = 1.0 + 0.01 * ((step.time - step.time[0]) >= 0.03)
    np.testing.assert_allclose(step.eval(), expected)
