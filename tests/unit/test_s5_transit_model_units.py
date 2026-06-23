import os
import sys

import numpy as np
import pytest

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from types import SimpleNamespace

from eureka.S5_lightcurve_fitting import models
from eureka.S5_lightcurve_fitting.models.AstroModel import (
    PlanetParams, get_ecl_midpt,
)
from tests.unit.s5_model_helpers import (
    _params, _transit_model, _transit_params,
)


def test_batman_transit_model_zero_radius_is_unity():
    params = _params(limb_dark='uniform', rp=0.0, per=2.0, t0=0.0,
                     inc=89.0, a=10.0, ecc=0.0, w=90.0)
    model = models.BatmanTransitModel(parameters=params,
                                      paramtitles=list(params.dict.keys()),
                                      num_planets=1, ld_from_S4=False,
                                      ld_from_file=False)
    model.time = np.array([-0.1, 0.0, 0.1])

    result = model.eval()

    np.testing.assert_allclose(result, np.ones(3))


def test_batman_transit_model_invalid_geometry_returns_penalty():
    """Invalid transit geometry should return penalties rather than crash."""
    params = _params(limb_dark='uniform', rp=0.1, per=2.0, t0=0.0,
                     inc=95.0, a=10.0, ecc=0.0, w=90.0)
    model = models.BatmanTransitModel(parameters=params,
                                      paramtitles=list(params.dict.keys()),
                                      num_planets=1, ld_from_S4=False,
                                      ld_from_file=False)
    model.time = np.array([0.0, 0.1])

    result = model.eval()

    np.testing.assert_allclose(result, [1e6, 1e6])


def test_batman_eclipse_model_zero_planet_flux_is_zero():
    params = _params(limb_dark='uniform', rp=0.1, fp=0.0, per=2.0, t0=0.0,
                     t_secondary=1.0, inc=89.0, a=10.0, ecc=0.0, w=90.0)
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None)
    model = models.BatmanEclipseModel(parameters=params,
                                      paramtitles=list(params.dict.keys()),
                                      num_planets=1, compute_ltt=False,
                                      log=log)
    model.time = np.array([0.9, 1.0, 1.1])

    result = model.eval()

    np.testing.assert_allclose(result, np.zeros(3))


@pytest.mark.parametrize('ecc,w', [
    (0.0, 90.0),
    (0.03, 45.0),
    (0.05, 135.0),
])
def test_poet_eclipse_model_matches_batman_for_eccentric_orbits(ecc, w):
    """Benchmark POET eclipse geometry against batman, including small ecc."""
    params = _params(limb_dark='uniform', rp=0.08, fp=0.01, per=3.0,
                     t0=0.0, inc=88.5, a=12.0, ecc=ecc, w=w)
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None)
    ref_model = models.BatmanTransitModel(
        parameters=params, paramtitles=list(params.dict.keys()),
        num_planets=1, ld_from_S4=False, ld_from_file=False
    )
    t_secondary = get_ecl_midpt(PlanetParams(ref_model))
    time = np.linspace(t_secondary-0.12, t_secondary+0.12, 61)
    batman = models.BatmanEclipseModel(
        parameters=params, paramtitles=list(params.dict.keys()),
        num_planets=1, compute_ltt=False, log=log
    )
    poet = models.PoetEclipseModel(
        parameters=params, paramtitles=list(params.dict.keys()),
        longparamlist=[list(params.dict.keys())],
        num_planets=1, compute_ltt=False, log=log
    )
    batman.time = time
    poet.time = time

    np.testing.assert_allclose(poet.eval(), batman.eval(), atol=1e-10, rtol=0)


def test_astro_model_combines_stellar_transit_eclipse_and_phase_components():
    """AstroModel combines components with Eureka flux convention."""
    class Component:
        def __init__(self, name, values):
            self.name = name
            self.values = np.ma.array(values)

        def eval(self, channel=None, pid=None, **kwargs):
            return self.values

    stellar = Component('stellar ramp', [2.0, 3.0])
    transit = Component('transit', [0.9, 0.8])
    eclipse = Component('eclipse', [0.1, 0.2])
    phase = Component('phase curve', [1.5, 2.0])
    model = models.AstroModel([stellar, transit, eclipse, phase],
                              num_planets=1)
    model.time = np.array([0.0, 1.0])

    result = model.eval()

    expected = np.array([2.0, 3.0])*np.array([0.9, 0.8])
    expected += np.array([0.1, 0.2])*np.array([1.5, 2.0])
    np.testing.assert_allclose(result, expected)


def test_gp_model_normalizes_kernel_inputs_and_masks_zero_residual_output():
    """GP inputs should normalize consistently and preserve masked samples."""
    params = _params(A=np.log(4.0), m=np.log(2.0))
    lc = SimpleNamespace(flux=np.ones(4), unc=np.full(4, 0.1),
                         unc_fit=np.full(4, 0.1))
    model = models.GPModel(['Matern32'], ['time'], lc,
                           gp_code_name='celerite', parameters=params,
                           normalize=True)
    model.time = np.ma.array([0.0, 1.0, 2.0, 3.0],
                             mask=[False, False, True, False])

    model.setup_inputs()
    result = model.eval(np.ones(4))

    expected_inputs = (model.time-model.time.mean())/model.time.std()
    np.testing.assert_allclose(model.kernel_inputs[0][0], expected_inputs)
    np.testing.assert_allclose(result[~result.mask], 0.0, atol=1e-12)
    np.testing.assert_array_equal(np.ma.getmaskarray(result),
                                  [False, False, True, False])


def test_gp_model_rejects_unsupported_celerite_kernel_configuration():
    """Unsupported celerite kernels should fail before sampler execution."""
    lc = SimpleNamespace(flux=np.ones(3), unc=np.full(3, 0.1),
                         unc_fit=np.full(3, 0.1))

    with pytest.raises(AssertionError, match='single kernel'):
        models.GPModel(['Matern32', 'Matern32'], ['time'], lc,
                       gp_code_name='celerite')

    with pytest.raises(AssertionError, match='supports only'):
        models.GPModel(['ExpSquared'], ['time'], lc, gp_code_name='celerite')


def test_gp_model_loglikelihood_is_finite_for_simple_celerite_case():
    params = _params(A=np.log(0.01), m=np.log(1.0))
    lc = SimpleNamespace(flux=np.array([1.0, 1.01, 0.99, 1.0]),
                         unc=np.full(4, 0.1), unc_fit=np.full(4, 0.1))
    model = models.GPModel(['Matern32'], ['time'], lc,
                           gp_code_name='celerite', parameters=params)
    model.time = np.array([0.0, 1.0, 2.0, 3.0])

    loglike = model.loglikelihood(np.ones(4))

    assert np.isfinite(loglike)


@pytest.mark.parametrize('ecc,w', [
    (0.0, 90.0),
    (0.03, 45.0),
    (0.05, 135.0),
    (0.08, -60.0),
])
def test_poet_transit_model_matches_batman_for_eccentric_orbits(ecc, w):
    """Benchmark POET transits against batman geometry conventions."""
    time = np.linspace(-0.12, 0.12, 61)
    params = _transit_params(ecc=ecc, w=w)

    batman_lc = _transit_model(models.BatmanTransitModel, params, time)
    poet_lc = _transit_model(models.PoetTransitModel, params, time)

    np.testing.assert_allclose(poet_lc, batman_lc, atol=1e-10, rtol=0)


@pytest.mark.parametrize('ecc,w', [
    (0.0, 90.0),
    (0.03, 45.0),
    (0.05, 135.0),
    (0.08, -60.0),
])
def test_fleck_transit_model_matches_batman_for_spotless_eccentric_orbits(
        ecc, w):
    """A spotless Fleck transit should reduce to the batman light curve."""
    time = np.linspace(-0.12, 0.12, 61)
    params = _transit_params(ecc=ecc, w=w)

    batman_lc = _transit_model(models.BatmanTransitModel, params, time)
    fleck_lc = _transit_model(models.FleckTransitModel, params, time)

    np.testing.assert_allclose(fleck_lc, batman_lc, atol=1e-6, rtol=0)


@pytest.mark.parametrize('ecc,w', [
    (0.0, 90.0),
    (0.03, 45.0),
    (0.05, 135.0),
    (0.08, -60.0),
])
def test_harmonica_transit_model_matches_batman_for_eccentric_orbits(ecc, w):
    """Harmonica with no map terms should match batman transit geometry."""
    time = np.linspace(-0.12, 0.12, 61)
    params = _transit_params(ecc=ecc, w=w)

    batman_lc = _transit_model(models.BatmanTransitModel, params, time)
    harmonica_lc = _transit_model(models.HarmonicaTransitModel, params, time)

    np.testing.assert_allclose(harmonica_lc, batman_lc, atol=2e-6, rtol=0)


def test_harmonica_transmission_coefficients_change_transit_shape():
    """Non-zero Harmonica map coefficients should affect transit shape."""
    time = np.linspace(-0.08, 0.08, 81)
    base_params = _transit_params(ecc=0.0, w=90.0)
    map_params = _transit_params(ecc=0.0, w=90.0)
    map_params.a1 = 0.02, 'free'
    map_params.b1 = -0.01, 'free'
    map_params.a2 = 0.01, 'free'
    map_params.b2 = 0.005, 'free'

    base_lc = _transit_model(models.HarmonicaTransitModel, base_params, time)
    map_lc = _transit_model(models.HarmonicaTransitModel, map_params, time)

    assert np.max(np.abs(map_lc-base_lc)) > 5e-4


def test_fleck_spot_on_transit_chord_changes_lightcurve():
    """Check Fleck spot latitude convention using a non-equatorial chord."""
    time = np.linspace(-0.08, 0.08, 81)
    spotless = _transit_params(ecc=0.0, w=90.0)
    spotless.inc.value = 88.5
    spotted = _transit_params(ecc=0.0, w=90.0)
    spotted.inc.value = 88.5
    chord_lat = -np.degrees(np.arcsin(spotted.a.value *
                                      np.cos(np.deg2rad(spotted.inc.value))))
    spotted.spotcon = 0.5, 'free'
    spotted.spotrad = 0.04, 'free'
    spotted.spotlat = chord_lat, 'free'
    spotted.spotlon = 0.0, 'free'

    spotless_lc = _transit_model(models.FleckTransitModel, spotless, time)
    spotted_lc = _transit_model(models.FleckTransitModel, spotted, time)

    assert np.max(np.abs(spotted_lc-spotless_lc)) > 5e-4
    assert spotted_lc[len(time)//2] > spotless_lc[len(time)//2]


def test_fleck_invalid_spot_geometry_returns_penalty():
    """Invalid spot geometry should produce the standard fitting penalty."""
    time = np.linspace(-0.02, 0.02, 5)
    params = _transit_params(ecc=0.0, w=90.0)
    params.spotcon = 0.5, 'free'
    params.spotrad = 1.5, 'free'
    params.spotlat = 0.0, 'free'
    params.spotlon = 0.0, 'free'

    result = _transit_model(models.FleckTransitModel, params, time)

    np.testing.assert_allclose(result, np.full(time.shape, 1e6))


def test_catwoman_transit_model_matches_batman_for_circular_equal_radii():
    """Equal-radius Catwoman planets should match circular batman transits."""
    time = np.linspace(-0.12, 0.12, 61)
    params = _transit_params(ecc=0.0, w=90.0)
    params.rp2 = 0.08, 'free'
    params.phi = 90.0, 'free'
    batman_lc = _transit_model(models.BatmanTransitModel, params, time)
    catwoman = models.CatwomanTransitModel(
        parameters=params, paramtitles=list(params.dict.keys()),
        longparamlist=[list(params.dict.keys())],
        num_planets=1, ld_from_S4=False, ld_from_file=False,
        max_err=0.001, fac=0.001
    )
    catwoman.time = time

    catwoman_lc = catwoman.eval()

    np.testing.assert_allclose(catwoman_lc, batman_lc, atol=1e-7, rtol=0)


def test_catwoman_transit_model_requires_second_radius_parameter():
    """Catwoman-specific geometry requires the second radius parameter."""
    params = _transit_params(ecc=0.0, w=90.0)

    with pytest.raises(AssertionError, match='rp2'):
        models.CatwomanTransitModel(
            parameters=params, paramtitles=list(params.dict.keys()),
            longparamlist=[list(params.dict.keys())],
            num_planets=1, ld_from_S4=False, ld_from_file=False,
            max_err=0.001, fac=0.001
        )
