import os
import sys

import numpy as np
import pytest

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S5_lightcurve_fitting import models
from eureka.S5_lightcurve_fitting.models.AstroModel import (
    PlanetParams, correct_light_travel_time,
)
from eureka.S5_lightcurve_fitting.models.KeplerOrbit import KeplerOrbit
from tests.unit.s5_model_helpers import (
    _ConstantModel, _FakeGPModel, _params,
)


def test_model_validates_channel_metadata_lengths():
    """Catch inconsistent multichannel metadata before model evaluation."""
    with pytest.raises(ValueError, match='fitted_channels must have length'):
        models.Model(nchannel_fitted=2, fitted_channels=[0])


def test_model_update_changes_parameter_values_and_attributes():
    params = _params(c0=1.0, c1=2.0)
    model = models.Model(parameters=params, freenames=['c0', 'c1'])

    model.update([3.0, 4.0])

    assert params.c0.value == 3.0
    assert params.c1.value == 4.0
    assert params.dict['c0'][0] == 3.0
    assert params.dict['c1'][0] == 4.0


def test_composite_model_multiplies_components_by_type_and_propagates_time():
    systematic = _ConstantModel([2.0, 3.0], modeltype='systematic')
    physical = _ConstantModel([0.5, 0.25], modeltype='physical')
    composite = models.CompositeModel([systematic, physical])
    composite.time = np.array([0.0, 1.0])

    np.testing.assert_allclose(composite.eval(), [1.0, 0.75])
    np.testing.assert_allclose(composite.syseval(), [2.0, 3.0])
    phys_flux, phys_time, nints = composite.physeval()
    np.testing.assert_allclose(phys_flux, [0.5, 0.25])
    np.testing.assert_allclose(phys_time, [0.0, 1.0])
    assert nints is None
    np.testing.assert_allclose(systematic.time, [0.0, 1.0])
    np.testing.assert_allclose(physical.time, [0.0, 1.0])


def test_model_multiplication_combines_parameters_and_evaluates_product():
    params1 = _params(a=1.0)
    params2 = _params(b=2.0)
    model1 = _ConstantModel([2.0, 2.0], parameters=params1)
    model2 = _ConstantModel([0.5, 0.25], parameters=params2)

    composite = model1*model2

    assert 'a' in composite.parameters.dict
    assert 'b' in composite.parameters.dict
    composite.time = np.array([0.0, 1.0])
    np.testing.assert_allclose(composite.eval(), [1.0, 0.5])


def test_model_interp_restores_original_time_and_nints():
    """Interpolation should be a temporary evaluation, not persistent state."""
    params = _params(c0=1.0, c1=2.0)
    model = models.PolynomialModel(parameters=params, nints=[3])
    original_time = np.array([0.0, 1.0, 2.0])
    model.time = original_time

    result = model.interp(np.array([10.0, 11.0]), [2])

    np.testing.assert_allclose(result, [0.0, 2.0])
    np.testing.assert_allclose(model.time, original_time)
    assert model.nints == [3]


def test_composite_model_includes_gp_prediction_when_requested():
    """GP components add residual structure only when explicitly requested."""
    systematic = _ConstantModel([2.0, 3.0], modeltype='systematic')
    gp = _FakeGPModel([0.1, -0.2])
    composite = models.CompositeModel([systematic, gp])
    composite.time = np.array([0.0, 1.0])

    no_gp = composite.eval()
    with_gp = composite.eval(incl_GP=True)

    np.testing.assert_allclose(no_gp, [2.0, 3.0])
    np.testing.assert_allclose(with_gp, [2.1, 2.8])
    np.testing.assert_allclose(gp.seen_fit, [2.0, 3.0])


def test_planet_params_resolves_aliases_geometry_and_limb_darkening():
    """Protect S5 aliases and derived orbital/limb-darkening conventions."""
    params = _params(limb_dark='kipping2013', rprs=0.11, ars=15.0, b=0.3,
                     per=3.0, t0=0.0, ecosw=0.03, esinw=0.04,
                     fpfs=0.001, u1=0.25, u2=0.4, Rs=1.0)
    model = models.Model(parameters=params)

    planet = PlanetParams(model)

    assert planet.rp == pytest.approx(0.11)
    assert planet.a == pytest.approx(15.0)
    assert planet.inc == pytest.approx(np.degrees(np.arccos(0.3/15.0)))
    assert planet.ecc == pytest.approx(0.05)
    assert planet.w == pytest.approx(np.degrees(np.arctan2(0.04, 0.03)))
    assert planet.fp == pytest.approx(0.001)
    np.testing.assert_allclose(planet.u_original, [0.25, 0.4])
    np.testing.assert_allclose(
        planet.u,
        [2*np.sqrt(0.25)*0.4, np.sqrt(0.25)*(1-2*0.4)]
    )


def test_planet_params_prefers_wavelength_specific_over_channel_parameters():
    """Wavelength-group parameters should override channel-level values."""
    params = _params(rp=0.1, rp_ch1=0.2, rp_wl2=0.3, per=3.0, t0=0.0,
                     inc=89.0, a=12.0, ecc=0.0, w=90.0)
    model = models.Model(parameters=params, nchannel_fitted=1,
                         fitted_channels=[1], wl_groups=[2])

    planet = PlanetParams(model, channel=1)

    assert planet.rp == pytest.approx(0.3)
    assert planet.rprs == pytest.approx(0.3)


def test_kepler_orbit_circular_true_anomaly_distance_and_xyz():
    """Validate circular-orbit geometry against analytic phase positions."""
    orbit = KeplerOrbit(a=12.0, Porb=3.0, inc=90.0, t0=0.0, e=0.0,
                        argp=90.0)
    time = np.array([0.0, 0.75, 1.5])

    true_anom = orbit.true_anomaly(time)
    distance = orbit.distance(time)
    x, y, z = orbit.xyz(time)

    np.testing.assert_allclose(true_anom, [0.0, np.pi/2, np.pi],
                               atol=1e-12)
    np.testing.assert_allclose(distance, 12.0)
    np.testing.assert_allclose(np.sqrt(x*x+y*y+z*z), 12.0)
    assert orbit.phase_eclipse == pytest.approx(0.5)


def test_kepler_orbit_eccentric_distance_matches_conic_formula():
    """Ensure eccentric Kepler distances obey the conic-section formula."""
    orbit = KeplerOrbit(a=12.0, Porb=3.0, inc=88.5, t0=0.0, e=0.05,
                        argp=45.0)
    time = np.linspace(-0.1, 0.1, 7)

    true_anom = orbit.true_anomaly(time)
    distance = orbit.distance(time)
    x, y, z = orbit.xyz(time)

    expected_distance = 12.0*(1-0.05**2)/(1+0.05*np.cos(true_anom))
    np.testing.assert_allclose(distance, expected_distance)
    np.testing.assert_allclose(np.sqrt(x*x+y*y+z*z), expected_distance)


def test_correct_light_travel_time_matches_circular_formula():
    """Pin light-travel-time correction against the circular analytic case."""
    params = _params(rp=0.1, per=3.0, t0=0.0, inc=89.0, a=12.0,
                     ecc=0.0, w=90.0, Rs=1.0)
    model = models.Model(parameters=params)
    planet = PlanetParams(model)
    time = np.array([0.0, 0.25, 0.5])

    corrected = correct_light_travel_time(time, planet)

    a_m = planet.a*planet.Rs*6.957e8
    transit_x = a_m*np.sin(np.deg2rad(planet.inc))
    old_x = transit_x*np.cos(2*np.pi*(time-planet.t0)/planet.per)
    expected = time - ((transit_x-old_x)/299792458.0)/(3600*24)
    np.testing.assert_allclose(corrected, expected, rtol=1e-7)


def test_correct_light_travel_time_eccentric_is_zero_at_transit():
    """Transit midtime should remain fixed by light-travel-time correction."""
    params = _params(rp=0.1, per=3.0, t0=0.0, inc=89.0, a=12.0,
                     ecc=0.05, w=45.0, Rs=1.0)
    model = models.Model(parameters=params)
    planet = PlanetParams(model)
    time = np.array([planet.t0])

    corrected = correct_light_travel_time(time, planet)

    np.testing.assert_allclose(corrected, time, atol=1e-14)
