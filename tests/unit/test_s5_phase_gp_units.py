import os
import sys

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S5_lightcurve_fitting import models
from eureka.S5_lightcurve_fitting.models.AstroModel import (
    PlanetParams, true_anomaly,
)
from tests.unit.s5_model_helpers import _params


def test_sinusoid_phase_curve_matches_circular_orbit_formula():
    params = _params(per=2.0, t0=0.0, t_secondary=0.5, inc=89.0, a=10.0,
                     ecc=0.0, w=90.0, AmpCos1=0.2, AmpSin1=0.1,
                     AmpCos2=0.05, AmpSin2=-0.02)
    model = models.SinusoidPhaseCurveModel(parameters=params, num_planets=1)
    model.time = np.array([0.0, 0.5, 1.0])

    result = model.eval()

    phi = 2*np.pi/2.0*(model.time-0.5)
    expected = (1 + 0.2*(np.cos(phi)-1) + 0.1*np.sin(phi) +
                0.05*(np.cos(2*phi)-1) - 0.02*np.sin(2*phi))
    np.testing.assert_allclose(result, expected)


def test_sinusoid_phase_curve_force_positivity_penalty():
    """Physically invalid negative phase curves should return fit penalties."""
    params = _params(per=2.0, t0=0.0, t_secondary=0.0, inc=89.0, a=10.0,
                     ecc=0.0, w=90.0, AmpCos1=2.0, AmpSin1=0.0,
                     AmpCos2=0.0, AmpSin2=0.0)
    model = models.SinusoidPhaseCurveModel(parameters=params, num_planets=1,
                                           force_positivity=True)
    model.time = np.array([0.0, 1.0])

    result = model.eval()

    assert np.all(result > 1e6)
    np.testing.assert_allclose(result[0], result[1])


def test_sinusoid_phase_curve_matches_eccentric_true_anomaly_formula():
    """Guard eccentric phase-curve geometry against true-anomaly mistakes."""
    params = _params(per=3.0, t0=0.0, inc=88.5, a=12.0, ecc=0.05,
                     w=45.0, AmpCos1=0.2, AmpSin1=-0.1,
                     AmpCos2=0.05, AmpSin2=0.03)
    model = models.SinusoidPhaseCurveModel(parameters=params, num_planets=1)
    model.time = np.array([-0.1, 0.0, 0.1])

    result = model.eval()

    planet = PlanetParams(model)
    phi = true_anomaly(planet, model.time) + planet.w*np.pi/180 + np.pi/2
    expected = (1 + 0.2*(np.cos(phi)-1) - 0.1*np.sin(phi) +
                0.05*(np.cos(2*phi)-1) + 0.03*np.sin(2*phi))
    np.testing.assert_allclose(result, expected)


def test_quasi_lambertian_phase_curve_matches_circular_orbit_formula():
    params = _params(per=2.0, t0=0.0, t_secondary=0.5, inc=89.0, a=10.0,
                     ecc=0.0, w=90.0, quasi_gamma=2.0,
                     quasi_offset=30.0)
    model = models.QuasiLambertianPhaseCurve(parameters=params, num_planets=1)
    model.time = np.array([0.0, 0.5, 1.0])

    result = model.eval()

    phi = 2*np.pi/2.0*(model.time-0.5)
    expected = np.abs(np.cos((phi+30*np.pi/180)/2))**2
    np.testing.assert_allclose(result, expected)


def test_quasi_lambertian_phase_curve_matches_eccentric_true_anomaly():
    """Check quasi-Lambertian phase on eccentric true-anomaly geometry."""
    params = _params(per=3.0, t0=0.0, inc=88.5, a=12.0, ecc=0.05,
                     w=45.0, quasi_gamma=1.5, quasi_offset=-20.0)
    model = models.QuasiLambertianPhaseCurve(parameters=params, num_planets=1)
    model.time = np.array([-0.1, 0.0, 0.1])

    result = model.eval()

    planet = PlanetParams(model)
    phi = true_anomaly(planet, model.time) + planet.w*np.pi/180 + np.pi/2
    expected = np.abs(np.cos((phi-20*np.pi/180)/2))**1.5
    np.testing.assert_allclose(result, expected)


def test_poet_phase_curve_matches_cosine_terms_and_eclipse_normalization():
    """POET phase curves should normalize relative to eclipse center."""
    params = _params(per=2.0, t0=0.0, t_secondary=0.5, inc=89.0, a=10.0,
                     ecc=0.0, w=90.0, cos1_amp=0.2, cos1_off=90.0,
                     cos2_amp=0.1, cos2_off=180.0)
    model = models.PoetPCModel(parameters=params, num_planets=1)
    model.time = np.array([0.0, 0.5, 1.0])

    result = model.eval()

    p = 2.0
    t1 = 90*p/360 - 0.5
    t2 = 180*p/360 - 0.5
    expected = (0.2/2*np.cos(2*np.pi*(model.time+t1)/p) +
                0.1/2*np.cos(4*np.pi*(model.time+t2)/p))
    expected += 1 - expected[np.argmin(np.abs(model.time-0.5))]
    np.testing.assert_allclose(result, expected)
