import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S5_lightcurve_fitting import models
from tests.unit.s5_model_helpers import _params


def test_exp_ramp_model_matches_double_exponential_formula():
    params = _params(r0=0.2, r1=0.5, r2=-0.1, r3=0.25)
    model = models.ExpRampModel(parameters=params)
    model.time = np.array([10.0, 11.0, 12.0])

    result = model.eval()

    t = np.array([0.0, 1.0, 2.0])
    expected = 1 + 0.2*np.exp(-0.5*t) - 0.1*np.exp(-0.25*t)
    np.testing.assert_allclose(result, expected)


def test_exp_ramp_model_respects_wavelength_specific_parameter_precedence():
    """Systematics should use wavelength-group parameters when available."""
    params = _params(r0=0.1, r1=1.0, r2=0.0, r3=0.0,
                     r0_ch1=0.2, r0_wl2=0.4)
    model = models.ExpRampModel(parameters=params, nchannel_fitted=1,
                                fitted_channels=[1], wl_groups=[2])
    model.time = np.array([0.0, 1.0])

    result = model.eval(channel=1)

    expected = 1 + 0.4*np.exp(-np.array([0.0, 1.0]))
    np.testing.assert_allclose(result, expected)


def test_step_model_applies_sorted_steps_and_ignores_zero_amplitudes():
    params = _params(step1=-0.1, steptime1=1.0,
                     step2=0.5, steptime2=2.0,
                     step3=0.0, steptime3=0.5)
    model = models.StepModel(parameters=params)
    model.time = np.array([10.0, 11.0, 12.0, 13.0])

    result = model.eval()

    np.testing.assert_allclose(result, [1.0, 0.9, 1.4, 1.4])


def test_step_model_uses_channel_specific_parameters_in_multwhite_mode():
    """Protect local time slicing and channel parameters in multwhite fits."""
    params = _params(step1=0.1, steptime1=1.0,
                     step1_ch1=0.5, steptime1_ch1=2.0)
    model = models.StepModel(parameters=params, multwhite=True,
                             nints=[3, 3], nchannel_fitted=2,
                             fitted_channels=[0, 1])
    model.time = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])

    result = model.eval()

    np.testing.assert_allclose(result, [1.0, 1.1, 1.1, 1.0, 1.0, 1.5])


def test_lorentzian_model_matches_symmetric_formula():
    params = _params(lor_amp=0.4, lor_hwhm=2.0, lor_t0=0.0,
                     lor_power=2.0)
    model = models.LorentzianModel(parameters=params)
    model.time = np.array([-1.0, 0.0, 1.0])

    result = model.eval()

    t = np.array([-1.0, 0.0, 1.0])
    expected = 1 + 0.4/(1 + (2*t/2.0)**2)
    np.testing.assert_allclose(result, expected)


def test_lorentzian_model_matches_asymmetric_amplitude_width_formula():
    params = _params(lor_amp_lhs=0.2, lor_amp_rhs=0.6,
                     lor_hwhm_lhs=1.0, lor_hwhm_rhs=2.0,
                     lor_t0=0.0, lor_power=2.0)
    model = models.LorentzianModel(parameters=params)
    model.time = np.array([-1.0, 0.0, 2.0])

    result = model.eval()

    expected = np.array([
        1 + 0.2/(1 + 1.0**2),
        1 + 0.2,
        1 + 0.2 - 0.6 + 0.6/(1 + 1.0**2),
    ])
    np.testing.assert_allclose(result, expected)


def test_lorentzian_model_rejects_ambiguous_parameterization():
    """Ambiguous symmetric/asymmetric Lorentzian inputs should fail early."""
    params = _params(lor_amp=0.4, lor_amp_lhs=0.2, lor_hwhm=2.0)
    model = models.LorentzianModel(parameters=params)
    model.time = np.array([0.0])

    with pytest.raises(ValueError, match='Ambiguous Lorentzian'):
        model.eval()


def test_centroid_model_matches_mean_centered_linear_decorrelation():
    params = _params(xpos=0.3)
    centroid = np.array([1.0, 2.0, 4.0])
    model = models.CentroidModel('xpos', parameters=params)
    model.centroid = centroid

    result = model.eval()

    expected = 1 + (centroid-centroid.mean())*0.3
    np.testing.assert_allclose(result, expected)


def test_centroid_model_validates_axis_name():
    with pytest.raises(ValueError, match='CentroidModel requires'):
        models.CentroidModel('angle')


def test_hst_ramp_model_matches_orbit_modulo_formula():
    params = _params(h0=0.2, h1=0.5, h2=0.1, h3=-0.01, h4=2.0, h5=0.25)
    model = models.HSTRampModel(parameters=params)
    model.time = np.array([5.0, 5.5, 6.25])

    result = model.eval()

    t = np.array([0.0, 0.5, 1.25])
    t_batch = (t-0.25) % 2.0
    expected = 1 + 0.2*np.exp(-0.5*t_batch) + 0.1*t_batch - 0.01*t_batch**2
    np.testing.assert_allclose(result, expected)


def test_damped_oscillator_model_matches_decay_formula_and_pre_t0_unity():
    """The oscillator must remain unity before its configured start time."""
    params = _params(osc_amp=0.2, osc_amp_decay=0.1,
                     osc_per=2.0, osc_per_decay=0.0,
                     osc_t0=1.0, osc_t1=1.0)
    model = models.DampedOscillatorModel(parameters=params)
    model.time = np.array([0.0, 1.0, 1.5, 2.0])

    result = model.eval()

    t = np.array([0.0, 1.0, 1.5, 2.0])
    amp = 0.2*np.exp(-0.1*(t-1.0))
    expected = 1 + amp*np.sin(2*np.pi*(t-1.0)/2.0)
    expected[t < 1.0] = 1.0
    np.testing.assert_allclose(result, expected)


def test_polynomial_model_evaluates_each_multwhite_channel_locally():
    params = _params(c0=1.0, c1=0.1, c1_ch1=0.5)
    model = models.PolynomialModel(parameters=params, multwhite=True,
                                   nints=[3, 3], nchannel_fitted=2,
                                   fitted_channels=[0, 1])
    model.time = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])

    result = model.eval()

    expected = np.array([0.9, 1.0, 1.1, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(result, expected)


def test_common_mode_model_reads_file_and_applies_quadratic_terms(tmp_path):
    """Common-mode correction should read files and apply polynomial terms."""
    from astropy.io import ascii
    from astropy.table import QTable

    common_mode_file = tmp_path/'common_mode.ecsv'
    ascii.write(QTable({'white_model': [0.8, 1.0, 1.4]}),
                common_mode_file, format='ecsv')
    meta = SimpleNamespace(common_mode_file=str(common_mode_file),
                           common_mode_name='white_model', verbose=False)
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None)
    params = _params(cm1=0.5, cm2=0.25)
    model = models.CommonModeModel(meta, log, parameters=params)
    model.time = np.array([0.0, 1.0, 2.0])

    result = model.eval()

    cm_flux = np.array([0.8, 1.0, 1.4])
    cm_flux -= cm_flux.mean()
    expected = 1 + 0.5*cm_flux + 0.25*cm_flux**2
    np.testing.assert_allclose(result, expected)
