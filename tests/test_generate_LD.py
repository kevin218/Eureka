"""Targeted tests for NIRSpec throughput selection and extrapolation."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from eureka.S4_generate_lightcurves import generate_LD


@pytest.fixture
def ld_inputs(monkeypatch):
    meta = SimpleNamespace(
        inst='nirspec', filter='G140H', nirspec_filter='F100LP',
        exotic_ld_file=None, exotic_ld_grid='mps1', exotic_ld_direc='',
        metallicity=0, teff=5500, logg=4.5, verbose=False, isplots_S4=3,
        nspecchan=1, wave_low=[0.95], wave_hi=[1.1],
        wave_min=0.95, wave_max=1.1, minmu=0.1)
    spec = SimpleNamespace(
        attrs={}, wave_1d=SimpleNamespace(attrs={'wave_units': 'microns'}))
    sld = Mock()
    wave = np.arange(10000., 15001., 100.)
    throughput = 0.1 + (wave - wave[0])*0.0004
    sld._read_sensitivity_data.return_value = wave, throughput
    for law, count in [('linear', 1), ('quadratic', 2), ('kipping', 2),
                       ('squareroot', 2), ('3_parameter_non_linear', 3),
                       ('4_parameter_non_linear', 4)]:
        getattr(sld, f'compute_{law}_ld_coeffs').return_value = np.zeros(count)
    monkeypatch.setattr(generate_LD, 'StellarLimbDarkening', Mock(
        return_value=sld))
    monkeypatch.setattr(generate_LD.plots_s4, 'plot_extrapolated_throughput',
                        Mock())
    return meta, spec, sld


@pytest.mark.parametrize('grating,filter_name,suffix', [
    ('G140H', 'F070LP', '-f070'), ('G140H', 'F100LP', '-f100'),
    ('G140M', 'F070LP', '-f070'), ('G140M', 'F100LP', '-f100'),
    ('G235H', None, ''), ('G235M', None, '')])
@pytest.mark.parametrize('white', [False, True])
@pytest.mark.parametrize('units', ['microns', 'angstroms'])
def test_nirspec_linear_extrapolation(ld_inputs, grating, filter_name, suffix,
                                      white, units):
    meta, spec, sld = ld_inputs
    meta.filter, meta.nirspec_filter = grating, filter_name
    if units == 'angstroms':
        meta.wave_low, meta.wave_hi = [9500], [11000]
        meta.wave_min, meta.wave_max = 9500, 11000
        spec.wave_1d.attrs['wave_units'] = units
    generate_LD.exotic_ld(meta, spec, Mock(), white=white)
    sld._read_sensitivity_data.assert_called_once_with(
        f'JWST_NIRSpec_{grating}{suffix}')
    args = sld.compute_linear_ld_coeffs.call_args.args
    assert args[1] == 'custom'
    wave, throughput = args[2:]
    assert wave[0] == 9500
    assert np.all(np.diff(wave) > 0)
    expected = np.maximum(0.1 + (wave[:1000] - 10000)*0.0004, 0)
    np.testing.assert_allclose(throughput[:1000], expected, atol=1e-15)
    original_wave, original_tp = sld._read_sensitivity_data.return_value
    np.testing.assert_array_equal(wave[1000:], original_wave)
    np.testing.assert_array_equal(throughput[1000:], original_tp)
    generate_LD.plots_s4.plot_extrapolated_throughput.assert_called_once()


@pytest.mark.parametrize('lower_edge', [1.0, 1.05])
@pytest.mark.parametrize('grating,suffix',
                         [('G140H', '-f100'), ('G235H', ''), ('G235M', '')])
def test_nirspec_within_coverage(ld_inputs, lower_edge, grating, suffix):
    meta, spec, sld = ld_inputs
    meta.filter = grating
    meta.wave_low = [lower_edge]
    generate_LD.exotic_ld(meta, spec, Mock())
    args = sld.compute_linear_ld_coeffs.call_args.args
    assert args[1:] == (f'JWST_NIRSpec_{grating}{suffix}', None, None)
    generate_LD.plots_s4.plot_extrapolated_throughput.assert_not_called()


@pytest.mark.parametrize('grating', ['G140H', 'G140M', 'G235H', 'G235M'])
def test_nirspec_fitting_context(ld_inputs, grating):
    meta, spec, sld = ld_inputs
    meta.filter = grating
    wave, _ = sld._read_sensitivity_data.return_value
    # A flat blue edge followed by a rise distinguishes the fitting windows.
    throughput = np.full_like(wave, 0.2)
    sld._read_sensitivity_data.return_value = wave, throughput
    generate_LD.exotic_ld(meta, spec, Mock())
    baseline = sld.compute_linear_ld_coeffs.call_args.args[3].copy()
    throughput[(wave >= 11000) & (wave < 13000)] += 0.1
    generate_LD.exotic_ld(meta, spec, Mock())
    result = sld.compute_linear_ld_coeffs.call_args.args[3].copy()
    if grating.startswith('G235'):
        assert np.all(result[:1000] < baseline[:1000])
    else:
        np.testing.assert_array_equal(result[:1000], baseline[:1000])
    # Data beyond 0.30 microns must not affect either extrapolation.
    throughput[wave >= 13000] += 0.5
    generate_LD.exotic_ld(meta, spec, Mock())
    np.testing.assert_array_equal(
        sld.compute_linear_ld_coeffs.call_args.args[3][:1000], result[:1000])


@pytest.mark.parametrize('filter_name', [None, 'F290LP'])
def test_g140_invalid_filter(ld_inputs, filter_name):
    meta, spec, sld = ld_inputs
    meta.nirspec_filter = filter_name
    with pytest.raises(ValueError, match='FILTER'):
        generate_LD.exotic_ld(meta, spec, Mock())
    sld._read_sensitivity_data.assert_not_called()


@pytest.mark.parametrize('lower_edge', [2.733, 2.87, 3.0])
@pytest.mark.parametrize('units', ['microns', 'angstroms'])
def test_g395h_blue_edge(ld_inputs, lower_edge, units):
    meta, spec, sld = ld_inputs
    meta.filter = 'G395H'
    wave = np.linspace(28700, 51769.2, 500)
    throughput = 0.3 + 0.2*np.sin((wave - wave[0])/20000)
    sld._read_sensitivity_data.return_value = wave, throughput
    scale = 1e4 if units == 'angstroms' else 1
    meta.wave_low, meta.wave_hi = [lower_edge*scale], [3.5*scale]
    spec.wave_1d.attrs['wave_units'] = units
    generate_LD.exotic_ld(meta, spec, Mock())
    args = sld.compute_linear_ld_coeffs.call_args.args
    if lower_edge < 2.87:
        assert args[1] == 'custom'
        assert args[2][0] == 27330
        assert np.all(np.diff(args[2]) > 0)
        assert np.all(args[3] >= 0)
        np.testing.assert_array_equal(args[2][10000:], wave)
        np.testing.assert_array_equal(args[3][10000:], throughput)
    else:
        assert args[1:] == ('JWST_NIRSpec_G395H', None, None)
        generate_LD.plots_s4.plot_extrapolated_throughput.assert_not_called()
