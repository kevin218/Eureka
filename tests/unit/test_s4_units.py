import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.convolution import Box1DKernel, convolve

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S4_generate_lightcurves import drift, outliers, s4_genLC


class _Log:
    def __init__(self):
        self.messages = []

    def writelog(self, message, **kwargs):
        self.messages.append(message)


class _Values:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, key):
        return self.values[key]


def _trace_image(center=7.3, sigma=1.2, ny=15, nx=6, background=0.1):
    y = np.arange(ny, dtype=float)
    profile = background + np.exp(-0.5*((y-center)/sigma)**2)
    return np.repeat(profile[:, None], nx, axis=1)


def test_highpassfilt_matches_boxcar_convolution_with_extend_boundary():
    """The S4 high-pass helper should match its documented boxcar smoothing."""
    signal = np.array([1.0, 2.0, 10.0, 2.0, 1.0])
    width = 3

    result = drift.highpassfilt(signal, width)

    expected = convolve(signal, Box1DKernel(width), boundary='extend')
    np.testing.assert_allclose(result, expected)


def test_spec1d_measures_integer_spectral_drifts_and_sign_convention():
    """Protect the S4 drift sign convention with known pixel shifts."""
    base = np.exp(-0.5*((np.arange(80)-40)/5)**2)
    input_shifts = np.array([0, 2, -3])
    spectra = np.vstack([np.roll(base, shift) for shift in input_shifts])
    meta = SimpleNamespace(
        drift_postclip=None,
        n_int=3,
        drift_iref=0,
        drift_preclip=0,
        sub_continuum=False,
        sub_mean=False,
        drift_range=10,
        verbose=False,
        isplots_S4=0,
        nplots=0,
        drift_hw=6,
    )

    drift1d, driftwidth, driftmask = drift.spec1D(spectra, meta, _Log())

    np.testing.assert_allclose(drift1d, input_shifts, atol=5e-3)
    assert np.all(driftwidth > 0)
    np.testing.assert_array_equal(driftmask, [False, False, False])


def test_get_outliers_flags_noisy_spectral_column_and_returns_plot_payload():
    """Verify MAD-based S4 column flagging on a synthetic noisy light curve."""
    nints = 20
    nwave = 15
    wave = np.linspace(1.0, 2.0, nwave)
    optspec = np.ones((nints, nwave))
    opterr = np.full_like(optspec, 0.01)
    optmask = np.zeros_like(optspec, dtype=bool)
    noisy_col = 7
    optspec[:, noisy_col] = 1 + 0.08*((-1)**np.arange(nints))
    spec = SimpleNamespace(
        wave_1d=_Values(wave),
        optspec=_Values(optspec),
        opterr=_Values(opterr),
        optmask=_Values(optmask),
        x=np.arange(nwave),
    )
    meta = SimpleNamespace(
        wave_min=wave[0],
        wave_max=wave[-1],
        inst='nircam',
        mad_box_width=5,
        mad_sigma=3,
        maxiters=3,
    )

    flagged, pp = outliers.get_outliers(meta, spec)

    np.testing.assert_array_equal(flagged, [noisy_col])
    assert pp['mad'][noisy_col] > 1e5
    assert pp['x_mad_outliers'].tolist() == [noisy_col]
    assert set(pp) == {
        'x', 'x_mask', 'x_mad_outliers', 'x_dev_outliers', 'mad', 'dev',
        'masked_mad', 'masked_dev', 'smoothed_mad', 'residual_mad',
        'smoothed_dev', 'residual_dev'
    }


def test_compute_wavelength_bins_builds_native_bins_and_pixel_bins():
    """S4 wavelength bin helper should preserve native and pixel-bin logic."""
    wave = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    native_meta = SimpleNamespace(wave_input=None, nspecchan=None,
                                  npixelbins=None, wave_min=1.1,
                                  wave_max=1.3, wave_low=None, wave_hi=None)

    native = s4_genLC.compute_wavelength_bins(native_meta, wave, _Log())

    np.testing.assert_allclose(native.wave, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(native.wave_low, [1.05, 1.15, 1.25])
    np.testing.assert_allclose(native.wave_hi, [1.15, 1.25, 1.35])
    assert native.nspecchan == 3

    pixel_meta = SimpleNamespace(wave_input=None, nspecchan=None,
                                 npixelbins=2, npixelshift=0, wave_min=1.0,
                                 wave_max=1.4, wave_low=None, wave_hi=None)
    pixel = s4_genLC.compute_wavelength_bins(pixel_meta, wave, _Log())

    np.testing.assert_allclose(pixel.wave_low, [0.95])
    np.testing.assert_allclose(pixel.wave_hi, [1.15])
    np.testing.assert_allclose(pixel.wave, [1.05])
    assert pixel.nspecchan == 1


def test_compute_wavelength_bins_respects_explicit_edges_and_warnings():
    """Explicit S4 wavelength edges should override mismatched nspecchan."""
    log = _Log()
    meta = SimpleNamespace(wave_input=None, nspecchan=3, npixelbins=None,
                           wave_min=1.0, wave_max=1.4,
                           wave_low=[1.0, 1.2], wave_hi=[1.2, 1.4])

    result = s4_genLC.compute_wavelength_bins(
        meta, np.array([1.0, 1.2, 1.4]), log
    )

    np.testing.assert_allclose(result.wave, [1.1, 1.3])
    assert result.nspecchan == 2
    assert any('differs from the size' in message for message in log.messages)


def test_compute_wavelength_bins_reads_edges_from_input_file(tmp_path):
    """S4 wavelength bins should support two-column edge files."""
    wave_file = tmp_path/'wave_bins.txt'
    np.savetxt(wave_file, np.array([[1.0, 1.2], [1.2, 1.5]]))
    log = _Log()
    meta = SimpleNamespace(wave_input=str(wave_file), nspecchan=None,
                           npixelbins=None, wave_min=1.0, wave_max=1.5,
                           wave_low=None, wave_hi=None)

    result = s4_genLC.compute_wavelength_bins(
        meta, np.array([1.0, 1.2, 1.5]), log
    )

    np.testing.assert_allclose(result.wave_low, [1.0, 1.2])
    np.testing.assert_allclose(result.wave_hi, [1.2, 1.5])
    np.testing.assert_allclose(result.wave, [1.1, 1.35])
    assert result.nspecchan == 2
    assert 'input file' in log.messages[0]


def test_compute_wavelength_bins_raises_when_pixel_range_is_missing():
    """Pixel-defined bins should fail when bounds are outside data."""
    meta = SimpleNamespace(wave_input=None, nspecchan=None, npixelbins=2,
                           npixelshift=0, wave_min=0.9, wave_max=2.0,
                           wave_low=None, wave_hi=None)

    with pytest.raises(ValueError, match='No wavelengths'):
        s4_genLC.compute_wavelength_bins(meta, np.array([1.0, 1.1]), _Log())


def test_compute_spectral_lightcurve_averages_masked_columns_and_errors():
    """S4 spectral binning should average flux and propagate uncertainties."""
    optspec = np.array([[10.0, 12.0, 14.0],
                        [20.0, 22.0, 24.0]])
    opterr = np.array([[1.0, 2.0, 3.0],
                       [2.0, 4.0, 6.0]])
    optmask = np.array([[False, True, False],
                        [False, False, False]])

    lc_data, lc_err, lc_mask = s4_genLC.compute_spectral_lightcurve(
        optspec, opterr, optmask, np.array([0, 1, 2]), return_mask=True
    )

    np.testing.assert_allclose(lc_data, [12.0, 22.0])
    np.testing.assert_allclose(lc_err, [np.sqrt(10)/2, np.sqrt(56)/3])
    np.testing.assert_array_equal(lc_mask, [False, False])


def test_compute_spectral_lightcurve_masks_fully_masked_integrations():
    """Fully masked bandpasses should produce masked flux and errors."""
    optspec = np.array([[10.0, 12.0], [20.0, 22.0]])
    opterr = np.array([[1.0, 2.0], [2.0, 4.0]])
    optmask = np.array([[True, True], [False, True]])

    lc_data, lc_err, lc_mask = s4_genLC.compute_spectral_lightcurve(
        optspec, opterr, optmask, np.array([0, 1]), return_mask=True
    )

    np.testing.assert_array_equal(np.ma.getmaskarray(lc_data), [True, False])
    np.testing.assert_array_equal(np.ma.getmaskarray(lc_err), [True, False])
    assert lc_data[1] == pytest.approx(20.0)
    assert lc_err[1] == pytest.approx(2.0)
    np.testing.assert_array_equal(lc_mask, [True, False])
