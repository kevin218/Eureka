from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from astropy.convolution import Box1DKernel, convolve
from tests.unit.helpers import RecordingLog

from eureka.S4_generate_lightcurves import (drift, outliers, plots_s4,
                                            s4_genLC, wfc3)


class _Values:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, key):
        return self.values[key]


@pytest.mark.parametrize('time_axis', ['x', 'y'])
@pytest.mark.parametrize('wave_low, wave_hi, expected_edges', [
    pytest.param([1., 1.6], [1.3, 2.], [1., 1.3, 1.6, 2.], id='gapped'),
    pytest.param([1., 1.5], [1.5, 2.], [1., 1.5, 2.], id='contiguous'),
])
def test_drift_plot_marks_all_unique_bin_edges(
        monkeypatch, time_axis, wave_low, wave_hi, expected_edges):
    """Show both ends of gapped bins without duplicating shared boundaries."""
    monkeypatch.setattr(plots_s4.plt, 'savefig', lambda *args, **kwargs: None)
    monkeypatch.setattr(plots_s4.plt, 'pause', lambda *args, **kwargs: None)
    wave = np.linspace(1., 2., 12)
    spectrum = xr.DataArray(np.ones((4, 12)), dims=('time', 'x'))
    meta = SimpleNamespace(
        inst='nircam', wave_min=1., wave_max=2., n_int=4,
        time_axis=time_axis, vmin=.95, vmax=1.05,
        wave=(np.array(wave_low) + np.array(wave_hi))/2,
        wave_low=np.array(wave_low), wave_hi=np.array(wave_hi),
        nspecchan=2, maed_s4=1., outputdir='', hide_plots=False,
    )
    try:
        plots_s4.lc_driftcorr(meta, wave, spectrum,
                              optmask=xr.zeros_like(spectrum, dtype=bool))

        axis = plots_s4.plt.figure(4101).axes[0]
        secondary = axis.child_axes[0]
        ticks = (secondary.get_xticks() if time_axis == 'y' else
                 secondary.get_yticks())
        np.testing.assert_allclose(ticks, expected_edges)
        segments = np.array(axis.collections[-1].get_segments())
        wave_axis = 0 if time_axis == 'y' else 1
        np.testing.assert_allclose(segments[:, 0, wave_axis], expected_edges)
    finally:
        plots_s4.plt.close(4101)


def test_sum_reads_propagates_rescaled_flux_errors_and_variances():
    """Missing reads reduce information even when flux is rescaled."""
    flux = np.array([[10., 10., 30., 70.],
                     [20., 20., 999., 90.],
                     [40., 40., 40., 40.],
                     [50., 50., 50., 50.]])
    err = np.repeat([[1.], [2.], [3.], [4.]], 4, axis=1)
    mask = np.array([[False, True, False, True],
                     [False, False, True, True],
                     [False, True, False, False],
                     [False, False, False, False]])
    spec = xr.Dataset(
        {'optspec': (['time', 'x'], flux),
         'opterr': (['time', 'x'], err),
         'optmask': (['time', 'x'], mask),
         'stdspec': (['time', 'x'], 2*flux),
         'stdvar': (['time', 'x'], 4*err**2)},
        coords={'time': np.arange(4, dtype=float), 'x': np.arange(4)},
    )
    for key in ('centroid_x', 'centroid_y', 'guess', 'scandir'):
        spec[key] = (['time'], np.zeros(4))
    lc = xr.Dataset(
        {'data': (['time'], np.ones(4)),
         'centroid_x': spec.centroid_x,
         'centroid_y': spec.centroid_y,
         'scandir': spec.scandir},
        coords={'time': spec.time},
    )
    meta = SimpleNamespace(nreads=2, correctDrift=False)

    spec, lc, meta = wfc3.sum_reads(spec, lc, meta)

    expected_flux = [[30., 40., 60., np.nan], [90., 100., 90., 90.]]
    expected_err = [[np.sqrt(5), 4., 2., np.nan], [5., 8., 5., 5.]]
    expected_stdvar = [[20., 64., 16., np.nan], [100., 256., 100., 100.]]
    np.testing.assert_allclose(spec.optspec.values, expected_flux)
    np.testing.assert_allclose(spec.opterr.values, expected_err)
    np.testing.assert_allclose(spec.stdspec.values, 2*np.array(expected_flux))
    np.testing.assert_allclose(spec.stdvar.values, expected_stdvar)
    np.testing.assert_array_equal(spec.optmask.values,
                                  [[False, False, False, True],
                                   [False, False, False, False]])
    np.testing.assert_allclose(spec.time.values, [0.5, 2.5])
    np.testing.assert_allclose(lc.time.values, [0.5, 2.5])
    assert spec.sizes['time'] == lc.sizes['time'] == 2
    assert meta.nreads == 1
    assert meta.n_int == 2


def test_sum_reads_averages_covariates_from_unmasked_reads():
    """A padded read must not leave NaN centroids on recovered exposures."""
    mask = np.array([[True, True], [False, False],
                     [False, False], [False, False]])
    covariates = {
        'centroid_x': [np.nan, 5.0, 10.0, 12.0],
        'centroid_y': [np.nan, 20.0, 30.0, 32.0],
        'centroid_sx': [np.nan, 2.0, 4.0, 6.0],
        'centroid_sy': [np.nan, 3.0, 5.0, 7.0],
    }
    spec = xr.Dataset(
        {'optspec': (['time', 'x'], np.ones((4, 2))),
         'opterr': (['time', 'x'], np.ones((4, 2))),
         'optmask': (['time', 'x'], mask),
         'stdspec': (['time', 'x'], np.ones((4, 2))),
         'stdvar': (['time', 'x'], np.ones((4, 2)))},
        coords={'time': [0.0, 2.0, 10.0, 14.0], 'x': [0, 1]},
    )
    lc = xr.Dataset(
        {'data': (['time'], np.ones(4))},
        coords={'time': spec.time},
    )
    for name, values in covariates.items():
        spec[name] = (['time'], values)
        lc[name] = (['time'], values)

    spec, lc, _ = wfc3.sum_reads(
        spec, lc, SimpleNamespace(nreads=2, correctDrift=False)
    )

    for dataset in (spec, lc):
        np.testing.assert_allclose(dataset.centroid_x.values, [5.0, 11.0])
        np.testing.assert_allclose(dataset.centroid_y.values, [20.0, 31.0])
        np.testing.assert_allclose(dataset.centroid_sx.values, [2.0, 5.0])
        np.testing.assert_allclose(dataset.centroid_sy.values, [3.0, 6.0])


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

    drift1d, driftwidth, driftmask = drift.spec1D(
        spectra, meta, RecordingLog()
    )

    np.testing.assert_allclose(drift1d, input_shifts, atol=5e-3)
    assert np.all(driftwidth > 0)
    np.testing.assert_array_equal(driftmask, [False, False, False])


def test_get_outliers_flags_noisy_spectral_column_and_returns_plot_payload():
    """Verify MAED-based S4 column flagging on a synthetic noisy curve."""
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
        maed_box_width=5,
        maed_sigma=3,
        maxiters=3,
    )

    flagged, pp = outliers.get_outliers(meta, spec)

    np.testing.assert_array_equal(flagged, [noisy_col])
    assert pp['maed'][noisy_col] > 1e5
    assert pp['x_maed_outliers'].tolist() == [noisy_col]
    assert set(pp) == {
        'x', 'x_mask', 'x_maed_outliers', 'x_dev_outliers', 'maed', 'dev',
        'masked_maed', 'masked_dev', 'smoothed_maed', 'residual_maed',
        'smoothed_dev', 'residual_dev'
    }


def test_compute_wavelength_bins_builds_native_bins_and_pixel_bins():
    """S4 wavelength bin helper should preserve native and pixel-bin logic."""
    wave = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    native_meta = SimpleNamespace(wave_input=None, nspecchan=None,
                                  npixelbins=None, wave_min=1.1,
                                  wave_max=1.3, wave_low=None, wave_hi=None)

    native = s4_genLC.compute_wavelength_bins(
        native_meta, wave, RecordingLog()
    )

    np.testing.assert_allclose(native.wave, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(native.wave_low, [1.05, 1.15, 1.25])
    np.testing.assert_allclose(native.wave_hi, [1.15, 1.25, 1.35])
    assert native.nspecchan == 3

    pixel_meta = SimpleNamespace(wave_input=None, nspecchan=None,
                                 npixelbins=2, npixelshift=0, wave_min=1.0,
                                 wave_max=1.4, wave_low=None, wave_hi=None)
    pixel = s4_genLC.compute_wavelength_bins(pixel_meta, wave, RecordingLog())

    np.testing.assert_allclose(pixel.wave_low, [0.95])
    np.testing.assert_allclose(pixel.wave_hi, [1.15])
    np.testing.assert_allclose(pixel.wave, [1.05])
    assert pixel.nspecchan == 1


@pytest.mark.parametrize('shift,expected_low,expected_hi,expected_wave', [
    pytest.param(0, [2.5, 5.5], [5.5, 8.5], [4.0, 7.0], id='unshifted'),
    pytest.param(1, [3.5, 6.5], [6.5, 9.5], [5.0, 8.0], id='shift-right'),
    pytest.param(-1, [1.5, 4.5], [4.5, 7.5], [3.0, 6.0], id='shift-left'),
])
def test_compute_wavelength_bins_shifts_both_pixel_range_endpoints(
        shift, expected_low, expected_hi, expected_wave):
    """A pixel shift moves the whole bandpass, including the final bin."""
    wave = np.arange(1.0, 13.0)
    meta = SimpleNamespace(wave_input=None, nspecchan=None, npixelbins=3,
                           npixelshift=shift, wave_min=3.0, wave_max=8.0,
                           wave_low=None, wave_hi=None)

    result = s4_genLC.compute_wavelength_bins(meta, wave, RecordingLog())

    np.testing.assert_allclose(result.wave_low, expected_low)
    np.testing.assert_allclose(result.wave_hi, expected_hi)
    np.testing.assert_allclose(result.wave, expected_wave)
    assert result.nspecchan == 2


@pytest.mark.parametrize('wave_max', [9.0, 10.0])
def test_compute_wavelength_bins_discards_partial_final_pixel_bin(wave_max):
    """Retain complete pixel bins when one or two selected samples remain."""
    wave = np.arange(1.0, 13.0)
    meta = SimpleNamespace(wave_input=None, nspecchan=None, npixelbins=3,
                           npixelshift=0, wave_min=3.0, wave_max=wave_max,
                           wave_low=None, wave_hi=None)

    result = s4_genLC.compute_wavelength_bins(meta, wave, RecordingLog())

    # Keep samples 3--8 in two bins; the trailing samples 9/10 are excluded.
    np.testing.assert_allclose(result.wave_low, [2.5, 5.5])
    np.testing.assert_allclose(result.wave_hi, [5.5, 8.5])
    np.testing.assert_allclose(result.wave, [4.0, 7.0])
    assert result.nspecchan == 2


def test_compute_wavelength_bins_uses_midpoints_on_nonuniform_grid():
    """Pixel-bin boundaries fall between adjacent detector pixels."""
    wave = np.array([1.0, 1.2, 1.5, 1.9, 2.4, 3.0,
                     3.7, 4.5, 5.4, 6.4, 7.5, 8.7])
    meta = SimpleNamespace(wave_input=None, nspecchan=None, npixelbins=3,
                           npixelshift=0, wave_min=1.5, wave_max=4.5,
                           wave_low=None, wave_hi=None)

    result = s4_genLC.compute_wavelength_bins(meta, wave, RecordingLog())

    np.testing.assert_allclose(result.wave_low, [1.35, 2.7])
    np.testing.assert_allclose(result.wave_hi, [2.7, 4.95])
    np.testing.assert_allclose(result.wave, [2.025, 3.825])
    for wave_low, wave_hi in zip(result.wave_low, result.wave_hi):
        pixels = wave[(wave >= wave_low) & (wave < wave_hi)]
        assert pixels.size == meta.npixelbins
    assert result.nspecchan == 2


@pytest.mark.parametrize('wave', [[], [1.0]], ids=['empty', 'single-sample'])
def test_compute_wavelength_bins_rejects_undersized_native_grid(wave):
    """Reject native grids without a neighboring wavelength for bin widths."""
    meta = SimpleNamespace(wave_input=None, nspecchan=None, npixelbins=None,
                           wave_min=1.0, wave_max=1.4,
                           wave_low=None, wave_hi=None)

    with pytest.raises(ValueError, match='at least two wavelength samples'):
        s4_genLC.compute_wavelength_bins(meta, np.array(wave), RecordingLog())


@pytest.mark.parametrize('wave_min,expected_wave,expected_low,expected_hi', [
    pytest.param(1.0, [1.0, 1.4], [0.8, 1.2], [1.2, 1.6], id='both-samples'),
    pytest.param(1.4, [1.4], [1.2], [1.6], id='one-selected-sample'),
])
def test_compute_wavelength_bins_uses_two_sample_native_grid(
        wave_min, expected_wave, expected_low, expected_hi):
    """Use both input samples for widths even when only one is selected."""
    meta = SimpleNamespace(wave_input=None, nspecchan=None, npixelbins=None,
                           wave_min=wave_min, wave_max=1.4,
                           wave_low=None, wave_hi=None)

    result = s4_genLC.compute_wavelength_bins(
        meta, np.array([1.0, 1.4]), RecordingLog()
    )

    np.testing.assert_allclose(result.wave, expected_wave)
    np.testing.assert_allclose(result.wave_low, expected_low)
    np.testing.assert_allclose(result.wave_hi, expected_hi)
    assert result.nspecchan == len(expected_wave)


def test_compute_wavelength_bins_allows_single_sample_with_explicit_edges():
    """Explicit edges define widths independently of input sample count."""
    meta = SimpleNamespace(wave_input=None, nspecchan=1, npixelbins=None,
                           wave_min=1.0, wave_max=1.0,
                           wave_low=[0.9], wave_hi=[1.1])

    result = s4_genLC.compute_wavelength_bins(
        meta, np.array([1.0]), RecordingLog()
    )

    np.testing.assert_allclose(result.wave, [1.0])
    np.testing.assert_allclose(result.wave_low, [0.9])
    np.testing.assert_allclose(result.wave_hi, [1.1])
    assert result.nspecchan == 1


def test_compute_wavelength_bins_respects_explicit_edges_and_warnings():
    """Explicit S4 wavelength edges should override mismatched nspecchan."""
    log = RecordingLog()
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
    log = RecordingLog()
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
        s4_genLC.compute_wavelength_bins(
            meta, np.array([1.0, 1.1]), RecordingLog()
        )


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
