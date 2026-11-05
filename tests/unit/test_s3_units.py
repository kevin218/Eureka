from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.constants import arcsec
from tests.unit.helpers import RecordingLog

from eureka.lib import util
from eureka.S3_data_reduction import (background, bright2flux, nircam, optspex,
                                      sigrej, source_pos, straighten, wfc3)


def _trace_image(center=7.3, sigma=1.2, ny=15, nx=6, background=0.1):
    y = np.arange(ny, dtype=float)
    profile = background + np.exp(-0.5*((y-center)/sigma)**2)
    return np.repeat(profile[:, None], nx, axis=1)


@pytest.mark.parametrize('relative_times, expected', [
    pytest.param([1., 2., 3.], [1., 2., 3.], id='complete'),
    pytest.param([1., 2., np.nan, np.nan, np.nan],
                 [1., 2., 2.1, 2.2, 2.3], id='trailing-padding'),
    pytest.param([np.nan, np.nan, np.nan, 4., 5.],
                 [3.7, 3.8, 3.9, 4., 5.], id='leading-padding'),
    pytest.param([1., np.nan, np.nan, 4.],
                 [1., 2., 3., 4.], id='internal-gap'),
    pytest.param([np.nan, 2., np.nan],
                 [1.9, 2., 2.1], id='single-finite-time'),
    pytest.param([np.nan, 2., np.nan, np.nan, 5., np.nan],
                 [1.9, 2., 3., 4., 5., 5.1], id='mixed-gaps'),
])
def test_wfc3_missing_read_times_are_distinct_and_preserve_finite_values(
        relative_times, expected):
    """Repair consecutive gaps even at the scale of full Julian dates."""
    jd = 2457000. + np.array(relative_times)
    original = jd.copy()

    result = wfc3._fill_missing_read_times(jd, exptime=86400.)

    np.testing.assert_allclose(result, 2457000. + np.array(expected),
                               rtol=0, atol=1e-9)
    np.testing.assert_array_equal(result[np.isfinite(jd)], jd[np.isfinite(jd)])
    np.testing.assert_array_equal(jd, original)
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result) > 0)


def test_wfc3_padded_read_times_allow_differenced_dq_alignment():
    """Three trailing placeholders previously created duplicate DQ indices."""
    time = wfc3._fill_missing_read_times(
        np.array([1., 2., np.nan, np.nan, np.nan]), exptime=86400.
    )
    raw = xr.DataArray(np.zeros((5, 1)), dims=('time', 'x'),
                       coords={'time': time})
    diff_time = time[:-1] + 0.5*np.diff(time)
    differenced = xr.Dataset(coords={'time': diff_time})

    # Exercise the same labeled assignment performed by wfc3.read.
    differenced['dq'] = raw[:-1]

    assert differenced.dq.shape == (4, 1)
    np.testing.assert_array_equal(differenced.time.values, diff_time)


def test_wfc3_missing_read_times_require_a_finite_anchor():
    with pytest.raises(ValueError, match='without a finite timestamp'):
        wfc3._fill_missing_read_times(np.full(3, np.nan), exptime=1.)


def test_source_pos_weighted_recovers_synthetic_trace_centroid_and_width():
    """Validate flux-weighted source centroids on a known Gaussian trace."""
    flux = np.ma.array(_trace_image(center=7.25, sigma=1.1, background=0.0))
    meta = SimpleNamespace(src_pos_type='weighted', spec_hw=4, isplots_S3=0)

    rounded, exact, width, integ = source_pos.source_pos(
        flux, meta, shdr={}, m=0, n=3, plot=False
    )

    assert rounded == 7
    assert exact == pytest.approx(7.25, abs=0.08)
    assert width == pytest.approx(1.1, abs=0.12)
    assert integ == 3


def test_source_pos_gaussian_recovers_synthetic_trace_centroid_and_width():
    """Validate Gaussian-fit source centroids on a known synthetic trace."""
    flux = np.ma.array(_trace_image(center=6.65, sigma=1.35))
    meta = SimpleNamespace(src_pos_type='gaussian', spec_hw=5, isplots_S3=0)

    rounded, exact, width, integ = source_pos.source_pos(
        flux, meta, shdr={}, m=0, n=2, plot=False
    )

    assert rounded == 7
    assert exact == pytest.approx(6.65, abs=0.03)
    assert width == pytest.approx(1.35, abs=0.03)
    assert integ == 2


def test_source_pos_header_subtracts_detector_window_offset():
    """Guard the FITS-header source position convention after window trims."""
    meta = SimpleNamespace(src_pos_type='header', ywindow=[100, 120])

    rounded, exact, width, integ = source_pos.source_pos(
        np.ones((3, 3)), meta, shdr={'SRCYPOS': 112.4}, m=0, n=5,
        plot=False
    )

    assert rounded == 12
    assert exact == pytest.approx(12.4)
    assert width == pytest.approx(0.0)
    assert integ == 5


def test_source_pos_header_requires_srcypos_keyword():
    """Ensure header-based source positions fail clearly without SRCYPOS."""
    meta = SimpleNamespace(src_pos_type='header', ywindow=[0, 10])

    with pytest.raises(AttributeError, match='SRCYPOS'):
        source_pos.source_pos(np.ones((3, 3)), meta, shdr={}, m=0, n=0,
                              plot=False)


def test_source_pos_wrapper_records_centroids_for_all_integrations():
    """Check wrapper bookkeeping for per-integration centroids."""
    flux = np.stack([
        _trace_image(center=6.2, sigma=1.0, background=0.0, nx=4),
        _trace_image(center=8.1, sigma=1.0, background=0.0, nx=4),
    ])
    data = xr.Dataset(
        {
            'flux': (('time', 'y', 'x'), flux),
            'mask': (('time', 'y', 'x'), np.zeros_like(flux, dtype=bool)),
        },
        coords={'time': [0.0, 1.0], 'y': np.arange(15), 'x': np.arange(4)},
        attrs={'shdr': {}},
    )
    meta = SimpleNamespace(src_pos_type='weighted', spec_hw=4, isplots_S3=0,
                           ncpu=1, int_start=0, n_int=2, verbose=False)

    data, _, _ = source_pos.source_pos_wrapper(
        data, meta, RecordingLog(), m=0, integ=None
    )

    np.testing.assert_allclose(data.centroid_y.values, [6.2, 8.1], atol=0.02)
    np.testing.assert_allclose(data.centroid_sy.values, [1.0, 1.0],
                               atol=0.02)
    assert data.centroid_y.attrs['units'] == 'pixels'
    assert data.centroid_sy.attrs['units'] == 'pixels'


def test_sigrej_flags_outlier_and_returns_final_statistics_without_mutating():
    """Protect sigma rejection outputs and the promise not to mutate masks."""
    data = np.array([
        [[1.0], [5.0]],
        [[1.0], [5.0]],
        [[1.0], [5.0]],
        [[1.0], [5.0]],
        [[10.0], [5.0]],
    ])
    input_mask = np.zeros_like(data, dtype=bool)

    mask, ival, fmean, fstddev, fmedian, fmedstddev = sigrej.sigrej(
        data, sigma=[3], mask=input_mask, estsig=[1.0], axis=0,
        ival=True, fmean=True, fstddev=True, fmedian=True, fmedstddev=True
    )

    expected_mask = np.zeros_like(data, dtype=bool)
    expected_mask[-1, 0, 0] = True
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(input_mask, np.zeros_like(input_mask))
    np.testing.assert_allclose(ival[0, 0, :, 0], [1.0, 5.0])
    np.testing.assert_allclose(fmean[:, 0], [1.0, 5.0])
    np.testing.assert_allclose(fstddev[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(fmedian[:, 0], [1.0, 5.0])
    np.testing.assert_allclose(fmedstddev[:, 0], [0.0, 0.0])


def test_sigrej_default_mask_flags_nonfinite_values():
    """Ensure non-finite data are masked without an input mask."""
    data = np.array([[1.0, 2.0], [np.nan, 2.1], [1.1, np.inf]])

    mask = sigrej.sigrej(data, sigma=[5], axis=0)

    np.testing.assert_array_equal(mask, [[False, False],
                                         [True, False],
                                         [False, True]])


def test_nircam_flag_ff_preserves_mask_metadata_and_flags_temporal_outlier():
    """Full-frame rejection must keep the labeled mask usable by Stage 3."""
    flux = np.ones((8, 2, 2))
    flux[-1, 0, 0] = 100
    mask = np.zeros_like(flux, dtype=bool)
    mask[0, 1, 1] = True
    data = xr.Dataset(
        {'flux': (('time', 'y', 'x'), flux),
         'mask': (('time', 'y', 'x'), mask.copy(), {'purpose': 'bad pixels'})},
        coords={'time': np.arange(8), 'y': [10, 11], 'x': [20, 21]},
    )
    original_mask = data.mask.copy(deep=True)
    meta = SimpleNamespace(bg_thresh=[2], verbose=False)
    log = RecordingLog()

    result = nircam.flag_ff(data, meta, log)

    expected_mask = mask.copy()
    expected_mask[-1, 0, 0] = True
    np.testing.assert_array_equal(result.mask.values, expected_mask)
    assert result.mask.dims == original_mask.dims
    assert result.mask.attrs == original_mask.attrs
    xr.testing.assert_identical(result.mask.coords, original_mask.coords)
    np.testing.assert_array_equal(result.flux.values, flux)
    assert log.messages[-1] == '    Flagged 3.125000% of pixels as bad.'


def test_fitbg_recovers_linear_background_while_ignoring_source_region():
    """Verify column background fits exclude the source aperture region."""
    y, x = np.indices((4, 9), dtype=float)
    expected = 10 + 2*y + 0.5*x
    data = expected.copy()
    data[:, 3:6] += 100
    mask = np.zeros_like(data, dtype=bool)
    meta = SimpleNamespace(bg_method='std', outputdir='')

    bg, outmask = background.fitbg(
        data, meta, mask, x1=3, x2=5, deg=1, threshold=5, isplots=0
    )

    np.testing.assert_allclose(bg, expected, atol=1e-12)
    np.testing.assert_array_equal(outmask, mask)


def test_fitbg_masks_background_outlier_before_refitting_polynomial():
    """Check background outliers are rejected before final fitting."""
    y, x = np.indices((4, 20), dtype=float)
    expected = 10 + 2*y + 0.5*x
    data = expected.copy()
    data[:, 8:12] += 100
    data[2, 16] += 200
    mask = np.zeros_like(data, dtype=bool)
    meta = SimpleNamespace(bg_method='median', outputdir='')

    bg, outmask = background.fitbg(
        data, meta, mask, x1=8, x2=11, deg=1, threshold=5, isplots=0
    )

    assert outmask[2, 16]
    np.testing.assert_allclose(bg, expected, atol=1e-12)


def test_fitbg2_uses_explicit_background_mask_regions():
    """Validate complex background masks for orders/source regions."""
    y, x = np.indices((4, 9), dtype=float)
    expected = 10 + 2*y + 0.5*x
    data = expected.copy()
    data[:, 3:6] += 100
    mask = np.zeros_like(data, dtype=bool)
    bgmask = np.zeros_like(data, dtype=bool)
    bgmask[:, 3:6] = True
    meta = SimpleNamespace(bg_method='std', outputdir='')

    bg, returned_bgmask = background.fitbg2(
        data, meta, mask, bgmask, deg=1, threshold=5, isplots=0
    )

    np.testing.assert_allclose(bg, expected, atol=1e-12)
    np.testing.assert_array_equal(returned_bgmask, bgmask)


def test_bgsubtraction_skip_adds_zero_background_without_changing_flux():
    """Ensure skipped background subtraction records a zero bg product."""
    flux = np.ones((2, 3, 4))
    data = xr.Dataset(
        {
            'flux': (('time', 'y', 'x'), flux.copy()),
            'mask': (('time', 'y', 'x'), np.zeros_like(flux, dtype=bool)),
        },
        coords={'time': [0, 1], 'y': np.arange(3), 'x': np.arange(4)},
    )
    data['flux'].attrs['flux_units'] = 'electrons'
    meta = SimpleNamespace(bg_deg=None, skip_bg=False, verbose=False)

    result = background.BGsubtraction(data, meta, RecordingLog(), m=0)

    np.testing.assert_allclose(result.flux.values, flux)
    np.testing.assert_allclose(result.bg.values, np.zeros_like(flux))
    assert result.bg.attrs['flux_units'] == 'electrons'


def test_standard_spectrum_replaces_masked_pixels_with_spectral_neighbors():
    """Verify box extraction repairs masked pixels from neighbors."""
    apdata = np.array([[[1.0, 2.0, 100.0, 4.0, 5.0],
                        [10.0, 20.0, 30.0, 40.0, 50.0]]])
    aperr = np.ones_like(apdata)
    apmask = np.zeros_like(apdata, dtype=bool)
    apmask[0, 0, 2] = True

    stdspec, stdvar = optspex.standard_spectrum(apdata, apmask, aperr)

    cleaned_first_row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_spec = cleaned_first_row + apdata[0, 1]
    np.testing.assert_allclose(stdspec[0], expected_spec)
    np.testing.assert_allclose(stdvar[0], np.full(5, 2.0))


def _wfc3_profile_batch(nreads=3, window_len=1):
    """Two exposures per direction with disjoint source positions per read."""
    n_int, ny, nx = 4*nreads, 3*nreads, 6
    flux = np.zeros((n_int, ny, nx))
    profiles = np.array([[10., 80., 10.], [20., 30., 50.]])
    directions = np.repeat([0, 1, 0, 1], nreads)
    for exposure, scale in enumerate([1., 1., 3., 3.]):
        for read in range(nreads):
            n = exposure*nreads + read
            flux[n, 3*read:3*read+3] = (
                profiles[directions[n], :, None] * scale * (read+1))
    data = xr.Dataset(
        {'flux': (['time', 'y', 'x'], flux),
         'err': (['time', 'y', 'x'], np.sqrt(flux+1)),
         'mask': (['time', 'y', 'x'], np.zeros_like(flux, dtype=bool)),
         'flatmask': (['time', 'y', 'x'], np.zeros_like(flux, dtype=bool)),
         'bg': (['time', 'y', 'x'], np.zeros_like(flux)),
         'v0': (['time', 'y', 'x'], np.ones_like(flux)),
         'medflux': (['y', 'x'], np.median(flux, axis=0)),
         'scandir': (['time'], directions),
         'wave_1d': (['x'], np.linspace(1., 2., nx))},
        coords={'time': np.arange(n_int), 'y': np.arange(ny),
                'x': np.arange(nx)},
    )
    data.flux.attrs.update(flux_units='ELECTRONS', time_units='BJD_TDB')
    data.wave_1d.attrs['wave_units'] = 'microns'
    guesses = np.arange(nreads)*3+1
    meta = SimpleNamespace(
        n_int=n_int, nreads=nreads, spec_hw=1, subnx=nx, verbose=False,
        guess=[SimpleNamespace(values=guesses.copy()) for _ in range(2)],
        fittype='meddata', window_len=window_len, median_thresh=5,
        orders=None, ncpu=1, int_start=0, int_end=0, isplots_S3=0,
        p5thresh=10, p7thresh=10, prof_deg=3, indep_batches=False,
    )
    return data, meta, profiles


@pytest.mark.parametrize('nreads', [1, 3], ids=['flt', 'scanned-ima'])
@pytest.mark.parametrize('window_len', [1, 3], ids=['unsmoothed', 'cleaned'])
def test_wfc3_meddata_preserves_read_profiles_and_recovers_known_flux(
        nreads, window_len):
    """Separate templates preserve source flux for both scan directions."""
    data, meta, profiles = _wfc3_profile_batch(nreads, window_len)
    expected_flux = data.flux.values.sum(axis=1)
    original_median = data.medflux.values.copy()

    aperture = wfc3.cut_aperture(data, meta, RecordingLog())
    apdata, aperr, apmask, apbg, apv0, apmedflux = aperture

    for n in range(meta.n_int):
        read = n % nreads
        direction = data.scandir.values[n]
        expected = np.repeat(
            (2*(read+1)*profiles[direction])[:, None], meta.subnx, axis=1)
        np.testing.assert_allclose(apmedflux[n], expected)
    np.testing.assert_array_equal(data.medflux.values, original_median)

    data = wfc3.standard_spectrum(data, meta, apdata, apmask, aperr)
    data, _, _ = optspex.optimize_wrapper(
        data, meta, RecordingLog(), apdata, apmask, apbg, apv0, apmedflux)

    np.testing.assert_allclose(data.optspec.values, expected_flux)
    assert not data.optmask.values.any()
    assert np.all(np.isfinite(data.opterr.values))
    assert np.all(data.opterr.values > 0)


def test_wfc3_meddata_excludes_padded_masked_and_nonfinite_reads():
    """Missing data must not contaminate another exposure's template."""
    data, meta, profiles = _wfc3_profile_batch()
    # A padded read can become finite after drift correction; its flatmask
    # still identifies it. Its valid counterpart is the third exposure.
    data.flux.values[0] = 1e6
    data.flatmask.values[0] = True
    # A masked outlier in one pixel leaves only its valid counterpart there.
    data.flux.values[3, 1, 0] = 1e6
    data.mask.values[3, 1, 0] = True
    # No usable data at all for direction 0, read position 2.
    data.mask.values[[2, 8]] = True
    # Nonfinite flux or uncertainty must also exclude a read from its group.
    data.flux.values[4] = np.nan
    data.err.values[5] = np.inf

    *_, apmedflux = wfc3.cut_aperture(data, meta, RecordingLog())

    for n in (0, 6):
        np.testing.assert_allclose(apmedflux[n, :, 0], 3*profiles[0])
    for n in (3, 9):
        np.testing.assert_allclose(apmedflux[n, :, 0], [40., 90., 100.])
    for n in (2, 8):
        np.testing.assert_array_equal(apmedflux[n], 0.)
    for n in (4, 10):
        np.testing.assert_allclose(apmedflux[n, :, 0], 6*profiles[1])
    for n in (5, 11):
        np.testing.assert_allclose(apmedflux[n, :, 0], 9*profiles[1])


def test_wfc3_meddata_uses_available_exposure_for_singleton_groups():
    data, meta, _ = _wfc3_profile_batch()
    data = data.isel(time=slice(0, 2*meta.nreads))
    meta.n_int = data.sizes['time']

    apdata, _, _, _, _, apmedflux = wfc3.cut_aperture(
        data, meta, RecordingLog())

    np.testing.assert_allclose(apmedflux, apdata)


@pytest.mark.parametrize('indep_batches', [False, True])
def test_wfc3_meddata_reuses_or_rebuilds_profiles_between_batches(
        indep_batches):
    """The setting controls the actual extraction templates across batches."""
    first, meta, _ = _wfc3_profile_batch()
    meta.indep_batches = indep_batches
    cache = {}
    *_, first_profiles = wfc3.cut_aperture(
        first, meta, RecordingLog(), profile_cache=cache)
    second = first.copy(deep=True)
    # Change the spatial shape, keeping the saved diagnostic median fixed.
    second.flux.values[:, 1::3, :] *= 2

    *_, second_profiles = wfc3.cut_aperture(
        second, meta, RecordingLog(), profile_cache=cache)

    expected = first_profiles.copy()
    if indep_batches:
        expected[:, 1, :] *= 2
        assert cache == {}
    else:
        assert len(cache) == 2*meta.nreads
    np.testing.assert_allclose(second_profiles, expected)


def test_wfc3_meddata_initializes_new_scan_direction_in_later_batch():
    data, meta, profiles = _wfc3_profile_batch()
    first = data.isel(time=np.flatnonzero(data.scandir.values == 0))
    second = data.isel(time=np.flatnonzero(data.scandir.values == 1))
    meta.n_int = first.sizes['time']
    cache = {}
    wfc3.cut_aperture(first, meta, RecordingLog(), profile_cache=cache)
    original = {key: value.copy() for key, value in cache.items()}
    assert set(cache) == {(0, read) for read in range(meta.nreads)}

    *_, new_profiles = wfc3.cut_aperture(
        second, meta, RecordingLog(), profile_cache=cache)

    assert set(cache) == {(direction, read) for direction in (0, 1)
                          for read in range(meta.nreads)}
    for key, value in original.items():
        np.testing.assert_array_equal(cache[key], value)
    for n in range(meta.n_int):
        read = n % meta.nreads
        np.testing.assert_allclose(new_profiles[n, :, 0],
                                   2*(read+1)*profiles[1])


def test_wfc3_meddata_does_not_cache_an_entirely_masked_group():
    valid, meta, profiles = _wfc3_profile_batch()
    first = valid.copy(deep=True)
    first.flatmask.values[[0, 6]] = True
    cache = {}
    *_, first_profiles = wfc3.cut_aperture(
        first, meta, RecordingLog(), profile_cache=cache)
    assert (0, 0) not in cache
    np.testing.assert_array_equal(first_profiles[[0, 6]], 0.)

    *_, valid_profiles = wfc3.cut_aperture(
        valid, meta, RecordingLog(), profile_cache=cache)

    assert (0, 0) in cache
    for n in (0, 6):
        np.testing.assert_allclose(valid_profiles[n, :, 0], 2*profiles[0])


def test_profile_meddata_clips_negative_values_and_normalizes_columns():
    """Profiles used for optimal extraction must be positive and normalized."""
    meddata = np.array([[-1.0, 2.0, 1.0],
                        [3.0, 2.0, 1.0],
                        [1.0, 6.0, 2.0]])

    profile = optspex.profile_meddata(meddata)

    expected = np.array([[0.0, 0.2, 0.25],
                         [0.75, 0.2, 0.25],
                         [0.25, 0.6, 0.5]])
    np.testing.assert_allclose(profile, expected)
    np.testing.assert_allclose(np.ma.sum(profile, axis=0), np.ones(3))


def test_optimize_meddata_recovers_spectrum_and_masks_cosmic_ray():
    """Check optimal extraction recovers flux after masking a cosmic ray."""
    profile = np.array([[0.2, 0.3, 0.25],
                        [0.5, 0.4, 0.5],
                        [0.3, 0.3, 0.25]])
    spectrum = np.array([100.0, 200.0, 300.0])
    subdata = profile*spectrum
    subdata[1, 1] += 200
    mask = np.zeros_like(subdata, dtype=bool)
    meta = SimpleNamespace(isplots_S3=0, int_end=99)

    optspec, opterr, optmask, integ, order = optspex.optimize(
        meta, subdata, mask, bg=np.zeros_like(subdata),
        spectrum=spectrum.copy(), Q=1.0, v0=np.ones_like(subdata),
        p7thresh=5, fittype='meddata', meddata=profile, n=4, order=2
    )

    np.testing.assert_allclose(optspec, spectrum)
    assert np.all(opterr > 0)
    assert optmask[1, 1]
    assert integ == 4
    assert order == 2


def test_get_clean_interpolates_masked_pixels_and_removes_median_outlier():
    """Median-frame cleaning should repair masked and sigma-clipped pixels."""
    data = xr.Dataset(coords={'y': np.arange(2), 'x': np.arange(5)})
    medflux = np.ma.array(
        [[1.0, 2.0, 100.0, 4.0, 5.0],
         [10.0, 11.0, 12.0, 13.0, 14.0]],
        mask=[[False, False, False, False, False],
              [False, False, True, False, False]],
    )
    mederr = np.ones((2, 5))
    meta = SimpleNamespace(window_len=3, median_thresh=2)

    clean = optspex.get_clean(data, meta, RecordingLog(), medflux, mederr)

    np.testing.assert_allclose(clean, [[1, 2, 3, 4, 5],
                                       [10, 11, 12, 13, 14]])


def test_rate2count_uses_effective_integration_time_header():
    """Confirm rate-to-count conversion uses the effective integration time."""
    data = xr.Dataset(
        {
            'flux': (('y', 'x'), np.ones((2, 2))),
            'err': (('y', 'x'), np.full((2, 2), 2.0)),
            'v0': (('y', 'x'), np.full((2, 2), 3.0)),
        },
        attrs={'mhdr': {'EFFINTTM': 5.0}},
    )

    result = bright2flux.rate2count(data)

    np.testing.assert_allclose(result.flux.values, np.full((2, 2), 5.0))
    np.testing.assert_allclose(result.err.values, np.full((2, 2), 10.0))
    np.testing.assert_allclose(result.v0.values, np.full((2, 2), 15.0))


def test_rate2count_requires_time_header():
    """Fail clearly when no exposure-time keyword supports rate conversion."""
    data = xr.Dataset(
        {
            'flux': (('y', 'x'), np.ones((2, 2))),
            'err': (('y', 'x'), np.ones((2, 2))),
            'v0': (('y', 'x'), np.ones((2, 2))),
        },
        attrs={'mhdr': {}},
    )

    with pytest.raises(ValueError, match='No FITS header keys'):
        bright2flux.rate2count(data)


def test_bright2flux_scales_flux_error_and_variance_by_pixel_area():
    """Pin the MJy/sr to Jy/pixel scaling for flux-like arrays."""
    data = xr.Dataset(
        {
            'flux': (('y', 'x'), np.ones((2, 2))),
            'err': (('y', 'x'), np.full((2, 2), 2.0)),
            'v0': (('y', 'x'), np.full((2, 2), 3.0)),
        }
    )
    pixel_area = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor = arcsec**2 * 1e6 * pixel_area

    result = bright2flux.bright2flux(data, pixel_area)

    np.testing.assert_allclose(result.flux.values, factor)
    np.testing.assert_allclose(result.err.values, 2*factor)
    np.testing.assert_allclose(result.v0.values, 3*factor**2)


def test_dn2electrons_scales_flux_error_and_variance_with_scalar_gain():
    """Scalar DN-to-electron gain should square only variance-like arrays."""
    data = xr.Dataset(
        {
            'flux': (('y', 'x'), np.ones((2, 2))),
            'err': (('y', 'x'), np.full((2, 2), 2.0)),
            'v0': (('y', 'x'), np.full((2, 2), 3.0)),
        },
        attrs={'mhdr': {'SUBSTRT1': 1, 'SUBSTRT2': 1, 'SUBSIZE1': 2,
                        'SUBSIZE2': 2},
               'shdr': {'DISPAXIS': 1}},
    )
    meta = SimpleNamespace(gain=4.0, gainfile=None)

    result = bright2flux.dn2electrons(data, meta, RecordingLog())

    np.testing.assert_allclose(result.flux.values, np.full((2, 2), 4.0))
    np.testing.assert_allclose(result.err.values, np.full((2, 2), 8.0))
    np.testing.assert_allclose(result.v0.values, np.full((2, 2), 48.0))


def test_dn2electrons_applies_supersampled_array_gain_window():
    """Array gains should be supersampled and trimmed to the data window."""
    data = xr.Dataset(
        {
            'flux': (('y', 'x'), np.ones((2, 2))),
            'err': (('y', 'x'), np.ones((2, 2))),
            'v0': (('y', 'x'), np.ones((2, 2))),
        },
        attrs={'mhdr': {'SUBSTRT1': 1, 'SUBSTRT2': 1, 'SUBSIZE1': 2,
                        'SUBSIZE2': 2},
               'shdr': {'DISPAXIS': 1}},
    )
    meta = SimpleNamespace(gain=[[2.0, 3.0], [4.0, 5.0]], gainfile=None,
                           expand=1, ywindow=[0, 2], xwindow=[0, 2])

    result = bright2flux.dn2electrons(data, meta, RecordingLog())

    gain = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(result.flux.values, gain)
    np.testing.assert_allclose(result.err.values, gain)
    np.testing.assert_allclose(result.v0.values, gain**2)


def test_interp_masked_helper_replaces_bad_pixels_without_touching_good_ones():
    """Bad-pixel interpolation should only alter masked pixels."""
    flux = np.array([[1.0, 2.0, 3.0],
                     [4.0, 0.0, 6.0],
                     [7.0, 8.0, 9.0]])
    mask = np.zeros_like(flux, dtype=bool)
    mask[1, 1] = True
    grid_x, grid_y = np.mgrid[0:2:complex(0, 3), 0:2:complex(0, 3)]

    interpolated, integ = util.interp_masked_helper(
        flux, mask, grid_x, grid_y, 'linear', i=7
    )

    np.testing.assert_allclose(interpolated[~mask], flux[~mask])
    assert interpolated[1, 1] == pytest.approx(5.0)
    assert integ == 7


def test_find_column_median_shifts_aligns_trace_to_detector_center():
    """Protect trace-straightening shift signs for a curved synthetic trace."""
    ny = 11
    centers = np.array([4]*5 + [5]*5 + [6]*5)
    y = np.arange(ny)[:, None]
    image = np.exp(-0.5*((y-centers[None, :])/0.7)**2)
    meta = SimpleNamespace(calibrated_spectra=False, isplots_S3=0)

    shifts, new_center = straighten.find_column_median_shifts(image, meta, m=0)

    assert new_center == 4
    np.testing.assert_array_equal(shifts, [0]*5 + [-1]*5 + [-2]*5)


def test_roll_columns_applies_per_column_shifts_without_mutating_shifts():
    """Ensure per-column rolling preserves reusable shift arrays."""
    data = np.arange(2*4*3).reshape(2, 4, 3)
    shifts = np.array([[0, 1, -1], [1, -1, 0]])
    original_shifts = shifts.copy()

    rolled = straighten.roll_columns(data, shifts)

    expected = np.stack([
        np.column_stack([
            np.roll(data[0, :, 0], 0),
            np.roll(data[0, :, 1], 1),
            np.roll(data[0, :, 2], -1),
        ]),
        np.column_stack([
            np.roll(data[1, :, 0], 1),
            np.roll(data[1, :, 1], -1),
            np.roll(data[1, :, 2], 0),
        ]),
    ])
    np.testing.assert_array_equal(rolled, expected)
    np.testing.assert_array_equal(shifts, original_shifts)
