import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.constants import arcsec

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S3_data_reduction import (
    background, bright2flux, optspex, sigrej, source_pos, straighten,
)
from eureka.lib import util


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
        data, meta, _Log(), m=0, integ=None
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

    result = background.BGsubtraction(data, meta, _Log(), m=0)

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

    clean = optspex.get_clean(data, meta, _Log(), medflux, mederr)

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
    np.testing.assert_allclose(result.v0.values, 3*factor)


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

    result = bright2flux.dn2electrons(data, meta, _Log())

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

    result = bright2flux.dn2electrons(data, meta, _Log())

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

