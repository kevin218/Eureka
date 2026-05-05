import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S5_lightcurve_fitting import models
from eureka.lib import (
    clipping, gaussian, imageedit, interp2d, meanerr, naninterp1d, readECF,
    smooth, sort_nicely, util,
)
from eureka.lib.readEPF import Parameters
from eureka.lib.split_channels import get_trim, split


class _Log:
    def __init__(self):
        self.messages = []

    def writelog(self, message, **kwargs):
        self.messages.append(message)


def test_gaussian_1d_matches_analytic_pdf_and_background():
    """Pin the 1D Gaussian normalization and polynomial background terms."""
    x = np.array([-1.0, 0.0, 1.0])
    width = 2.0
    center = 0.5
    bgpars = [0.25, 0.0, 1.5]

    result = gaussian.gaussian(x, width=width, center=center, bgpars=bgpars)
    expected = (
        np.exp(-0.5*((x-center)/width)**2)/(width*np.sqrt(2*np.pi))
        + bgpars[0]*x + bgpars[2]
    )

    np.testing.assert_allclose(result, expected)


def test_gaussian_2d_uses_independent_widths_centers_and_plane_background():
    """Check 2D Gaussian axes and planar background dimensions."""
    yx = np.indices((3, 4), dtype=float)
    width = np.array([1.5, 2.0])
    center = np.array([1.0, 2.0])
    height = 3.0
    bgpars = [0.1, -0.2, 5.0]

    result = gaussian.gaussian(
        yx, width=width, center=center, height=height, bgpars=bgpars
    )
    exponent = ((yx[0]-center[0])/width[0])**2
    exponent += ((yx[1]-center[1])/width[1])**2
    expected = height*np.exp(-0.5*exponent)
    expected += yx[0]*bgpars[0] + yx[1]*bgpars[1] + bgpars[2]

    np.testing.assert_allclose(result, expected)


def test_meanerr_returns_weighted_mean_uncertainty_and_status_bits():
    """Weighted means should ignore bad data and report status bits."""
    data = np.array([10.0, 12.0, np.nan, 30.0])
    derr = np.array([1.0, 2.0, 1.0, 0.0])
    mask = np.array([False, False, False, True])

    mean, err, status = meanerr.meanerr(
        data, derr, mask=mask, err=True, status=True
    )

    weights = np.array([1/1.0**2, 1/2.0**2])
    expected_mean = np.average(data[:2], weights=weights)
    expected_err = np.sqrt(1/np.sum(weights))
    np.testing.assert_allclose((mean, err), (expected_mean, expected_err))
    assert status == 7


def test_naninterp1d_interpolates_edges_and_all_nan_fallback():
    """NaN interpolation should fill edges and all-NaN arrays."""
    data = np.array([np.nan, 2.0, np.nan, 6.0, np.nan])

    result = naninterp1d.naninterp1d(data.copy())

    np.testing.assert_allclose(result, [2.0, 2.0, 4.0, 6.0, 6.0])
    np.testing.assert_allclose(
        naninterp1d.naninterp1d(np.array([np.nan, np.nan]), replace_val=-1),
        [-1, -1],
    )


def test_sort_nicely_orders_embedded_integers_in_place():
    """Natural sorting should order filenames by embedded integer values."""
    filenames = ['spec10.fits', 'spec2.fits', 'spec1.fits', 'spec11.fits']

    returned = sort_nicely.sort_nicely(filenames)

    assert returned is filenames
    assert filenames == ['spec1.fits', 'spec2.fits', 'spec10.fits',
                         'spec11.fits']


def test_split_channels_returns_channel_slices():
    """Channel splitting should follow cumulative integration counts."""
    first = np.arange(10)
    second = first + 100

    assert get_trim([3, 4, 3], 1) == (3, 7)
    split_first, split_second = split([first, second], [3, 4, 3], 1)

    np.testing.assert_array_equal(split_first, [3, 4, 5, 6])
    np.testing.assert_array_equal(split_second, [103, 104, 105, 106])


def test_binData_masks_bad_values_and_scales_mean_error():
    """Binning should ignore bad values and scale mean errors consistently."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 10.0, 14.0])

    result = util.binData(data, nbin=2, err=True)

    expected = np.ma.array([np.mean([1.0, 2.0])/np.sqrt(3),
                            np.mean([4.0, 10.0, 14.0])/np.sqrt(3)])
    np.testing.assert_allclose(result, expected)


def test_normalize_spectrum_preserves_masks_and_scales_errors():
    """Spectrum normalization should preserve masks while scaling errors."""
    meta = SimpleNamespace(inst='nircam')
    optspec = np.array([[2.0, 4.0], [4.0, 8.0], [6.0, np.nan]])
    opterr = np.full_like(optspec, 0.2)
    optmask = np.zeros_like(optspec, dtype=bool)
    optmask[1, 0] = True

    normspec, normerr = util.normalize_spectrum(
        meta, optspec, opterr=opterr, optmask=optmask
    )

    expected_norm = np.ma.array(
        [[0.5, 2/3], [1.0, 4/3], [1.5, np.nan]],
        mask=[[False, False], [True, False], [False, True]],
    )
    expected_err = np.ma.array(
        [[0.05, 1/30], [0.05, 1/30], [0.05, 1/30]],
        mask=expected_norm.mask,
    )
    np.testing.assert_allclose(normspec, expected_norm)
    np.testing.assert_allclose(normerr, expected_err)
    np.testing.assert_array_equal(np.ma.getmaskarray(normspec),
                                  expected_norm.mask)


def test_get_mad_1d_uses_median_absolute_first_difference_in_ppm():
    """MAD estimates use first differences and return ppm-scale values."""
    data = np.ma.array([1.0, 1.1, 1.4, 1.45, 1.95])

    assert util.get_mad_1d(data, ind_min=1, ind_max=5) == pytest.approx(300000)


def test_smooth_flat_window_matches_manual_convolution_result_length():
    """Flat smoothing should match the explicit padded convolution."""
    x = np.arange(1.0, 10.0)
    window_len = 5
    padded = np.r_[
        2*np.ma.median(x[0:window_len//5])-x[window_len:1:-1],
        x,
        2*np.ma.median(x[-window_len//5:])-x[-1:-window_len:-1],
    ]
    expected = np.ma.convolve(
        np.ones(window_len)/window_len, padded, mode='same'
    )[window_len-1:-window_len+1]

    result = smooth.smooth(x, window_len=window_len, window='flat')

    np.testing.assert_allclose(result, expected)
    assert result.shape == x.shape


def test_polynomial_model_evaluates_centered_time_coefficients():
    """The polynomial systematic is evaluated around each channel midpoint."""
    params = Parameters()
    params.c0 = 1.0, 'free'
    params.c1 = 2.0, 'free'
    params.c2 = -0.5, 'free'
    model = models.PolynomialModel(parameters=params, nchannel=1)
    model.time = np.array([0.0, 1.0, 2.0])

    result = model.eval()

    local_time = np.array([-1.0, 0.0, 1.0])
    expected = 1.0 + 2.0*local_time - 0.5*local_time**2
    np.testing.assert_allclose(result, expected)


def test_polynomial_model_keeps_time_mask_in_output():
    """Masked time samples should remain masked in polynomial evaluations."""
    params = Parameters()
    params.c0 = 1.0, 'free'
    params.c1 = 2.0, 'free'
    model = models.PolynomialModel(parameters=params, nchannel=1)
    model.time = np.ma.array([0.0, np.nan, 2.0], mask=[False, True, False])

    result = model.eval()

    np.testing.assert_array_equal(np.ma.getmaskarray(result),
                                  [False, True, False])


def test_clip_outliers_masks_or_replaces_known_time_series_outlier():
    """Outlier clipping should flag spikes and replacement modes."""
    data = np.ones(21)
    data[10] = 10.0
    log = _Log()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyUserWarning)
        clipped, outliers, noutliers = clipping.clip_outliers(
            data, log, wavelength=2.0, sigma=2, box_width=3, maxiters=3,
            fill_value='mask'
        )

    assert noutliers == 3
    expected_outliers = np.zeros(21, dtype=bool)
    expected_outliers[9:12] = True
    np.testing.assert_array_equal(outliers, expected_outliers)
    np.testing.assert_array_equal(np.ma.getmaskarray(clipped), outliers)
    assert 'Identified 3 outliers' in log.messages[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyUserWarning)
        filled, outliers, noutliers = clipping.clip_outliers(
            data, _Log(), wavelength=2.0, sigma=2, box_width=3, maxiters=3,
            fill_value=-99
        )
    assert noutliers == 3
    assert not np.ma.is_masked(filled[3])
    assert filled[10] == -99


def test_trimimage_pads_out_of_bounds_pixels_and_carries_mask_uncertainty():
    """Image trimming should preserve in-bounds pixels and mask padding."""
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0] = True
    uncd = np.ones_like(data, dtype=float)
    uncd[0:2, 0:3] = 5

    subim, submask, subunc = imageedit.trimimage(
        data, c=(0, 1), r=(1, 2), mask=mask, uncd=uncd
    )

    expected = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 1, 2, 3],
                         [0, 5, 6, 7, 8]], dtype=float)
    np.testing.assert_allclose(subim, expected)
    np.testing.assert_array_equal(submask[0], np.ones(5, dtype=bool))
    assert submask[1, 1]
    assert np.all(subunc[0] == 5)


def test_pasteimage_handles_partial_overlap_at_image_edge():
    """Pasting should only write the overlapping part of a subimage."""
    data = np.zeros((5, 5), dtype=int)
    subim = np.arange(9).reshape(3, 3)

    result = imageedit.pasteimage(data, subim, dy_=(1, 4), syx=(1, 1))

    expected = np.zeros((5, 5), dtype=int)
    expected[0:3, 3:5] = subim[:, 0:2]
    np.testing.assert_array_equal(result, expected)


def test_interp2d_oversamples_plane_and_preserves_original_pixel_values():
    """2D interpolation should preserve a plane at original pixel locations."""
    image = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = interp2d.interp2d(image, expand=3)

    assert result.shape == (4, 4)
    np.testing.assert_allclose(result[0, 0], 1.0)
    np.testing.assert_allclose(result[0, -1], 2.0)
    np.testing.assert_allclose(result[-1, 0], 3.0)
    np.testing.assert_allclose(result[-1, -1], 4.0)
    np.testing.assert_allclose(result[1, 1], 2.0)


def test_metaclass_reads_ecf_values_and_normalizes_paths(tmp_path):
    """ECF parsing should evaluate values and join paths to topdir."""
    ecf = tmp_path/'S3_demo.ecf'
    ecf.write_text(
        'topdir "./root"\n'
        'inputdir "input"\n'
        'outputdir "output"\n'
        'eventlabel "demo"\n'
        'answer 42\n'
        'values [1, 2, 3]\n'
    )

    meta = readECF.MetaClass(folder=str(tmp_path), file='S3_demo.ecf',
                             stage=3)

    assert meta.answer == 42
    assert meta.values == [1, 2, 3]
    assert os.path.normpath(meta.inputdir).endswith(
        os.path.join('root', 'input')
    )
    assert os.path.normpath(meta.outputdir).endswith(
        os.path.join('root', 'output')
    )
    assert os.path.normpath(meta.inputdir_raw) == 'input'
    assert os.path.normpath(meta.outputdir_raw) == 'output'


def test_metaclass_missing_file_requires_kwargs(tmp_path):
    """MetaClass should fail clearly when an ECF file is missing."""
    with pytest.raises(ValueError, match='does not exist'):
        readECF.MetaClass(folder=str(tmp_path), file='missing.ecf')
