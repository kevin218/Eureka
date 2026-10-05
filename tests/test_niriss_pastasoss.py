"""Exercise Eureka's trace corrections with JWST's PASTASOSS API."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
from jwst.extract_1d.soss_extract import pastasoss
from stdatamodels.jwst import datamodels

from eureka.S3_data_reduction import niriss
from eureka.S3_data_reduction.s3_meta import S3MetaClass


@pytest.fixture
def reference_model(monkeypatch):
    """Use a real reference model with simple, analytically known traces."""
    model = datamodels.PastasossModel()
    model.meta.pwcpos_cmd = 245.8
    model.meta.pwcpos_bounds = [245.5, 246.1]
    # Reverse the reference order to catch assumptions about list indices.
    for order in [2, 1]:
        x = np.arange(4., 64.)
        model.traces.append({
            'spectral_order': order,
            'trace': np.column_stack((x, 40 * order + 0.1 * x)),
            'pivot_x': 20. * order,
            'pivot_y': 40. * order + 2.,
        })
        model.wavecal_models.append({
            'spectral_order': order,
            'coefficients': np.array([3. / order, -0.64, 0.]),
            'scale_extents': np.array([[0., -0.3], [64., 0.3]]),
        })
    retrieve = Mock(return_value=model)
    monkeypatch.setattr(pastasoss, 'retrieve_default_pastasoss_model',
                        retrieve)
    object.__setattr__(model, 'close', Mock(wraps=model.close))
    yield model, retrieve
    model.close()


def trace_inputs(subarray='SUBSTRIP256', yoffset=None, xoffset=0.,
                 pwcpos=245.85, orders=(1, 2)):
    data = xr.Dataset(coords={'x': np.arange(70), 'order': list(orders)})
    data.attrs['mhdr'] = {'PWCPOS': pwcpos, 'XOFFSET': xoffset * 0.0653,
                          'SUBARRAY': subarray}
    meta = SimpleNamespace(all_orders=list(orders), src_ypos=[30, 80],
                           trace_yoffset=yoffset, verbose=False)
    log = SimpleNamespace(writelog=Mock())
    return data, meta, log


@pytest.mark.parametrize('subarray', ['SUBSTRIP256', 'SUBSTRIP96'])
@pytest.mark.parametrize('yoffset', [None, -2.5, 3.])
def test_jwst_traces_and_additional_yoffset(reference_model, subarray,
                                          yoffset):
    model, retrieve = reference_model
    data, meta, log = trace_inputs(subarray=subarray, yoffset=yoffset)
    expected = [pastasoss.get_soss_traces(245.85, order, subarray, model)
                for order in meta.all_orders]

    result = niriss.get_wave(data, meta, log)

    for order, x, y, wavelength in expected:
        x = x.astype(int)
        np.testing.assert_allclose(result.trace.sel(order=order)[x],
                                   y + (yoffset or 0.))
        np.testing.assert_allclose(result.wave_1d.sel(order=order)[x],
                                   wavelength)
        outside = ~np.isin(result.x, x)
        assert np.isnan(result.wave_1d.sel(order=order)[outside]).all()
        np.testing.assert_allclose(result.trace.sel(order=order)[outside],
                                   meta.src_ypos[order - 1])
    assert result.wave_1d.attrs['wave_units'] == 'microns'
    assert meta.trace_yoffset == yoffset
    retrieve.assert_called_once_with()
    model.close.assert_called_once_with()


@pytest.mark.parametrize('subarray', ['SUBSTRIP256', 'SUBSTRIP96'])
@pytest.mark.parametrize('yoffset', [None, 3.])
def test_xoffset_uses_reference_pupil_position_and_pivots(
        reference_model, subarray, yoffset):
    """Compare translate-then-rotate results to an analytic straight line."""
    model, retrieve = reference_model
    data, meta, log = trace_inputs(subarray=subarray, yoffset=yoffset,
                                   xoffset=2.)
    result = niriss.get_wave(data, meta, log)
    x = np.arange(4, 64)
    radians = np.radians(245.85 - model.meta.pwcpos_cmd)
    c, s = np.cos(radians), np.sin(radians)
    for order in meta.all_orders:
        wavelength = 3. / order - 0.01 * (x - 2.)
        intercept = 40. * order + 30. / order + (yoffset or 0.)
        if subarray == 'SUBSTRIP96':
            intercept -= 10.
        slope = -10.
        pivot_wave = 3. / order - 0.01 * (20. * order)
        pivot_y = 40. * order + 2. + (yoffset or 0.)
        rotated_slope = (s + slope * c) / (c - slope * s)
        expected_y = (pivot_y + rotated_slope * (wavelength - pivot_wave)
                      + (intercept + slope * pivot_wave - pivot_y)
                      / (c - slope * s))
        np.testing.assert_allclose(result.wave_1d.sel(order=order)[x],
                                   wavelength, atol=1e-10)
        np.testing.assert_allclose(result.trace.sel(order=order)[x],
                                   expected_y, atol=1e-10)
    retrieve.assert_called_once_with()
    model.close.assert_called_once_with()


def test_xoffset_keeps_columns_aligned_when_rotation_trims(reference_model):
    model, _ = reference_model
    data, meta, log = trace_inputs(yoffset=211., xoffset=2., pwcpos=245.8)

    result = niriss.get_wave(data, meta, log)

    # Order 1 extends above row 255 after the user correction; order 2 is
    # entirely outside the detector. Trimmed wavelengths must remain NaN.
    valid = np.arange(4, 43)
    np.testing.assert_allclose(result.trace.sel(order=1)[valid],
                               251. + 0.1 * (valid - 2.))
    np.testing.assert_allclose(result.wave_1d.sel(order=1)[valid],
                               3. - 0.01 * (valid - 2.))
    assert np.isnan(result.wave_1d.sel(order=1)[43:]).all()
    assert np.isnan(result.wave_1d.sel(order=2)).all()
    model.close.assert_called_once_with()


def test_missing_order_raises_and_closes_reference(reference_model):
    model, _ = reference_model
    data, meta, log = trace_inputs(orders=(1, 3))

    with pytest.raises(ValueError, match='Order 3 is not available'):
        niriss.get_wave(data, meta, log)

    model.close.assert_called_once_with()


@pytest.mark.parametrize('xoffset', [0., 2.])
def test_invalid_pupil_position_raises_and_closes_reference(reference_model,
                                                           xoffset):
    model, _ = reference_model
    data, meta, log = trace_inputs(pwcpos=247., xoffset=xoffset)

    with pytest.raises(ValueError, match='outside bounds'):
        niriss.get_wave(data, meta, log)

    model.close.assert_called_once_with()


@pytest.mark.parametrize('yoffset', [None, -2.5, 0.])
def test_niriss_metadata_preserves_trace_yoffset(yoffset):
    meta = S3MetaClass(topdir='.', spec_hw=17, bg_hw=22, window_len=13,
                       p7thresh=7,
                       trace_yoffset=yoffset)

    meta.set_NIRISS_defaults()

    assert meta.trace_yoffset == yoffset


def test_niriss_metadata_accepts_legacy_trace_offset():
    meta = S3MetaClass(topdir='.', spec_hw=17, bg_hw=22, window_len=13,
                       p7thresh=7, trace_offset=-2.)

    meta.set_NIRISS_defaults()

    assert meta.trace_yoffset == -2.
