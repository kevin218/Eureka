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
            'coefficients': np.array([3. / order, -0.64, 0.12]),
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
@pytest.mark.parametrize('pwcpos', [245.6, 245.8, 245.85, 246.])
@pytest.mark.parametrize('xoffset', [-2., 2.])
def test_xoffset_uses_reference_pupil_position_and_pivots(
        reference_model, subarray, yoffset, pwcpos, xoffset):
    """Compare translate-then-rotate results to an analytic straight line."""
    model, retrieve = reference_model
    data, meta, log = trace_inputs(subarray=subarray, yoffset=yoffset,
                                   xoffset=xoffset, pwcpos=pwcpos)
    result = niriss.get_wave(data, meta, log)
    x = np.arange(4, 64)
    radians = np.radians(pwcpos - model.meta.pwcpos_cmd)
    c, s = np.cos(radians), np.sin(radians)
    for order in meta.all_orders:
        # A translation by dx before rotation becomes (dx*c, dx*s) after
        # rotation. Evaluate the observed-pupil wavecal at x - dx*c.
        pupil_scaled = (pwcpos - model.meta.pwcpos_cmd + 0.3) / 0.6
        wavelength = (3. / order - 0.01 * (x - xoffset * c)
                      + 0.12 * pupil_scaled)
        intercept = 40. * order - 0.1 * xoffset + (yoffset or 0.)
        if subarray == 'SUBSTRIP96':
            intercept -= 10.
        slope = 0.1
        pivot_x = 20. * order
        pivot_y = 40. * order + 2. + (yoffset or 0.)
        rotated_slope = (s + slope * c) / (c - slope * s)
        expected_y = (pivot_y + rotated_slope * (x - pivot_x)
                      + (intercept + slope * pivot_x - pivot_y)
                      / (c - slope * s))
        np.testing.assert_allclose(result.wave_1d.sel(order=order)[x],
                                   wavelength, atol=1e-10)
        np.testing.assert_allclose(result.trace.sel(order=order)[x],
                                   expected_y, atol=1e-10)
    retrieve.assert_called_once_with()
    model.close.assert_called_once_with()


@pytest.mark.parametrize('subarray', ['SUBSTRIP256', 'SUBSTRIP96'])
@pytest.mark.parametrize('pwcpos', [245.6, 246.])
@pytest.mark.parametrize('xoffset', [-1e-8, 1e-8])
def test_small_xoffset_approaches_jwst_trace(reference_model, subarray,
                                            pwcpos, xoffset):
    """A small shift must not switch geometry or refit the reference trace."""
    model, _ = reference_model
    # Include structure that an eighth-order polynomial cannot reproduce.
    for trace in model.traces:
        trace.trace[:, 1] += 0.02 * np.sin(trace.trace[:, 0])
    data, meta, log = trace_inputs(subarray=subarray, xoffset=xoffset,
                                   pwcpos=pwcpos)
    # Explicit offsets bypass the header-dither noise threshold.
    meta.trace_xoffset = xoffset
    expected = [pastasoss.get_soss_traces(pwcpos, order, subarray, model)
                for order in meta.all_orders]

    result = niriss.get_wave(data, meta, log)

    for order, x, y, wavelength in expected:
        x = x.astype(int)
        np.testing.assert_allclose(result.trace.sel(order=order)[x], y,
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(result.wave_1d.sel(order=order)[x],
                                   wavelength, rtol=0, atol=1e-9)
        outside = ~np.isin(result.x, x)
        assert np.isnan(result.wave_1d.sel(order=order)[outside]).all()
    model.close.assert_called_once_with()


@pytest.mark.parametrize('subarray', ['SUBSTRIP256', 'SUBSTRIP96'])
@pytest.mark.parametrize('ecf_x,ecf_y,expected_x,expected_y', [
    (None, None, 2., 1.5),
    (0., 0., 0., 0.),
    (3., -2., 3., -2.),
    (-3., -2., -3., -2.),
    (0., None, 0., 1.5),
    (None, 0., 2., 0.),
    (1e-8, 1e-8, 1e-8, 1e-8),
    (-1e-8, -1e-8, -1e-8, -1e-8),
])
def test_ecf_offsets_override_header_dithers(
        reference_model, subarray, ecf_x, ecf_y, expected_x, expected_y):
    model, _ = reference_model
    data, meta, log = trace_inputs(subarray=subarray, xoffset=2.,
                                   pwcpos=model.meta.pwcpos_cmd)
    data.attrs['mhdr']['YOFFSET'] = 1.5 * 0.0658
    meta.trace_xoffset = ecf_x
    meta.trace_yoffset = ecf_y

    result = niriss.get_wave(data, meta, log)

    x = np.arange(4, 64)
    subarray_offset = -10. if subarray == 'SUBSTRIP96' else 0.
    for order in meta.all_orders:
        np.testing.assert_allclose(
            result.trace.sel(order=order)[x],
            40. * order + 0.1 * (x - expected_x) + expected_y
            + subarray_offset, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            result.wave_1d.sel(order=order)[x],
            3. / order - 0.01 * (x - expected_x) + 0.06,
            rtol=0, atol=1e-12)
    assert meta.trace_xoffset == ecf_x
    assert meta.trace_yoffset == ecf_y
    model.close.assert_called_once_with()


@pytest.mark.parametrize('subarray', ['SUBSTRIP256', 'SUBSTRIP96'])
@pytest.mark.parametrize('axis,header_offset,expected_offset', [
    ('x', 0.0099, 0.),
    ('x', 0.01, 0.01),
    ('x', 0.0101, 0.0101),
    ('x', -0.0099, 0.),
    ('x', -0.01, -0.01),
    ('x', -0.0101, -0.0101),
    ('y', 0.0099, 0.),
    ('y', 0.01, 0.01),
    ('y', 0.0101, 0.0101),
    ('y', -0.0099, 0.),
    ('y', -0.01, -0.01),
    ('y', -0.0101, -0.0101),
])
def test_header_dither_threshold(reference_model, subarray, axis,
                                 header_offset, expected_offset):
    """Check science coordinates below, at, and above the header cutoff."""
    model, _ = reference_model
    data, meta, log = trace_inputs(subarray=subarray,
                                   pwcpos=model.meta.pwcpos_cmd)
    meta.trace_xoffset = None
    pixel_scale = 0.0653 if axis == 'x' else 0.0658
    data.attrs['mhdr'][f'{axis.upper()}OFFSET'] = header_offset * pixel_scale

    result = niriss.get_wave(data, meta, log)

    x = np.arange(4, 64)
    xoffset = expected_offset if axis == 'x' else 0.
    yoffset = expected_offset if axis == 'y' else 0.
    subarray_offset = -10. if subarray == 'SUBSTRIP96' else 0.
    for order in meta.all_orders:
        np.testing.assert_allclose(
            result.trace.sel(order=order)[x],
            40. * order + 0.1 * (x - xoffset) + yoffset + subarray_offset,
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            result.wave_1d.sel(order=order)[x],
            3. / order - 0.01 * (x - xoffset) + 0.06,
            rtol=0, atol=1e-12)
        outside = ~np.isin(result.x, x)
        assert np.isnan(result.wave_1d.sel(order=order)[outside]).all()
    assert meta.trace_xoffset is None
    assert meta.trace_yoffset is None
    model.close.assert_called_once_with()


@pytest.mark.parametrize('pwcpos', [245.6, 246.])
@pytest.mark.parametrize('xoffset', [-2.3, 2.3])
def test_xoffset_preserves_pupil_dependent_wavecal(reference_model, pwcpos,
                                                  xoffset):
    """Evaluate an independent quadratic wavecal, including mixed terms."""
    model, _ = reference_model
    for wavecal in model.wavecal_models:
        wavecal.coefficients = np.array([
            3. / wavecal.spectral_order, -0.64, 0.12, 0.032, 0.048, -0.024,
        ])
    data, meta, log = trace_inputs(xoffset=xoffset, pwcpos=pwcpos)

    result = niriss.get_wave(data, meta, log)

    x = np.arange(4, 64)
    radians = np.radians(pwcpos - model.meta.pwcpos_cmd)
    u = (x - xoffset * np.cos(radians)) / 64.
    v = (pwcpos - model.meta.pwcpos_cmd + 0.3) / 0.6
    for order in meta.all_orders:
        expected = (3. / order - 0.64 * u + 0.12 * v + 0.032 * u**2
                    + 0.048 * u * v - 0.024 * v**2)
        np.testing.assert_allclose(result.wave_1d.sel(order=order)[x],
                                   expected, rtol=0, atol=1e-12)
    model.close.assert_called_once_with()


@pytest.mark.parametrize('xoffset,last_valid', [(2., 42), (-2., 38)])
def test_xoffset_keeps_columns_aligned_when_rotation_trims(reference_model,
                                                           xoffset,
                                                           last_valid):
    model, _ = reference_model
    data, meta, log = trace_inputs(yoffset=210.95, xoffset=xoffset,
                                   pwcpos=245.8)

    result = niriss.get_wave(data, meta, log)

    # Order 1 extends above row 255 after the user correction; order 2 is
    # entirely outside the detector. Trimmed wavelengths must remain NaN.
    # Put the boundary between columns so interpolation roundoff
    # cannot change whether a point exactly at row 255 is retained.
    valid = np.arange(4, last_valid + 1)
    np.testing.assert_allclose(result.trace.sel(order=1)[valid],
                               250.95 + 0.1 * (valid - xoffset))
    np.testing.assert_allclose(result.wave_1d.sel(order=1)[valid],
                               3. - 0.01 * (valid - xoffset) + 0.06)
    assert np.isnan(result.wave_1d.sel(order=1)[last_valid + 1:]).all()
    assert np.isnan(result.wave_1d.sel(order=2)).all()
    model.close.assert_called_once_with()


@pytest.mark.parametrize('pwcpos', [245.6, 246.])
@pytest.mark.parametrize('xoffset', [-2., 2.])
def test_xoffset_updates_columns_trimmed_before_translation(reference_model,
                                                           pwcpos, xoffset):
    """A translated trace may enter or leave columns at the detector edge."""
    model, _ = reference_model
    trace = next(t for t in model.traces if t.spectral_order == 1)
    trace.trace[:, 1] += 210.95
    trace.pivot_y += 210.95
    _, unshifted_x, _, _ = pastasoss.get_soss_traces(
        pwcpos, 1, refmodel=model)
    data, meta, log = trace_inputs(xoffset=xoffset, pwcpos=pwcpos)

    result = niriss.get_wave(data, meta, log)

    x = np.arange(4, 64)
    angle = np.radians(pwcpos - model.meta.pwcpos_cmd)
    c, s = np.cos(angle), np.sin(angle)
    slope = (s + 0.1 * c) / (c - 0.1 * s)
    # Rotate y = 250.95 + 0.1*(x - xoffset) about (20, 252.95).
    y = 252.95 + slope * (x - 20.) - 0.1 * xoffset / (c - 0.1 * s)
    valid = x[y <= 255.]
    if xoffset > 0:
        assert valid[-1] > unshifted_x[-1]
    else:
        assert valid[-1] < unshifted_x[-1]
    np.testing.assert_allclose(result.trace.sel(x=valid, order=1),
                               y[y <= 255.], rtol=0, atol=1e-10)
    pupil_scaled = (pwcpos - model.meta.pwcpos_cmd + 0.3) / 0.6
    np.testing.assert_allclose(
        result.wave_1d.sel(x=valid, order=1),
        3. - 0.01 * (valid - xoffset * c) + 0.12 * pupil_scaled,
        rtol=0, atol=1e-12)
    outside = ~np.isin(result.x, valid)
    assert np.isnan(result.wave_1d.sel(order=1)[outside]).all()
    model.close.assert_called_once_with()


def test_missing_order_raises_and_closes_reference(reference_model):
    model, _ = reference_model
    data, meta, log = trace_inputs(orders=(1, 3))

    with pytest.raises(ValueError, match='Order 3 is not available'):
        niriss.get_wave(data, meta, log)

    model.close.assert_called_once_with()


@pytest.mark.parametrize('xoffset', [-2., 0., 2.])
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


@pytest.mark.parametrize('legacy_offset', [None, -2., 0., 3.])
def test_niriss_metadata_accepts_legacy_trace_offset(legacy_offset):
    meta = S3MetaClass(topdir='.', spec_hw=17, bg_hw=22, window_len=13,
                       p7thresh=7, trace_offset=legacy_offset)

    meta.set_NIRISS_defaults()

    assert meta.trace_yoffset == legacy_offset
    meta.set_NIRISS_defaults()
    assert meta.trace_yoffset == legacy_offset


@pytest.mark.parametrize('source', ['kwargs', 'ecf'])
@pytest.mark.parametrize('legacy_first', [True, False])
@pytest.mark.parametrize('legacy_offset', [None, 3.])
@pytest.mark.parametrize('yoffset', [None, -2.5, 0.])
def test_niriss_metadata_modern_offset_takes_precedence(
        tmp_path, capsys, source, legacy_first, legacy_offset, yoffset):
    """Keep explicit modern values for either input spelling order."""
    offsets = [('trace_offset', legacy_offset), ('trace_yoffset', yoffset)]
    if not legacy_first:
        offsets.reverse()
    parameters = dict(topdir='.', inputdir='Stage2', outputdir='Stage3',
                      spec_hw=17, bg_hw=22, window_len=13, p7thresh=7)
    parameters.update(offsets)
    if source == 'kwargs':
        meta = S3MetaClass(**parameters)
    else:
        ecf = tmp_path / 'S3_NIRISS.ecf'
        ecf.write_text(''.join(f'{key} {value!r}\n'
                               for key, value in parameters.items()))
        meta = S3MetaClass(folder=str(tmp_path), file=ecf.name)

    meta.set_NIRISS_defaults()

    assert meta.trace_yoffset == yoffset
    assert meta.params['trace_yoffset'] == yoffset
    assert 'renamed to trace_yoffset' in capsys.readouterr().out
    # Stage defaults may be applied again to copied or reused metadata.
    meta.set_NIRISS_defaults()
    assert meta.trace_yoffset == yoffset


def test_niriss_metadata_initializes_trace_offsets():
    meta = S3MetaClass(topdir='.', spec_hw=17, bg_hw=22, window_len=13,
                       p7thresh=7)

    meta.set_NIRISS_defaults()

    assert meta.trace_xoffset is None
    assert meta.trace_yoffset is None
