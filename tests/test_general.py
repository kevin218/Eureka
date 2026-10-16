import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
import astraeus.xarrayIO as xrio
from astropy.io import fits

from eureka.lib import apphot, clipping, util
from eureka.lib.medstddev import medstddev
from eureka.lib.readECF import MetaClass
from eureka.optimizer import S1opt_optimizer, objective_funcs


def test_trim(capsys):
    # eureka.lib.util.trim test

    # Let's trim by giving metadata some xwindow and ywindow information
    # which is normally given by the user in the S3_ecf
    trim_x0 = 10
    trim_x1 = 90
    trim_y0 = 2
    trim_y1 = 14

    meta = MetaClass()
    meta.inst = 'nircam'
    nt = 7
    ny = 20
    nx = 100
    # Let's assume we have a dataset with 7 integrations and every spectrum has
    # the dimensions of 100x20
    flux = np.ones((nt, ny, nx))
    time = np.arange(nt)
    flux_units = 'electrons'
    time_units = 'timeless'
    data = xrio.makeDataset()
    data['flux'] = xrio.makeFluxLikeDA(flux, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(flux, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(flux, time, flux_units, time_units,
                                     name='dq')

    meta.ywindow = [trim_y0, trim_y1]
    meta.xwindow = [trim_x0, trim_x1]

    res_dat, res_md = util.trim(data, meta)

    # Let's check if the dimensions agree
    assert res_dat.flux.shape == (nt, (trim_y1 - trim_y0),
                                  (trim_x1 - trim_x0))


def test_add_meta_to_xarray_serializes_range_attrs(tmp_path):
    meta = SimpleNamespace(params={
        'sweep_jump_rejection_threshold': range(4, 15),
        'none_value': None,
        'bibliography': [['citation one', 'citation two']],
        'string_array': np.array(['one', 'two']),
        'scalar_value': 1.5,
    })
    data = xrio.makeDataset()

    util.add_meta_to_xarray(meta, data)

    assert data.attrs['sweep_jump_rejection_threshold'] == list(range(4, 15))
    assert data.attrs['none_value'] == 'None'
    assert data.attrs['bibliography'] == [
        'citation one_ENDOFCITATION_citation two']
    assert data.attrs['string_array'] == ['one', 'two']
    assert data.attrs['scalar_value'] == 1.5
    assert xrio.writeXR(str(tmp_path / 'metadata'), data, verbose=False)


def test_s1opt_best_params_are_json_serializable():
    best = {
        'sweep_jump_rejection_threshold': np.int64(4),
        'nested_values': [np.float64(1.5), np.bool_(True)],
        'array_values': np.array([np.int32(2), np.int32(3)]),
    }

    converted = S1opt_optimizer._convert_to_native_types(best)

    assert converted == {
        'sweep_jump_rejection_threshold': 4,
        'nested_values': [1.5, True],
        'array_values': [2, 3],
    }
    assert json.loads(json.dumps(converted)) == converted


def test_medstddev(capsys):
    # eureka.lib.util.medstddev.medstddev test
    a = np.array([1, 3, 4, 5, 6, 7, 7])
    std, med = medstddev(a, medi=True)
    np.testing.assert_allclose((std, med), (2.2146697055682827, 5.0))

    # use masks
    mask = np.array([False, False, False, True, True, True, True])
    std, med = medstddev(a, mask, medi=True)
    np.testing.assert_allclose((std, med), (1.5275252316519468, 3.0))

    # automatically mask invalid values
    a = np.array([np.nan, 1, 4, np.inf, 6])
    std, med = medstddev(a, medi=True)
    np.testing.assert_allclose((std, med), (2.5166114784235836, 4.0))

    # critical cases:
    # only one value, return std = 0.0
    a = np.array([1, 4, 6])
    mask = np.array([True, True, False])
    std, med = medstddev(a, mask, medi=True)
    assert std == 0.0
    assert med == 6.0

    # no good values, return std = nan, med = nan
    mask[-1] = True
    std, med = medstddev(a, mask, medi=True)
    assert np.isnan(std)
    assert np.isnan(med)


def test_readfiles_accepts_inputdir_without_trailing_separator(tmp_path):
    inputdir = tmp_path / 'Uncalibrated'
    inputdir.mkdir()
    filename = inputdir / (
        'jw00000000001_00001_00001-seg001_mirimage_uncal.fits'
    )

    hdu = fits.PrimaryHDU()
    hdu.header['INSTRUME'] = 'MIRI'
    hdu.header['DETECTOR'] = 'MIRIMAGE'
    hdu.header['FILTER'] = 'F1500W'
    hdu.header['EXP_TYPE'] = 'MIR_IMAGE'
    hdu.writeto(filename)

    meta = SimpleNamespace(inputdir=str(inputdir), suffix='uncal',
                           firstSegOnly_S1=True, firstSegOnly_S3=False,
                           filename='S1opt_test.ecf')

    meta = util.readfiles(meta)

    assert meta.num_data_files == 1
    assert meta.segment_list[0] == str(filename)
    assert meta.inst == 'miri'


def test_optimizer_cleanup_failure_does_not_discard_fitness(monkeypatch):
    meta = SimpleNamespace(delete_intermediate=True, scaling_MAED_spec=0.01,
                           scaling_MAED_white=1.0,
                           opt_param_name='skip_rscd',
                           eventlabel='test')
    s1_meta = SimpleNamespace(outputdir='Stage1')
    s2_meta = SimpleNamespace(outputdir='Stage2')
    s3_meta = SimpleNamespace(outputdir='Stage3')
    s4_meta = SimpleNamespace(outputdir='Stage4')

    def fake_genlc(eventlabel, input_meta, s3_meta):
        input_meta.maed_s4 = 100.0
        input_meta.maed_s4_binned = [10.0]
        return None, None, input_meta

    def failing_rmtree(outputdir):
        if outputdir == 'Stage3':
            raise OSError(objective_funcs.errno.ENOTEMPTY,
                          'Directory not empty', outputdir)

    monkeypatch.setattr(objective_funcs.s1, 'rampfitJWST',
                        lambda eventlabel, input_meta: input_meta)
    monkeypatch.setattr(objective_funcs.s2, 'calibrateJWST',
                        lambda eventlabel, input_meta, s1_meta: input_meta)
    monkeypatch.setattr(objective_funcs.s3, 'reduce',
                        lambda eventlabel, input_meta, s2_meta:
                        (None, input_meta))
    monkeypatch.setattr(objective_funcs.s4, 'genlc', fake_genlc)
    monkeypatch.setattr(objective_funcs.shutil, 'rmtree', failing_rmtree)
    monkeypatch.setattr(objective_funcs.time, 'sleep', lambda delay: None)

    fitness = objective_funcs.single(False, meta, stage=1, s1_meta=s1_meta,
                                     s2_meta=s2_meta, s3_meta=s3_meta,
                                     s4_meta=s4_meta)

    assert fitness == 11.0


def test_optimizer_cleanup_retries_transient_enotempty(monkeypatch):
    meta = SimpleNamespace(delete_intermediate=True, scaling_MAED_spec=0.01,
                           scaling_MAED_white=1.0,
                           opt_param_name='skip_rscd',
                           eventlabel='test')
    s1_meta = SimpleNamespace(outputdir='Stage1')
    s2_meta = SimpleNamespace(outputdir='Stage2')
    s3_meta = SimpleNamespace(outputdir='Stage3')
    s4_meta = SimpleNamespace(outputdir='Stage4')
    attempts = {'Stage3': 0}

    def fake_genlc(eventlabel, input_meta, s3_meta):
        input_meta.maed_s4 = 100.0
        input_meta.maed_s4_binned = [10.0]
        return None, None, input_meta

    def transient_rmtree(outputdir):
        if outputdir == 'Stage3':
            attempts['Stage3'] += 1
            if attempts['Stage3'] == 1:
                raise OSError(objective_funcs.errno.ENOTEMPTY,
                              'Directory not empty', outputdir)

    monkeypatch.setattr(objective_funcs.s1, 'rampfitJWST',
                        lambda eventlabel, input_meta: input_meta)
    monkeypatch.setattr(objective_funcs.s2, 'calibrateJWST',
                        lambda eventlabel, input_meta, s1_meta: input_meta)
    monkeypatch.setattr(objective_funcs.s3, 'reduce',
                        lambda eventlabel, input_meta, s2_meta:
                        (None, input_meta))
    monkeypatch.setattr(objective_funcs.s4, 'genlc', fake_genlc)
    monkeypatch.setattr(objective_funcs.shutil, 'rmtree', transient_rmtree)
    monkeypatch.setattr(objective_funcs.os.path, 'exists',
                        lambda outputdir: outputdir == 'Stage3')
    monkeypatch.setattr(objective_funcs.os.path, 'isdir',
                        lambda outputdir: False)
    monkeypatch.setattr(objective_funcs.time, 'sleep', lambda delay: None)

    fitness = objective_funcs.single(False, meta, stage=1, s1_meta=s1_meta,
                                     s2_meta=s2_meta, s3_meta=s3_meta,
                                     s4_meta=s4_meta)

    assert fitness == 11.0
    assert attempts['Stage3'] == 2


def test_remove_output_directory_retries_transient_ebusy(monkeypatch):
    attempts = {'output': 0}

    def transient_rmtree(outputdir):
        attempts[outputdir] += 1
        if attempts[outputdir] == 1:
            raise OSError(objective_funcs.errno.EBUSY,
                          'Device or resource busy', outputdir)

    monkeypatch.setattr(objective_funcs.shutil, 'rmtree', transient_rmtree)
    monkeypatch.setattr(objective_funcs.os.path, 'exists',
                        lambda outputdir: True)
    monkeypatch.setattr(objective_funcs.os.path, 'isdir',
                        lambda outputdir: False)
    monkeypatch.setattr(objective_funcs.time, 'sleep', lambda delay: None)

    removed = objective_funcs._remove_output_directory('output')

    assert removed
    assert attempts['output'] == 2


def test_remove_output_directory_rmdirs_empty_after_rmtree_error(monkeypatch):
    calls = {'rmtree': 0, 'rmdir': 0}

    def failing_rmtree(outputdir):
        calls['rmtree'] += 1
        raise OSError(objective_funcs.errno.EBUSY,
                      'Device or resource busy', outputdir)

    def successful_rmdir(outputdir):
        calls['rmdir'] += 1

    monkeypatch.setattr(objective_funcs.shutil, 'rmtree', failing_rmtree)
    monkeypatch.setattr(objective_funcs.os.path, 'exists',
                        lambda outputdir: True)
    monkeypatch.setattr(objective_funcs.os.path, 'isdir',
                        lambda outputdir: True)
    monkeypatch.setattr(objective_funcs.os, 'listdir',
                        lambda outputdir: [])
    monkeypatch.setattr(objective_funcs.os, 'rmdir', successful_rmdir)
    monkeypatch.setattr(objective_funcs.time, 'sleep', lambda delay: None)

    removed = objective_funcs._remove_output_directory('output')

    assert removed
    assert calls == {'rmtree': 1, 'rmdir': 1}


def test_remove_output_directory_rmdirs_after_final_wait(monkeypatch):
    calls = {'rmtree': 0, 'rmdir': 0}

    def failing_rmtree(outputdir):
        calls['rmtree'] += 1
        raise OSError(objective_funcs.errno.EBUSY,
                      'Device or resource busy', outputdir)

    def delayed_rmdir(outputdir):
        calls['rmdir'] += 1
        if calls['rmdir'] == 1:
            raise OSError(objective_funcs.errno.EBUSY,
                          'Device or resource busy', outputdir)

    monkeypatch.setattr(objective_funcs, '_RMTREE_ATTEMPTS', 1)
    monkeypatch.setattr(objective_funcs.shutil, 'rmtree', failing_rmtree)
    monkeypatch.setattr(objective_funcs.os.path, 'exists',
                        lambda outputdir: True)
    monkeypatch.setattr(objective_funcs.os.path, 'isdir',
                        lambda outputdir: True)
    monkeypatch.setattr(objective_funcs.os, 'listdir',
                        lambda outputdir: [])
    monkeypatch.setattr(objective_funcs.os, 'rmdir', delayed_rmdir)
    monkeypatch.setattr(objective_funcs.time, 'sleep', lambda delay: None)

    removed = objective_funcs._remove_output_directory('output')

    assert removed
    assert calls == {'rmtree': 1, 'rmdir': 2}


def test_rolling_median_ignores_masked_values_and_handles_boundaries():
    data = np.ma.masked_array([1., 2., 100., np.nan, 5., 6., 7.],
                              mask=[0, 0, 1, 0, 0, 0, 0])

    # Masked and NaN values are ignored inside each window
    np.testing.assert_allclose(
        clipping.rolling_median(data, 3, boundary=None),
        [1.5, 1.5, 2., 5., 5.5, 6., 6.5])
    # Edge handling follows the documented boundary options
    np.testing.assert_allclose(
        clipping.rolling_median(data, 3, boundary='extend')[[0, -1]], [1, 7])
    np.testing.assert_allclose(
        clipping.rolling_median(data, 3, boundary='wrap')[[0, -1]], [2, 6])
    np.testing.assert_allclose(
        clipping.rolling_median(data, 3, boundary='fill',
                                fill_value=0.)[[0, -1]], [1, 6])
    # Even widths are rounded up to the next odd width
    np.testing.assert_allclose(clipping.rolling_median(data, 2),
                               clipping.rolling_median(data, 3))


def test_clip_outliers_restores_neighbours_of_strong_outlier():
    # A single strong outlier used to bias the box-car moving mean, causing
    # good neighbouring points to be permanently clipped as well
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None)
    rng = np.random.default_rng(42)
    data = 1 + rng.normal(0, 6e-4, 2000)
    data[1000] -= 0.13
    mask = np.zeros(data.size, dtype=bool)
    mask[10] = True

    _, outliers, noutliers = clipping.clip_outliers(
        data, log, 15., mask=mask, sigma=6, box_width=33, maxiters=20,
        boundary='fill', fill_value='mask')

    # Only the true outlier is clipped, and input-masked points remain masked
    assert np.array_equal(np.where(outliers)[0], [10, 1000])
    assert noutliers == 2


def test_photutils_apphot_uses_interpolated_pixels():
    # Interpolated bad pixels should still count towards the source flux
    # rather than simply being dropped from the source aperture
    ny, nx, nt = 40, 40, 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    star = 1e4*np.exp(-((xx-20.3)**2 + (yy-19.6)**2)/(2*1.8**2)) + 50.
    flux = np.repeat(star[None], nt, axis=0)
    mask = np.zeros(flux.shape, dtype=bool)
    mask[1, 19:21, 19:21] = True
    flux[mask] = np.nan
    err = np.sqrt(np.abs(flux))

    time = np.arange(nt, dtype=float)
    data = xrio.makeDataset()
    for name, arr in [('flux', flux), ('err', err), ('mask', mask)]:
        data[name] = xrio.makeFluxLikeDA(arr, time, 'electrons', 'BMJD',
                                         name=name)
    data = util.phot_arrays(data)
    data.centroid_x.values[:] = 20.3
    data.centroid_y.values[:] = 19.6

    meta = SimpleNamespace(moving_centroid=False, photap=6, photap_b=None,
                           aperture_shape='circle', skyin=9, skyout=15,
                           aperture_edge='exact', skip_apphot_bg=False,
                           interp_method='cubic', ncpu=1, verbose=False)
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None)

    data = util.interp_masked(data, meta, log)
    assert np.array_equal(data.interp_mask.values, mask)
    for i in range(nt):
        data = apphot.photutils_apphot(data, meta, i)

    assert data.nappix.values[1] == data.nappix.values[0]
    assert np.all(np.isfinite(data.aperr.values))
    np.testing.assert_allclose(data.aplev.values[1], data.aplev.values[0],
                               rtol=0.02)
