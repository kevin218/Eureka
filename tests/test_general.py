import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
import astraeus.xarrayIO as xrio
from astropy.io import fits

from eureka.lib import readECF, util
from eureka.lib.medstddev import medstddev
from eureka.lib.readECF import MetaClass
from eureka.optimizer import S1opt_optimizer, objective_funcs
from eureka.S3_data_reduction import miri, nircam, niriss, nirspec


@pytest.mark.parametrize('instrument, observatory, context, previous', [
    ('nircam', 'jwst', 'jwst_1584.pmap', 'hst_1367.pmap'),
    ('wfc3', 'hst', 'hst_1367.pmap', 'jwst_1584.pmap'),
])
def test_instrument_switch_clears_foreign_crds_context(
        monkeypatch, instrument, observatory, context, previous):
    monkeypatch.setenv('CRDS_SERVER_URL',
                       f'https://{observatory}-crds.stsci.edu')
    monkeypatch.setenv('CRDS_CONTEXT', previous)

    def get_context_name(requested):
        assert requested == observatory
        assert 'CRDS_CONTEXT' not in os.environ
        return context

    monkeypatch.setattr(readECF.crds, 'get_context_name', get_context_name)
    meta = MetaClass()
    meta.inst = instrument

    assert os.environ['CRDS_CONTEXT'] == context
    assert meta.pmap == context.split('_')[1].split('.')[0]


@pytest.mark.parametrize('instrument, observatory, pmap', [
    ('nircam', 'jwst', '1584'), ('wfc3', 'hst', '1367'),
])
def test_explicit_pmap_does_not_query_crds(monkeypatch, instrument,
                                         observatory, pmap):
    context = f'{observatory}_{pmap}.pmap'
    monkeypatch.setenv('CRDS_SERVER_URL',
                       f'https://{observatory}-crds.stsci.edu')
    monkeypatch.setenv('CRDS_CONTEXT', context)

    def unexpected_lookup(requested):
        pytest.fail('A saved pmap must not require a CRDS context lookup.')

    monkeypatch.setattr(readECF.crds, 'get_context_name', unexpected_lookup)
    meta = MetaClass(pmap=pmap)
    meta.inst = instrument

    assert meta.pmap == pmap
    assert os.environ['CRDS_CONTEXT'] == context


@pytest.mark.parametrize('instrument, filter_name, photometry, '
                         'poly_wavelength, wavelength_mode', [
    (nirspec, 'F290LP', False, False, 'present'),
    (miri, 'LRS', False, False, 'present'),
    (miri, 'LRS', False, False, 'missing'),
    (miri, 'LRS', False, False, 'zeros'),
    (miri, 'F560W', True, False, 'present'),
    (nircam, 'F444W', False, False, 'present'),
    (nircam, 'F444W', False, True, 'present'),
    (nircam, 'F322W2', False, True, 'present'),
    (nircam, 'F210M', True, False, 'present'),
    (niriss, 'CLEAR', False, False, 'present'),
], ids=['nirspec', 'miri-spectroscopy', 'miri-missing-wave',
        'miri-zero-wave', 'miri-photometry',
        'nircam-spectroscopy', 'nircam-poly-f444w', 'nircam-poly-f322w2',
        'nircam-photometry', 'niriss'])
def test_jwst_reader_releases_fits_input(tmp_path, monkeypatch, instrument,
                                        filter_name, photometry,
                                        poly_wavelength, wavelength_mode):
    """Returned science arrays must not keep the FITS source locked."""
    path = tmp_path / 'input_calints.fits'
    science = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    primary = fits.PrimaryHDU()
    if instrument is nirspec:
        primary.header['GRATING'] = 'G395H'
    primary.header['FILTER'] = filter_name
    if instrument is nircam:
        primary.header['CHANNEL'] = 'SHORT' if photometry else 'LONG'
    primary.header['INTSTART'] = 1
    primary.header['INTEND'] = 2
    sci = fits.ImageHDU(science, name='SCI')
    sci.header['BUNIT'] = 'DN/s'
    sci.header['DISPAXIS'] = 2 if instrument is miri else 1
    times = np.array([1., 2.])
    hdus = [primary, sci,
            fits.ImageHDU(np.ones_like(science), name='ERR'),
            fits.ImageHDU(np.zeros(science.shape, dtype=np.int32), name='DQ'),
            fits.ImageHDU(np.ones_like(science), name='VAR_RNOISE'),
            fits.ImageHDU(np.ones((3, 4)), name='WAVELENGTH'),
            fits.BinTableHDU.from_columns([
                fits.Column(name='int_mid_BJD_TDB', format='D', array=times)],
                name='INT_TIMES')]
    if wavelength_mode == 'missing':
        hdus = [hdu for hdu in hdus if hdu.name != 'WAVELENGTH']
    elif wavelength_mode == 'zeros':
        hdus[5].data[:] = 0
    fits.HDUList(hdus).writeto(path)
    monkeypatch.setattr(miri, 'wave_MIRI_jwst',
                        lambda *args: np.full((3, 4), 5.0))
    meta = SimpleNamespace(expand=1, time_file=None, firstFile=False,
                           photometry=photometry,
                           poly_wavelength=poly_wavelength,
                           wave_pixel_offset=None,
                           orders=[1, 2], all_orders=[1, 2])

    data, _, _ = instrument.read(str(path), xrio.makeDataset(), meta, None)
    moved_path = path.with_name('released.fits')
    path.rename(moved_path)

    expected = science
    if instrument is miri and not photometry:
        expected = np.swapaxes(expected, 1, 2)[:, :, ::-1]
    elif instrument is niriss:
        expected = np.repeat(expected[..., np.newaxis], 2, axis=3)
    np.testing.assert_array_equal(data.flux.values, expected)
    np.testing.assert_array_equal(data.err.values, np.ones_like(expected))
    np.testing.assert_array_equal(data.dq.values, np.zeros_like(expected))
    np.testing.assert_array_equal(data.v0.values, np.ones_like(expected))
    np.testing.assert_array_equal(data.time.values, times)
    if wavelength_mode != 'present':
        np.testing.assert_array_equal(data.wave_2d.values,
                                      np.full((4, 3), 5.0))


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
