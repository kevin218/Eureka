import numpy as np
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib import util
from eureka.lib.readECF import MetaClass
from eureka.lib.medstddev import medstddev
from eureka.optimizer import objective_funcs
from astropy.io import fits
import astraeus.xarrayIO as xrio


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
                           isopt_S1=True, isopt_S3=False,
                           filename='S1opt_test.ecf')

    meta = util.readfiles(meta)

    assert meta.num_data_files == 1
    assert meta.segment_list[0] == str(filename)
    assert meta.inst == 'miri'


def test_optimizer_cleanup_failure_does_not_discard_fitness(monkeypatch):
    meta = SimpleNamespace(delete_intermediate=True, scaling_MAD_spec=0.01,
                           scaling_MAD_white=1.0,
                           opt_param_name='skip_firstframe',
                           eventlabel='test')
    s1_meta = SimpleNamespace(outputdir='Stage1')
    s2_meta = SimpleNamespace(outputdir='Stage2')
    s3_meta = SimpleNamespace(outputdir='Stage3')
    s4_meta = SimpleNamespace(outputdir='Stage4')

    def fake_genlc(eventlabel, input_meta, s3_meta):
        input_meta.mad_s4 = 100.0
        input_meta.mad_s4_binned = [10.0]
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
    meta = SimpleNamespace(delete_intermediate=True, scaling_MAD_spec=0.01,
                           scaling_MAD_white=1.0,
                           opt_param_name='skip_firstframe',
                           eventlabel='test')
    s1_meta = SimpleNamespace(outputdir='Stage1')
    s2_meta = SimpleNamespace(outputdir='Stage2')
    s3_meta = SimpleNamespace(outputdir='Stage3')
    s4_meta = SimpleNamespace(outputdir='Stage4')
    attempts = {'Stage3': 0}

    def fake_genlc(eventlabel, input_meta, s3_meta):
        input_meta.mad_s4 = 100.0
        input_meta.mad_s4_binned = [10.0]
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
    monkeypatch.setattr(objective_funcs.time, 'sleep', lambda delay: None)

    removed = objective_funcs._remove_output_directory('output')

    assert removed
    assert attempts['output'] == 2
