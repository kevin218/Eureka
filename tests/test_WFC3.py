import sys
import os
from importlib import reload
import time as time_pkg
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import COMMON_IMPORTS, pathdirectory
import eureka.lib.plots
from eureka.S3_data_reduction import optspex
from eureka.S3_data_reduction import wfc3
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
try:
    import image_registration
    imported_image_registration = True
except ModuleNotFoundError:
    imported_image_registration = False


def test_WFC3_flt_preparation_sets_nreads_full(monkeypatch):
    class Log:
        def writelog(self, *args, **kwargs):
            pass

    def noop(meta, log):
        return meta, log

    def get_reference_frames(meta, log):
        assert meta.nreads == 1
        assert meta.nreads_full == 1
        return meta, log

    monkeypatch.setattr(wfc3, 'separate_direct', noop)
    monkeypatch.setattr(wfc3, 'separate_scan_direction', noop)
    monkeypatch.setattr(wfc3, 'get_reference_frames', get_reference_frames)
    monkeypatch.setattr(wfc3.hst, 'imageCentroid',
                        lambda *args, **kwargs: np.array([[1., 2.]]))

    meta = SimpleNamespace(segment_list=np.array(['dummy_flt.fits']),
                           direct_list=np.array(['direct_flt.fits']),
                           centroidguess=[1., 2.], centroidtrim=0,
                           ny=1, CRPIX1=0., CRPIX2=0.,
                           postarg1=np.array([0.]),
                           postarg2=np.array([0.]))

    meta, _ = wfc3.preparation_step(meta, Log())

    assert meta.nreads == 1
    assert meta.nreads_full == 1
    assert not meta.photometry


def test_WFC3_cut_aperture_returns_medflux():
    class Log:
        def writelog(self, *args, **kwargs):
            pass

    n_int, ny, nx = 2, 5, 4
    flux = np.arange(n_int*ny*nx, dtype=float).reshape(n_int, ny, nx)
    medflux = np.arange(ny*nx, dtype=float).reshape(ny, nx)
    data = xr.Dataset(coords={'time': np.arange(n_int),
                              'y': np.arange(ny),
                              'x': np.arange(nx)})
    data['flux'] = (['time', 'y', 'x'], flux)
    data['err'] = (['time', 'y', 'x'], np.ones_like(flux))
    data['mask'] = (['time', 'y', 'x'], np.zeros_like(flux, dtype=bool))
    data['bg'] = (['time', 'y', 'x'], np.zeros_like(flux))
    data['v0'] = (['time', 'y', 'x'], np.ones_like(flux))
    data['medflux'] = (['y', 'x'], medflux)
    data['scandir'] = (['time'], np.zeros(n_int, dtype=int))

    meta = SimpleNamespace(n_int=n_int, nreads=1, spec_hw=1, subnx=nx,
                           guess=[SimpleNamespace(values=np.array([2]))],
                           verbose=False)

    _, _, _, _, _, apmedflux = wfc3.cut_aperture(data, meta, Log())

    expected = medflux[1:4]
    assert apmedflux.shape == (n_int, 3, nx)
    assert np.array_equal(apmedflux[0], expected)
    assert np.array_equal(apmedflux[1], expected)


def test_optspex_meddata_requires_median_aperture():
    meta = SimpleNamespace(isplots_S3=0, int_end=0)
    subdata = np.ones((2, 3))
    mask = np.zeros_like(subdata, dtype=bool)

    with pytest.raises(ValueError, match='requires a median flux aperture'):
        optspex.optimize(meta, subdata, mask, np.zeros_like(subdata),
                         np.ones(3), 1, np.ones_like(subdata),
                         fittype='meddata', meddata=None)


def test_WFC3(capsys):
    if not imported_image_registration:
        raise Exception("HST-relevant packages have not been installed,"
                        " so the WFC3 test is being skipped. You can install "
                        "all HST-related dependencies using "
                        "`pip install eureka-bang[hst]`.")
    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nWFC3 S3-4 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'WFC3'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}WFC3_ecfs{os.sep}'

    reload(s3)
    reload(s4)
    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)

    # run assertions for S3
    meta.outputdir_raw = f'data{os.sep}WFC3{os.sep}Stage3{os.sep}'
    name = pathdirectory(meta, 'S3', s3_meta.run_s3, ap=5, bg=8,
                         old_datetime=s3_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s3_cites = np.union1d(COMMON_IMPORTS[2], ["wfc3"])
    assert np.array_equal(s3_meta.citations, s3_cites)

    # run assertions for S4
    meta.outputdir_raw = f'data{os.sep}WFC3{os.sep}Stage4{os.sep}'
    name = pathdirectory(meta, 'S4', s4_meta.run_s4, ap=5, bg=8,
                         old_datetime=s4_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s4_cites = np.union1d(s3_cites, COMMON_IMPORTS[3])
    assert np.array_equal(s4_meta.citations, s4_cites)

    # remove temporary files
    os.system(f"rm -r data{os.sep}WFC3{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}WFC3{os.sep}Stage4")
