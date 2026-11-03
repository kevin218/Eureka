"""Resource handling when loading Stage 5 posterior samples."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from eureka.S6_planet_spectra import s6_spectra


def sample_meta(tmp_path, nspecchan=1):
    return SimpleNamespace(sharedp=False, nspecchan=nspecchan,
                           inputdir=str(tmp_path) + os.sep, y_param='rp')


def test_load_s5_saves_releases_files(tmp_path):
    values = np.array([0.1, 0.2, 0.3])
    paths = []
    for ch, param in enumerate(('rp', 'other')):
        path = tmp_path / f'S5_emcee_samples_ch{ch}.h5'
        xr.Dataset({param: ('sample', values)}).to_netcdf(
            path, engine='h5netcdf')
        paths.append(path)

    samples = s6_spectra.load_s5_saves(sample_meta(tmp_path, 2), None,
                                      ['emcee'])

    # Windows refuses this rename if an HDF5 reader still holds the file.
    for path in paths:
        path.rename(path.with_suffix('.closed.h5'))
    np.testing.assert_array_equal(samples[0], values)
    np.testing.assert_array_equal(samples[1], np.zeros(len(values)))


def test_load_s5_saves_missing_file(tmp_path):
    samples = s6_spectra.load_s5_saves(sample_meta(tmp_path), None,
                                      ['emcee'], n_samples=3)
    np.testing.assert_array_equal(samples[0], np.zeros(3))


def test_load_s5_saves_closes_on_error(tmp_path, monkeypatch):
    class BrokenDataset:
        variables = {'rp': None}
        closed = False

        def __getitem__(self, key):
            raise RuntimeError('Cannot load samples')

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    ds = BrokenDataset()
    monkeypatch.setattr(s6_spectra.xrio, 'readXR',
                        lambda *args, **kwargs: ds)
    with pytest.raises(RuntimeError, match='Cannot load samples'):
        s6_spectra.load_s5_saves(sample_meta(tmp_path), None, ['emcee'])
    assert ds.closed
