import numpy as np
import sys
import os

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib import util
from eureka.lib.readECF import MetaClass
from eureka.lib.medstddev import medstddev
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
