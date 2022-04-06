# Last Updated: 2022-04-05

import numpy as np
import sys, os
from importlib import reload

sys.path.insert(0, '../')
from eureka.lib import util
from eureka.lib.readECF import MetaClass

class DataClass:
  def __init__(self):
    return

def test_trim(capsys):
    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\neureka.lib.util.trim test: ", end='', flush=True)

    #Let's trim by giving metadata some xwindow and ywindow information which is normally given by the user in the S3_ecf
    trim_x0 = 10
    trim_x1 = 90
    trim_y0 = 2
    trim_y1 = 14

    meta = MetaClass()
    data = DataClass()
    n = 7
    ny = 20
    nx = 100
    #Let's assume we have a dataset with 7 integrations and every spectrum has the dimensions of 100x20
    data.data = np.ones((n, ny, nx))
    data.err = np.ones((n, ny, nx))
    data.dq = np.ones((n, ny, nx))
    data.wave = np.ones((n, ny, nx))
    data.v0 = np.ones((n, ny, nx))

    meta.ywindow = [trim_y0,trim_y1]
    meta.xwindow = [trim_x0,trim_x1]

    res_dat, res_md = util.trim(data, meta)

    #Let's check if the dimensions agree
    assert res_dat.subdata.shape == (n, (trim_y1 - trim_y0), (trim_x1 - trim_x0))
