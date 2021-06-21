import numpy as np
from ..lib import util


class Metadata():
  def __init__(self):
    return

class Data:
  def __init__(self):
    return

md = Metadata()
dat = Data()

n = 7
ny = 20
nx = 100

dat.data = np.ones((n, ny, nx))
dat.err = np.ones((n, ny, nx))
dat.dq = np.ones((n, ny, nx))
dat.wave = np.ones((n, ny, nx))
dat.v0 = np.ones((n, ny, nx))


def test_b2f():
    trim_x=50
    trim_y=10

    md.ywindow = [0,trim_y]
    md.xwindow = [0,trim_x]

    res_dat, res_md = util.trim(dat, md)
    assert int(np.prod(res_dat.subdata.shape)) == int(n * trim_y * trim_x)
    