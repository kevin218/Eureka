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



######################

#Let's assume we have a dataset with 7 integrations and every spectrum has the dimensions of 100x20
n = 7
ny = 20
nx = 100

dat.data = np.ones((n, ny, nx))
dat.err = np.ones((n, ny, nx))
dat.dq = np.ones((n, ny, nx))
dat.wave = np.ones((n, ny, nx))
dat.v0 = np.ones((n, ny, nx))


def test_b2f():
    #Let's trim by giving metadata some xwindow and ywindow information which is normally given by the user in the S3_ecf
    trim_x0 = 10
    trim_x1 = 90
    trim_y0 = 2
    trim_y1 = 14

    md.ywindow = [trim_y0,trim_y1]
    md.xwindow = [trim_x0,trim_x1]

    res_dat, res_md = util.trim(dat, md)

    #Let's check if the dimensions agree
    assert res_dat.subdata.shape == (n, (trim_y1 - trim_y0), (trim_x1 - trim_x0))


######################
