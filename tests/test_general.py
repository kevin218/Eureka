# Created by: Giannina Guzman-Caloca
# Last Updated: 2021-02-18
# Last Updated By: Yoni Brande


import numpy as np
import sys
sys.path.insert(0, '../')
from eureka.lib import util
from eureka.S2_calibrations import s2_calibrate as s2
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
from eureka.S5_lightcurve_fitting import s5_fit as s5
from eureka.lib.util import pathdirectory
import pytest


class MetaClass:
  def __init__(self):
    return

class DataClass:
  def __init__(self):
    return

meta= MetaClass()
data = DataClass()

######################

#Let's assume we have a dataset with 7 integrations and every spectrum has the dimensions of 100x20
n = 7
ny = 20
nx = 100

data.data = np.ones((n, ny, nx))
data.err = np.ones((n, ny, nx))
data.dq = np.ones((n, ny, nx))
data.wave = np.ones((n, ny, nx))
data.v0 = np.ones((n, ny, nx))


def test_b2f():
    #Let's trim by giving metadata some xwindow and ywindow information which is normally given by the user in the S3_ecf
    trim_x0 = 10
    trim_x1 = 90
    trim_y0 = 2
    trim_y1 = 14

    meta.ywindow = [trim_y0,trim_y1]
    meta.xwindow = [trim_x0,trim_x1]

    res_dat, res_md = util.trim(data, meta)

    #Let's check if the dimensions agree
    assert res_dat.subdata.shape == (n, (trim_y1 - trim_y0), (trim_x1 - trim_x0))

#######################################################################

import sys, os, time
from importlib import reload

def test_NIRCam(capsys):
    print(os.system("pwd"))
    # is able to display any message without failing a test
    # useful to leave messages for future users who run the tests
    with capsys.disabled():
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
              "included in demo ecf files and documentation (docs/source/ecf.rst)\n")
        print("NIRCam test:")

    # explicitly define meta variables to be able to run pathdirectory fn locally
    meta.eventlabel='NIRCam'
    meta.topdir='../tests'
    ecf_path='./NIRCam_ecfs/'

    # run S3 and S4
    reload(s3)
    reload(s4)
    reload(s5)
    s3_meta = s3.reduceJWST(meta.eventlabel, ecf_path=ecf_path)
    s4_meta = s4.lcJWST(meta.eventlabel, ecf_path=ecf_path, s3_meta=s3_meta)
    s5_meta = s5.fitJWST(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)

    # run assertions for S3
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage3/'
    name = pathdirectory(meta, 'S3', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S4
    # NOTE::  check if we want to include aperture info in S4 file output names''
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage4/'
    name = pathdirectory(meta, 'S4', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S5
    # NOTE::  check if we want to include aperture info in S4 file output names''
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage5/'
    name = pathdirectory(meta, 'S5', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # remove temporary files
    os.system("rm -r data/JWST-Sim/NIRCam/Stage3/*")
    os.system("rm -r data/JWST-Sim/NIRCam/Stage4/*")
    os.system("rm -r data/JWST-Sim/NIRCam/Stage5/*")

def test_NIRSpec(capsys): # NOTE:: doesn't work, see issues in github (array mismatch)

    # is able to display any message without failing a test
    # useful to leave messages for future users who run the tests
    with capsys.disabled():
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
              "included in demo ecf files and documentation (docs/source/ecf.rst)\n")
        print("NIRSpec test:")

    # explicitly define meta variables to be able to run pathdirectory fn locally
    meta.eventlabel='NIRSpec'
    meta.topdir='../tests'
    ecf_path='./NIRSpec_ecfs/'

    # run stage 3 and 4
    reload(s2)
    reload(s3)
    reload(s4)
    reload(s5)
    s2_meta = s2.calibrateJWST(meta.eventlabel, ecf_path=ecf_path)
    s3_meta = s3.reduceJWST(meta.eventlabel, ecf_path=ecf_path, s2_meta=s2_meta)
    s4_meta = s4.lcJWST(meta.eventlabel, ecf_path=ecf_path, s3_meta=s3_meta)
    s5_meta = s5.fitJWST(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)

    
    # assert stage 2 outputs
    meta.outputdir_raw='/data/JWST-Sim/NIRSpec/Stage2/'
    name = pathdirectory(meta, 'S2', 1)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')
    
    # assert stage 3 outputs
    meta.outputdir_raw='/data/JWST-Sim/NIRSpec/Stage3/'
    name = pathdirectory(meta, 'S3', 1, ap=8, bg=10)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')
    
    # assert stage 4 outputs
    meta.outputdir_raw='/data/JWST-Sim/NIRSpec/Stage4/'
    name = pathdirectory(meta, 'S4', 1, ap=8, bg=10)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')
    
    # assert stage 4 outputs
    meta.outputdir_raw='/data/JWST-Sim/NIRSpec/Stage5/'
    name = pathdirectory(meta, 'S5', 1, ap=8, bg=10)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')
    
    # remove temp files
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage2/*")
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage3/*")
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage4/*")
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage5/*")
    
'''
def test_MIRI(capsys): # NOTE:: still not implemented

    # is able to display any message without failing a test
    # useful to leave messages for future users who run the tests
    with capsys.disabled():
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
              "included in demo ecf files and documentation (docs/source/ecf.rst)\n")
        print("MIRI test:")

    meta.eventlabel='MIRI'
    meta.topdir='../tests'

    # run S3 and S4
    reload(s3)
    reload(s4)
    s3_meta = s3.reduceJWST(meta.eventlabel)
    s4_meta = s4.lcJWST(meta.eventlabel, s3_meta=s3_meta)

    # assert Stage 3 outputs
    meta.outputdir_raw='/data/JWST-Sim/MIRI/Stage3/'
    name = pathdirectory(meta, 'S3', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # assert Stage 4 outputs
    meta.outputdir_raw='/data/JWST-Sim/MIRI/Stage4/'
    name = pathdirectory(meta, 'S4', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # remove temp files
    os.system("rm -r data/JWST-Sim/MIRI/Stage3/*")
    os.system("rm -r data/JWST-Sim/MIRI/Stage4/*")
'''
