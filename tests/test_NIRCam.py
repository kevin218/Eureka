# Last Updated: 2022-04-05

import numpy as np
import sys, os
from importlib import reload

sys.path.insert(0, '../')
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
try:
    from eureka.S2_calibrations import s2_calibrate as s2
except ModuleNotFoundError as e:
    pass
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
from eureka.S5_lightcurve_fitting import s5_fit as s5
from eureka.S6_planet_spectra import s6_spectra as s6

def test_NIRCam(capsys):
    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
              "included in demo ecf files and documentation (docs/source/ecf.rst)")
        print("\nNIRCam S3-6 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel='NIRCam'
    meta.topdir='../tests'
    ecf_path='./NIRCam_ecfs/'

    reload(s3)
    reload(s4)
    reload(s5)
    reload(s6)
    s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path)
    s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path, s3_meta=s3_meta)
    s5_meta = s5.fitlc(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)
    s6_meta = s6.plot_spectra(meta.eventlabel, ecf_path=ecf_path, s5_meta=s5_meta)

    # run assertions for S3
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage3/'
    name = pathdirectory(meta, 'S3', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S4
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage4/'
    name = pathdirectory(meta, 'S4', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S5
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage5/'
    name = pathdirectory(meta, 'S5', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S6
    meta.outputdir_raw='data/JWST-Sim/NIRCam/Stage6/'
    name = pathdirectory(meta, 'S6', 1, ap=20, bg=20)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # remove temporary files
    os.system("rm -r data/JWST-Sim/NIRCam/Stage3")   
    os.system("rm -r data/JWST-Sim/NIRCam/Stage4")
    os.system("rm -r data/JWST-Sim/NIRCam/Stage5")
    os.system("rm -r data/JWST-Sim/NIRCam/Stage6")
