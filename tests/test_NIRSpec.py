import sys
import os
from importlib import reload
import time as time_pkg

import numpy as np

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import COMMON_IMPORTS, pathdirectory
import eureka.lib.plots
from eureka.S2_calibrations import s2_calibrate as s2
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
from eureka.S5_lightcurve_fitting import s5_fit as s5


def test_NIRSpec(capsys):
    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nNIRSpec S2-5 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'NIRSpec'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}NIRSpec_ecfs{os.sep}'

    reload(s2)
    reload(s3)
    reload(s4)
    reload(s5)
    s2_meta = s2.calibrateJWST(meta.eventlabel, ecf_path=ecf_path)

    s2_cites = np.union1d(COMMON_IMPORTS[1], ["nirspec"])
    assert np.array_equal(s2_meta.citations, s2_cites)

    s3_cites = np.union1d(s2_cites, COMMON_IMPORTS[2])

    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path,
                                 s2_meta=s2_meta)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)
    s5_meta = s5.fitlc(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)

    # run assertions for S2
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
                          f'Stage2{os.sep}')
    name = pathdirectory(meta, 'S2', 1,
                         old_datetime=s2_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S3
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
                          f'Stage3{os.sep}')
    name = pathdirectory(meta, 'S3', 1, ap=5, bg=10,
                         old_datetime=s3_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    assert np.array_equal(s3_meta.citations, s3_cites)

    # run assertions for S4
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
                          f'Stage4{os.sep}')
    name = pathdirectory(meta, 'S4', 1, ap=5, bg=10,
                         old_datetime=s4_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s4_cites = np.union1d(s3_cites, COMMON_IMPORTS[3])
    assert np.array_equal(s4_meta.citations, s4_cites)

    # run assertions for S5
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
                          f'Stage5{os.sep}')
    name = pathdirectory(meta, 'S5', 1, ap=5, bg=10,
                         old_datetime=s5_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s5_cites = np.union1d(s4_cites, COMMON_IMPORTS[4] + ["batman"])
    assert np.array_equal(s5_meta.citations, s5_cites)

    # remove temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage2")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage4")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage5")
