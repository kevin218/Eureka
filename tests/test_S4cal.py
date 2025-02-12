import sys
import os
from importlib import reload
import time as time_pkg

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
import eureka.lib.plots
from eureka.S2_calibrations import s2_calibrate as s2
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4cal_StellarSpectra import s4cal_StellarSpec as s4cal


def test_S4cal(capsys):
    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nCalibrated Stellar Spectra S2-4 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'NIRSpec'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}S4cal_ecfs{os.sep}'

    reload(s2)
    s2_meta = s2.calibrateJWST(meta.eventlabel, ecf_path=ecf_path)
    reload(s3)
    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path,
                                 s2_meta=s2_meta)
    reload(s4cal)
    s4_meta, s4_spec, s4_ds = s4cal.medianCalSpec(meta.eventlabel,
                                                  ecf_path=ecf_path,
                                                  s3_meta=s3_meta)

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
    name = pathdirectory(meta, 'S3', 1, ap=5, bg=6,
                         old_datetime=s3_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S4cal
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
                          f'Stage4cal{os.sep}')
    name = pathdirectory(meta, 'S4cal', 1,
                         old_datetime=s4_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # remove temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage2")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage4cal")
