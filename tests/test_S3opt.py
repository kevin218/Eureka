import sys
import os
from importlib import reload
import time as time_pkg

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
# import eureka.optimizer.S1opt_optimizer as s1opt
from eureka.S2_calibrations import s2_calibrate as s2
import eureka.optimizer.S3opt_optimizer as s3opt


def test_S3opt(capsys):
    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nS3-S4 Optimizer test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'NIRSpec'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}S3opt_ecfs{os.sep}'

    reload(s2)
    s2_meta = s2.calibrateJWST(meta.eventlabel, ecf_path=ecf_path)
    reload(s3opt)
    # Stages 3 and 4 optimization
    s3opt_meta, history, best = s3opt.wrapper(meta.eventlabel,
                                              ecf_path=ecf_path,
                                              initial_run=True)

    # run assertions for S3opt #FINDME
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
                          f'Stage3{os.sep}')
    name = pathdirectory(meta, 'S3', s3_meta.run_s3, ap=5, bg=6,
                         old_datetime=s3_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S4cal
    # meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
    #                       f'Stage4cal{os.sep}')
    # name = pathdirectory(meta, 'S4cal', s4_meta.run_S4cal,
    #                      old_datetime=s4_meta.datetime)
    # assert os.path.exists(name)
    # assert os.path.exists(name+os.sep+'figs')

    # remove temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage3opt")
