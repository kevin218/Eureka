import sys
import os
from importlib import reload
import time as time_pkg

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
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

    eventlabel = 'NIRSpec'
    ecf_path = f'.{os.sep}S3opt_ecfs{os.sep}'

    reload(s2)
    s2.calibrateJWST(eventlabel, ecf_path=ecf_path)
    reload(s3opt)
    # Stages 3 and 4 optimization
    s3opt_meta, history, best = s3opt.wrapper(eventlabel,
                                              ecf_path=ecf_path,
                                              initial_run=True)

    # run assertions for S3opt
    dirname = (f'data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}'
               f'Stage3opt{os.sep}opt_ECFs{os.sep}')
    assert os.path.exists(dirname + 'S3_NIRSpec.ecf')
    assert os.path.exists(dirname + 'S4_NIRSpec.ecf')

    # remove temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRSpec{os.sep}Stage3opt")
