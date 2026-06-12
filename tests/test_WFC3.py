import sys
import os
from importlib import reload
import time as time_pkg

import numpy as np

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import COMMON_IMPORTS, pathdirectory
import eureka.lib.plots
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
try:
    import image_registration
    imported_image_registration = True
except ModuleNotFoundError:
    imported_image_registration = False


def test_WFC3(capsys):
    if not imported_image_registration:
        raise Exception("HST-relevant packages have not been installed,"
                        " so the WFC3 test is being skipped. You can install "
                        "all HST-related dependencies using "
                        "`pip install eureka-bang[hst]`.")
    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nWFC3 S3-4 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'WFC3'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}WFC3_ecfs{os.sep}'

    reload(s3)
    reload(s4)
    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)

    # run assertions for S3
    meta.outputdir_raw = f'data{os.sep}WFC3{os.sep}Stage3{os.sep}'
    name = pathdirectory(meta, 'S3', s3_meta.run_s3, ap=5, bg=8,
                         old_datetime=s3_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s3_cites = np.union1d(COMMON_IMPORTS[2], ["wfc3"])
    assert np.array_equal(s3_meta.citations, s3_cites)

    # run assertions for S4
    meta.outputdir_raw = f'data{os.sep}WFC3{os.sep}Stage4{os.sep}'
    name = pathdirectory(meta, 'S4', s4_meta.run_s4, ap=5, bg=8,
                         old_datetime=s4_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s4_cites = np.union1d(s3_cites, COMMON_IMPORTS[3])
    assert np.array_equal(s4_meta.citations, s4_cites)

    # remove temporary files
    os.system(f"rm -r data{os.sep}WFC3{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}WFC3{os.sep}Stage4")
