# Last Updated: 2022-04-05

import sys
import os
from importlib import reload
import time as time_pkg

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
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
                        "`pip install .[hst]`.")

    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.pdf')

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
    name = pathdirectory(meta, 'S3', 1, ap=5, bg=5)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # remove temporary files
    os.system(f"rm -r data{os.sep}WFC3{os.sep}Stage3")
