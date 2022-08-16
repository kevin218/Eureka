# Last Updated: 2022-08-16

import sys
import os
from importlib import reload
import time as time_pkg

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
import eureka.lib.plots
# try:
#     from eureka.S2_calibrations import s2_calibrate as s2
# except ModuleNotFoundError:
#     pass
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4


def test_NIRCam(capsys):
    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.pdf')

    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nPhotometry NIRCam S3-4 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'Photometry_NIRCam'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}Photometry_NIRCam_ecfs{os.sep}'

    reload(s3)
    reload(s4)
    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)

    # run assertions for S3
    meta.outputdir_raw = (f'data{os.sep}Photometry{os.sep}NIRCam{os.sep}'
                          f'Stage3{os.sep}')
    name = pathdirectory(meta, 'S3', 1, ap=60, bg=7090)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S4
    meta.outputdir_raw = (f'data{os.sep}Photometry{os.sep}NIRCam{os.sep}'
                          f'Stage4{os.sep}')
    name = pathdirectory(meta, 'S4', 1, ap=60, bg=7090)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # remove temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}Stage4")
