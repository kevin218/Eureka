# Last Updated: 2022-04-05

import sys
import os
from importlib import reload
import time as time_pkg

import numpy as np

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import COMMON_IMPORTS, pathdirectory
import eureka.lib.plots
# try:
#     from eureka.S2_calibrations import s2_calibrate as s2
# except ModuleNotFoundError:
#     pass
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
from eureka.S5_lightcurve_fitting import s5_fit as s5
from eureka.S6_planet_spectra import s6_spectra as s6


def test_NIRCam(capsys):
    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
              "are\nincluded in demo ecf files and documentation "
              "(docs/source/ecf.rst).")
        print("\nNIRCam S3-6 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'NIRCam'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}NIRCam_ecfs{os.sep}'

    reload(s3)
    reload(s4)
    reload(s5)
    reload(s6)

    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)
    s5_meta = s5.fitlc(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)
    s6_meta = s6.plot_spectra(meta.eventlabel, ecf_path=ecf_path, 
                              s5_meta=s5_meta)

    # run assertions for S3
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}'
                          f'Stage3{os.sep}')
    name = pathdirectory(meta, 'S3', 1, ap=8, bg=12)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')
    
    s3_cites = np.union1d(COMMON_IMPORTS[2], ["nircam"])
    assert np.array_equal(s3_meta.citations, s3_cites)

    # run assertions for S4
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}'
                          f'Stage4{os.sep}')
    name = pathdirectory(meta, 'S4', 1, ap=8, bg=12)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s4_cites = np.union1d(s3_cites, COMMON_IMPORTS[3])
    assert np.array_equal(s4_meta.citations, s4_cites)

    # run assertions for S5
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}'
                          f'Stage5{os.sep}')
    name = pathdirectory(meta, 'S5', 1, ap=8, bg=12)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s5_cites = np.union1d(s4_cites, COMMON_IMPORTS[4] + 
                          ["emcee", "dynesty", "batman"])
    assert np.array_equal(s5_meta.citations, s5_cites)

    # run assertions for S6
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}'
                          f'Stage6{os.sep}')
    name = pathdirectory(meta, 'S6', 1, ap=8, bg=12)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s6_cites = np.union1d(s5_cites, COMMON_IMPORTS[5])
    assert np.array_equal(s6_meta.citations, s6_cites)

    # remove temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}Stage4")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}Stage5")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}NIRCam{os.sep}Stage6")
