# Last Updated: 2022-04-05

import sys
import os
from importlib import reload
import time as time_pkg

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
import eureka.lib.plots
try:
    from eureka.S1_detector_processing import s1_process as s1
    from eureka.S2_calibrations import s2_calibrate as s2
except ModuleNotFoundError:
    pass
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
from eureka.S5_lightcurve_fitting import s5_fit as s5
from eureka.S6_planet_spectra import s6_spectra as s6


def test_MIRI(capsys):
    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.pdf')

    s2_installed = 'eureka.S2_calibrations.s2_calibrate' in sys.modules
    if not s2_installed:
        with capsys.disabled():
            print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
                  "are\nincluded in demo ecf files and documentation "
                  "(docs/source/ecf.rst).\nSkipping MIRI Stage 2 test as "
                  "could not import eureka.S2_calibrations.s2_calibrate.")
            print("\nMIRI S3-6 test: ", end='', flush=True)
    else:
        with capsys.disabled():
            # is able to display any message without failing a test
            # useful to leave messages for future users who run the tests
            print("\n\nIMPORTANT: Make sure that any changes to the ecf files "
                  "are\nincluded in demo ecf files and documentation "
                  "(docs/source/ecf.rst).")
            print("\nMIRI S2-6 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run
    # pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel = 'MIRI'
    meta.datetime = time_pkg.strftime('%Y-%m-%d')
    meta.topdir = f'..{os.sep}tests'
    ecf_path = f'.{os.sep}MIRI_ecfs{os.sep}'

    if s2_installed:
        # Only run S1-2 stuff if jwst package has been installed
        # reload(s1)
        reload(s2)
    reload(s3)
    reload(s4)
    reload(s5)
    reload(s6)
    if s2_installed:
        # Only run S1-2 stuff if jwst package has been installed
        # s1_meta = s1.rampfitJWST(meta.eventlabel, ecf_path=ecf_path)
        s2_meta = s2.calibrateJWST(meta.eventlabel, ecf_path=ecf_path)
    else:
        s2_meta = None
    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path,
                                 s2_meta=s2_meta)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)
    s5_meta = s5.fitlc(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)
    s6.plot_spectra(meta.eventlabel, ecf_path=ecf_path, s5_meta=s5_meta)

    # run assertions for S2
    if s2_installed:
        # Only run S1-2 stuff if jwst package has been installed
        # meta.outputdir_raw=f'{os.sep}data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage1{os.sep}'
        # name = pathdirectory(meta, 'S1', 1)
        # assert os.path.exists(name)

        meta.outputdir_raw = (f'{os.sep}data{os.sep}JWST-Sim{os.sep}MIRI'
                              f'{os.sep}Stage2{os.sep}')
        name = pathdirectory(meta, 'S2', 1)
        assert os.path.exists(name)
        assert os.path.exists(name+os.sep+'figs')

    # run assertions for S3
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage3{os.sep}')
    name = pathdirectory(meta, 'S3', 1, ap=2, bg=4)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S4
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage4{os.sep}')
    name = pathdirectory(meta, 'S4', 1, ap=2, bg=4)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S5
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage5{os.sep}')
    name = pathdirectory(meta, 'S5', 1, ap=2, bg=4)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S6
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage6{os.sep}')
    name = pathdirectory(meta, 'S6', 1, ap=2, bg=4)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # remove temporary files
    if s2_installed:
        # os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}"
        #           f"Stage1{os.sep}S1_*")
        os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}"
                  f"Stage2{os.sep}S2_*")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage3")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage4")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage5")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage6")
