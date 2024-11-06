# Last Updated: 2024-02-15

import sys
import os
from importlib import reload
import time as time_pkg
from copy import deepcopy

import numpy as np

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib.readECF import MetaClass
from eureka.lib.util import COMMON_IMPORTS, pathdirectory
from eureka.lib.citations import CITATIONS
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

try:
    from eureka.S5_lightcurve_fitting import differentiable_models
    pymc3_installed = True
except ModuleNotFoundError:
    pymc3_installed = False


def test_MIRI(capsys):
    # Set up some parameters to make plots look nicer.
    # You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

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

        s2_cites = np.union1d(COMMON_IMPORTS[1], ["miri"])
        assert np.array_equal(s2_meta.citations, s2_cites)
        s3_cites = np.union1d(s2_cites, COMMON_IMPORTS[2])
    else:
        s2_meta = None
        s3_cites = np.union1d(COMMON_IMPORTS[2], ["miri"])

    s3_spec, s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path,
                                 s2_meta=s2_meta)
    s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, ecf_path=ecf_path,
                                       s3_meta=s3_meta)
    s5_meta = s5.fitlc(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)

    # Test differentiable models if pymc3 related dependencies are installed
    if pymc3_installed:
        # Copy the S5 meta and manually edit some settings
        s5_meta2 = deepcopy(s5_meta)
        s5_meta2.fit_method = '[exoplanet,nuts]'
        s5_meta2.run_myfuncs = s5_meta2.run_myfuncs.replace(
            'fleck_tr,batman_ecl,sinusoid_pc', 'starry')
        s5_meta2.fit_par = './s5_fit_par_starry.epf'
        s5_meta2.tune = 10
        s5_meta2.draws = 100
        s5_meta2.chains = 1
        s5_meta2.target_accept = 0.5
        s5_meta2.isplots_S5 = 3
        # Reset the citations list
        s5_meta2.citations = s4_meta.citations
        s5_meta2.bibliography = [CITATIONS[entry] for entry
                                 in s5_meta2.citations]
        # Run S5 with the new parameters
        s5_meta2 = s5.fitlc(meta.eventlabel, s4_meta=s4_meta,
                            input_meta=s5_meta2)

    s6_meta = s6.plot_spectra(meta.eventlabel, ecf_path=ecf_path,
                              s5_meta=s5_meta)

    # run assertions for S2
    if s2_installed:
        # Only run S1-2 stuff if jwst package has been installed
        # meta.outputdir_raw=f'{os.sep}data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage1{os.sep}'
        # name = pathdirectory(meta, 'S1', 1,
        #                      old_datetime=s1_meta.datetime)
        # assert os.path.exists(name)

        meta.outputdir_raw = (f'{os.sep}data{os.sep}JWST-Sim{os.sep}MIRI'
                              f'{os.sep}Stage2{os.sep}')
        name = pathdirectory(meta, 'S2', 1,
                             old_datetime=s2_meta.datetime)
        assert os.path.exists(name)
        assert os.path.exists(name+os.sep+'figs')

    # run assertions for S3
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage3{os.sep}')
    name = pathdirectory(meta, 'S3', 1, ap=4, bg=10,
                         old_datetime=s3_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    assert np.array_equal(s3_meta.citations, s3_cites)

    # run assertions for S4
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage4{os.sep}')
    name = pathdirectory(meta, 'S4', 1, ap=4, bg=10,
                         old_datetime=s4_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s4_cites = np.union1d(s3_cites, COMMON_IMPORTS[3])
    assert np.array_equal(s4_meta.citations, s4_cites)

    # run assertions for S5
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage5{os.sep}')
    name = pathdirectory(meta, 'S5', 1, ap=4, bg=10,
                         old_datetime=s5_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s5_cites = np.union1d(s4_cites, COMMON_IMPORTS[4] +
                          ["dynesty", "batman", "fleck"])
    assert np.array_equal(s5_meta.citations, s5_cites)

    if pymc3_installed:
        s5_cites2 = np.union1d(s4_cites, COMMON_IMPORTS[4] +
                               ["pymc3", "exoplanet", "starry"])
        assert np.array_equal(s5_meta2.citations, s5_cites2)

    # run assertions for S6
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage6{os.sep}')
    name = pathdirectory(meta, 'S6', 1, ap=4, bg=10,
                         old_datetime=s6_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    s6_cites = np.union1d(s5_cites, COMMON_IMPORTS[5])
    assert np.array_equal(s6_meta.citations, s6_cites)

    # Rerun Stages 5 and 6 using POET
    # remove Stage 5 and 6 temporary files
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage5")
    os.system(f"rm -r data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}Stage6")

    ecf_path = f'.{os.sep}MIRI_ecfs{os.sep}POET{os.sep}'
    s5_meta = s5.fitlc(meta.eventlabel, ecf_path=ecf_path, s4_meta=None)
    s6_meta = s6.plot_spectra(meta.eventlabel, ecf_path=ecf_path,
                              s5_meta=s5_meta)

    # run assertions for S5
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage5{os.sep}')
    name = pathdirectory(meta, 'S5', 1, ap=4, bg=10,
                         old_datetime=s5_meta.datetime)
    assert os.path.exists(name)
    assert os.path.exists(name+os.sep+'figs')

    # run assertions for S6
    meta.outputdir_raw = (f'data{os.sep}JWST-Sim{os.sep}MIRI{os.sep}'
                          f'Stage6{os.sep}')
    name = pathdirectory(meta, 'S6', 1, ap=4, bg=10,
                         old_datetime=s6_meta.datetime)
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
