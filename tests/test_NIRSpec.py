# Last Updated: 2022-04-05

import numpy as np
import sys, os
from importlib import reload

sys.path.insert(0, '../')
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
try:
    from eureka.S2_calibrations import s2_calibrate as s2
except ModuleNotFoundError as e:
    pass
from eureka.S3_data_reduction import s3_reduce as s3
from eureka.S4_generate_lightcurves import s4_genLC as s4
from eureka.S5_lightcurve_fitting import s5_fit as s5

def test_NIRSpec(capsys):

    s2_installed = 'eureka.S2_calibrations.s2_calibrate' in sys.modules
    if not s2_installed:
        with capsys.disabled():
            print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
                "included in demo ecf files and documentation (docs/source/ecf.rst)")
            print('Skipping NIRSpec Stage 2 tests as could not import eureka.S2_calibrations.s2_calibrate')
            print("NIRSpec S3-5 test: ", end='', flush=True)
    else:
        with capsys.disabled():
            # is able to display any message without failing a test
            # useful to leave messages for future users who run the tests
            print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
                "included in demo ecf files and documentation (docs/source/ecf.rst)")
            print("\NIRSpec S2-5 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel='NIRSpec'
    meta.topdir='../tests'
    ecf_path='./NIRSpec_ecfs/'

    if s2_installed:
        # Only run S2 stuff if jwst package has been installed
        reload(s2)
    reload(s3)
    reload(s4)
    reload(s5)
    if s2_installed:
        # Only run S2 stuff if jwst package has been installed
        s2_meta = s2.calibrateJWST(meta.eventlabel, ecf_path=ecf_path)
    else:
        s2_meta = None
    s3_meta = s3.reduceJWST(meta.eventlabel, ecf_path=ecf_path, s2_meta=s2_meta)
    s4_meta = s4.lcJWST(meta.eventlabel, ecf_path=ecf_path, s3_meta=s3_meta)
    s5_meta = s5.fitJWST(meta.eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)
    
    # run assertions for S2
    if s2_installed:
        # Only run S2 stuff if jwst package has been installed
        meta.outputdir_raw='/data/JWST-Sim/NIRSpec/Stage2/'
        name = pathdirectory(meta, 'S2', 1)
        assert os.path.exists(name)
        assert os.path.exists(name+'/figs')
    
    # run assertions for S3
    meta.outputdir_raw='/data/JWST-Sim/NIRSpec/Stage3/'
    name = pathdirectory(meta, 'S3', 1, ap=8, bg=10)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S4
    meta.outputdir_raw='data/JWST-Sim/NIRSpec/Stage4/'
    name = pathdirectory(meta, 'S4', 1, ap=8, bg=10)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # run assertions for S5
    meta.outputdir_raw='data/JWST-Sim/NIRSpec/Stage5/'
    name = pathdirectory(meta, 'S5', 1, ap=8, bg=10)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')
    
    # remove temp files
    if s2_installed:
        # Only run S2 stuff if jwst package has been installed
        os.system("rm -r data/JWST-Sim/NIRSpec/Stage2/S2_*")
        pass
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage3")
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage4")
    os.system("rm -r data/JWST-Sim/NIRSpec/Stage5")
