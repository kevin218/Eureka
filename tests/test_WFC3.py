# Last Updated: 2022-04-05

import numpy as np
import sys, os
from importlib import reload

sys.path.insert(0, '../')
from eureka.lib.readECF import MetaClass
from eureka.lib.util import pathdirectory
import eureka.lib.plots
from eureka.S3_data_reduction import s3_reduce as s3

def test_WFC3(capsys):
    # Set up some parameters to make plots look nicer. You can set usetex=True if you have LaTeX installed
    eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.pdf')

    with capsys.disabled():
        # is able to display any message without failing a test
        # useful to leave messages for future users who run the tests
        print("\n\nIMPORTANT: Make sure that any changes to the ecf files are "+
              "included in demo ecf files and documentation (docs/source/ecf.rst)")
        print("\nWFC3 S3 test: ", end='', flush=True)

    # explicitly define meta variables to be able to run pathdirectory fn locally
    meta = MetaClass()
    meta.eventlabel='WFC3'
    meta.topdir='../tests'
    ecf_path='./WFC3_ecfs/'

    reload(s3)
    s3_meta = s3.reduce(meta.eventlabel, ecf_path=ecf_path)

    # run assertions for S3
    meta.outputdir_raw='data/WFC3/Stage3/'
    name = pathdirectory(meta, 'S3', 1, ap=8, bg=40)
    assert os.path.exists(name)
    assert os.path.exists(name+'/figs')

    # remove temporary files
    os.system("rm -r data/WFC3/Stage3")
