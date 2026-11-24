import os
import sys

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib import util
from eureka.lib.readECF import MetaClass

EVENTLABEL = 'testevent'


def _make_meta(tmp_path):
    # Set topdir before outputdir so the __setattr__ join logic in
    # MetaClass triggers exactly as it does when reading a real ECF
    meta = MetaClass()
    meta.eventlabel = EVENTLABEL
    meta.datetime = '2026-01-01'
    meta.topdir = str(tmp_path)+os.sep
    meta.outputdir = 'Stage3'
    return meta


def test_pathdirectory_without_base_reproduces_nesting_bug(tmp_path):
    """Documents the hazard: reusing meta.outputdir_raw across repeated
    meta.outputdir assignments (the old call pattern) nests each new
    aperture/annulus directory inside the previous one."""
    meta = _make_meta(tmp_path)

    run = util.makedirectory(meta, 'S3', None, ap=4, bg=10)
    outputdir1 = util.pathdirectory(meta, 'S3', run, ap=4, bg=10)
    meta.outputdir = outputdir1

    run = util.makedirectory(meta, 'S3', run, ap=6, bg=14)
    outputdir2 = util.pathdirectory(meta, 'S3', run, ap=6, bg=14)

    assert outputdir1.rstrip(os.sep) in outputdir2


def test_pathdirectory_with_explicit_base_stays_flat(tmp_path):
    """Passing the cached base outputdir_raw (as s3_reduce.py, s4_genLC.py,
    s5_fit.py, and s6_spectra.py now do) keeps every aperture/annulus
    directory a flat sibling, regardless of prior meta.outputdir mutation."""
    meta = _make_meta(tmp_path)
    base_outputdir_raw = meta.outputdir_raw

    run = util.makedirectory(meta, 'S3', None, ap=4, bg=10,
                             outputdir_raw=base_outputdir_raw)
    outputdir1 = util.pathdirectory(meta, 'S3', run, ap=4, bg=10,
                                    outputdir_raw=base_outputdir_raw)
    meta.outputdir = outputdir1

    run = util.makedirectory(meta, 'S3', run, ap=6, bg=14,
                             outputdir_raw=base_outputdir_raw)
    outputdir2 = util.pathdirectory(meta, 'S3', run, ap=6, bg=14,
                                    outputdir_raw=base_outputdir_raw)
    meta.outputdir = outputdir2

    run = util.makedirectory(meta, 'S3', run, ap=8, bg=18,
                             outputdir_raw=base_outputdir_raw)
    outputdir3 = util.pathdirectory(meta, 'S3', run, ap=8, bg=18,
                                    outputdir_raw=base_outputdir_raw)

    assert outputdir1.rstrip(os.sep) not in outputdir2
    assert outputdir2.rstrip(os.sep) not in outputdir3
    assert (os.path.dirname(os.path.dirname(outputdir1.rstrip(os.sep))) ==
           os.path.dirname(os.path.dirname(outputdir2.rstrip(os.sep))) ==
           os.path.dirname(os.path.dirname(outputdir3.rstrip(os.sep))))
    assert os.path.exists(outputdir1)
    assert os.path.exists(outputdir2)
    assert os.path.exists(outputdir3)
