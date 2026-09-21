"""Fixtures shared only by the Stage 3 regression suite."""
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eureka.S3_data_reduction import s3_reduce
from eureka.S3_data_reduction.s3_meta import S3MetaClass

REFERENCE_ROOT = Path(__file__).parent / "references"


@pytest.fixture
def run_s3(tmp_path, pytestconfig):
    """Run Stage 3 with production ECF settings and isolated output."""
    repo_root = Path(pytestconfig.rootpath)  # Eureka repository root

    def _run(case):
        input_meta = S3MetaClass(folder=str(repo_root / case.ecf_dir),
                                 eventlabel=case.eventlabel)
        input_meta = deepcopy(input_meta)
        # MetaClass normalizes directories relative to topdir. Keep the source
        # tree as that root and use a relative route to pytest's workspace.
        input_meta.topdir = str(repo_root)
        input_meta.inputdir = str(repo_root / case.input_dir)
        input_meta.inputdir_raw = input_meta.inputdir
        # Regression references are saved SpecData products, so always execute
        # the production output-writing path in the isolated pytest workspace.
        input_meta.save_output = True
        outputdir = tmp_path / "stage3"
        outputdir_raw = os.path.relpath(outputdir, repo_root)
        input_meta.outputdir = outputdir_raw
        input_meta.outputdir_raw = outputdir_raw

        return s3_reduce.reduce(case.eventlabel, input_meta=input_meta)

    return _run
