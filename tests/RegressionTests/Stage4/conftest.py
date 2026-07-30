"""Fixtures shared only by the Stage 4 regression suite."""
from pathlib import Path
from shutil import copy2

import pytest

from eureka.S4_generate_lightcurves import s4_genLC
from eureka.S4_generate_lightcurves.s4_meta import S4MetaClass
from eureka.lib import readECF


REFERENCE_ROOT = Path(__file__).parent / "references"
S3_REFERENCE_ROOT = Path(__file__).parents[1] / "Stage3" / "references"


@pytest.fixture
def run_s4(tmp_path, pytestconfig, monkeypatch):
    """Run standalone Stage 4 from an approved Stage 3 reference product."""
    repo_root = Path(pytestconfig.rootpath)

    def _run(case):
        input_dir = tmp_path / "stage3-input"
        input_dir.mkdir()
        copy2(S3_REFERENCE_ROOT / case.s3_reference_dir / "SpecData.h5",
              input_dir / case.input_filename)

        input_meta = S4MetaClass(folder=str(repo_root / case.ecf_dir),
                                 eventlabel=case.eventlabel)
        input_meta.topdir = str(tmp_path)
        input_meta.inputdir = f"{input_dir}/"
        input_meta.inputdir_raw = "stage3-input/"
        input_meta.outputdir = "stage4-output/"
        input_meta.outputdir_raw = "stage4-output/"

        # The WFC3 S3 product embeds its HST pmap (1345).  MetaClass restores
        # attributes in file order, where ``inst`` precedes ``pmap``; without
        # a local CRDS cache it consequently asks CRDS for the context before
        # reaching the saved value.  The scientific Stage 4 calculation does
        # not use CRDS, so we provide that already-saved context locally and keep
        # this regression test independent of network/cache state.
        if case.name == "wfc3_spectroscopy":
            monkeypatch.setattr(readECF.crds, "get_context_name",
                                lambda observatory: "hst_1345.pmap")

        return s4_genLC.genlc(case.eventlabel, input_meta=input_meta)

    return _run
