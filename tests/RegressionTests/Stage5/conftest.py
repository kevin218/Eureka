"""Configure Stage 5 regression runs.
"""
from pathlib import Path
from shutil import copy2

import astraeus.xarrayIO as xrio
import numpy as np
import pytest

from eureka.lib import manageevent as me
from eureka.S5_lightcurve_fitting import s5_fit
from eureka.S5_lightcurve_fitting.s5_meta import S5MetaClass

REFERENCE_ROOT = Path(__file__).parent / "references"
S4_REFERENCE_ROOT = Path(__file__).parents[1] / "Stage4" / "references"


@pytest.fixture
def run_s5(tmp_path, pytestconfig):
    """Return a callable that runs one S5 case using the 
    corresponding S4 outputs.

    The returned S5 metadata identifies the temporary
    output directory, which the regression test then compares with the Stage 5
    golden reference files.
    """
    repo_root = Path(pytestconfig.rootpath)

    def _run(case):
        """Run ``case`` with temporary copies of its two S4 references."""
        input_dir = tmp_path / "stage4-input"
        input_dir.mkdir()
        # Sampler cases reuse the same fixed Stage 4 input as their matching
        # LSQ instrument/mode case, while their S5 reference directory keeps
        # a distinct name for the selected fitter.
        s4_case = case.s4_reference_case or case.name
        reference_dir = S4_REFERENCE_ROOT / s4_case
        specdata_path = input_dir / "SpecData.h5"
        lcdata_path = input_dir / "LCData.h5"

        # Copy the original reference files
        copy2(reference_dir / "SpecData.h5", specdata_path)
        copy2(reference_dir / "LCData.h5", lcdata_path)

        # ``loadevent`` knows how to reconstruct a metadata object from a
        # SpecData file's xarray attributes. Those inherited attributes include
        # the event label, data format, aperture/background values, and expand
        # factor that S5 needs to reproduce normal output routing.
        s4_meta = me.loadevent(str(specdata_path))

        # S4 adds its final spectral-bin definitions to LCData rather than to
        # SpecData metadata. Supply the values S5 expects on its predecessor
        # metadata object directly from the fixed LCData reference.
        lc = xrio.readXR(str(lcdata_path), verbose=False)
        s4_meta.nspecchan = lc.sizes["wavelength"]
        s4_meta.wave_low = lc.wave_low.values
        s4_meta.wave_hi = lc.wave_hi.values
        s4_meta.n_int = lc.sizes["time"]

        # SpecData stores these as scalar attributes, while S5 iterates over
        # aperture/background ranges. Normalizing them to one-element arrays
        # preserves this case's single approved S4 aperture pair.
        s4_meta.spec_hw_range = np.atleast_1d(s4_meta.spec_hw)
        s4_meta.bg_hw_range = np.atleast_1d(s4_meta.bg_hw)

        # Redirect every predecessor path that S5 follows to the temporary
        # workspace. S5 resumes its log from the S4 log, so provide an empty
        # local log file rather than touching an integration-test artifact.
        s4_meta.topdir = str(tmp_path)
        s4_meta.outputdir = f"{input_dir}/"
        s4_meta.outputdir_raw = "stage4-input/"
        s4_meta.filename_S4_LCData = str(lcdata_path)
        s4_meta.s4_logname = str(input_dir / f"S4_{case.eventlabel}.log")
        # ``mergeevents`` retains the predecessor ``folder`` while S5 resolves
        # its EPF. Point it at the selected S5 ECF, which matters for MIRI's
        # POET configuration nested below the standard MIRI ECF directory.
        s4_meta.folder = str(repo_root / case.ecf_dir)
        Path(s4_meta.s4_logname).touch()

        # Read the case's real S5 ECF, but override only test-environment
        # concerns: all paths are under pytest's workspace and plotting is off
        # because figures are intentionally outside this regression contract.
        input_meta = S5MetaClass(folder=str(repo_root / case.ecf_dir),
                                 file=case.ecf_filename)
        input_meta.topdir = str(tmp_path)
        input_meta.outputdir = "stage5-output/"
        input_meta.outputdir_raw = "stage5-output/"
        input_meta.isplots_S5 = 0
        input_meta.hide_plots = True

        return s5_fit.fitlc(case.eventlabel, s4_meta=s4_meta,
                            input_meta=input_meta)

    return _run
