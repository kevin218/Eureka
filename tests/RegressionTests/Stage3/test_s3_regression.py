"""Science-product regression tests for Eureka Stage 3."""

import astraeus.xarrayIO as xrio
import numpy as np
import pytest

from ..regression_helpers import (assert_array, assert_attribute,
                                  overwrite_reference)
from .cases import CASES
from .conftest import REFERENCE_ROOT

# Test tolerances.
RTOL = 1e-4  # Relative tolerance for most arrays.
MAED_RTOL = 2e-3  # Relative tolerance for MAED values.
CENTROID_ATOL = 1e-2  # Absolute tolerance for centroid positions.
# WFC3 optspec differs from its approved reference by up to 0.45% in
# CI.
WFC3_OPTSPEC_RTOL = 5e-3
# WFC3 opterr differs from its approved reference by up to 6.76e-4
# in CI.
WFC3_OPTERR_RTOL = 1e-3
# The median image can differ by a bit greater than 1e-4 from its
# reference in CI.
MEDFLUX_RTOL = 2e-4


def _rtol(case, variable):
    """Return the relative tolerance for one regression variable."""
    if case.name == "wfc3_spectroscopy" and variable == "optspec":
        return WFC3_OPTSPEC_RTOL
    if case.name == "wfc3_spectroscopy" and variable == "opterr":
        return WFC3_OPTERR_RTOL
    if variable == "medflux":
        return MEDFLUX_RTOL
    return RTOL


def _reference_paths(case):
    reference_dir = REFERENCE_ROOT / case.reference_dir
    return reference_dir / "SpecData.h5"


def _assert_case_semantics(case, actual, expected):
    # Checks NIRISS multi-order extraction (which is special from other modes)
    if case.check_niriss_orders:
        np.testing.assert_array_equal(actual["order"].values,
                                      expected["order"].values,
                                      err_msg=f"{case.name}: order coordinate")
        assert actual.optspec.ndim == 3, (
            f"{case.name}: expected time, wavelength, and order dimensions."
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s3_science_products(case, run_s3, overwrite_ref_files):
    """Stage 3 outputs must match their approved instrument/mode baseline."""
    actual, _ = run_s3(case)
    specdata_path = _reference_paths(case)
    assert specdata_path.is_file(), f"Missing reference: {specdata_path}"

    if overwrite_ref_files:
        overwrite_reference(case, "SpecData", actual, specdata_path, stage=3)
        return

    expected = xrio.readXR(str(specdata_path), verbose=False)
    for variable in case.variables:
        assert variable in actual, f"{case.name}: missing output {variable}"
        assert variable in expected, (
            f"{case.name}: missing reference {variable}"
        )
        assert_array(case, variable, actual[variable].values,
                     expected[variable].values, rtol=_rtol(case, variable),
                     centroid_atol=CENTROID_ATOL)

    # Assertions for HDF5 metadata
    assert_attribute(case, "SpecData", actual, expected, "maed_s3",
                     rtol=MAED_RTOL, atleast_1d=True)
    _assert_case_semantics(case, actual, expected)
