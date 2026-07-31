"""Science-product regression tests for Eureka Stage 3."""

import astraeus.xarrayIO as xrio
import numpy as np
import pytest

from .cases import CASES
from .conftest import REFERENCE_ROOT

# Define tolerances for tests
RTOL = 1e-4 # relative tolerance, used for most arrays in general
MAD_RTOL = 1e-3 # relative tolerance for MAD value
CENTROID_ATOL = 1e-2 # absolute tolerance for centroid position

MAX_REPORTED_MISMATCHES = 10 # max number of mismatches explicitly output if array comparison test fails


def _reference_paths(case):
    reference_dir = REFERENCE_ROOT / case.reference_dir
    return reference_dir / "SpecData.h5"


def _mismatch_details(case, variable, actual, expected):
    """Return the first mismatched array elements for a failed comparison."""
    if variable in case.exact_variables:
        if np.issubdtype(actual.dtype, np.inexact):
            mismatches = ~np.isclose(actual, expected, rtol=0, atol=0,
                                     equal_nan=True)
        else:
            mismatches = actual != expected
    elif variable in case.atol_variables:
        mismatches = ~np.isclose(actual, expected, rtol=0,
                                 atol=CENTROID_ATOL, equal_nan=True)
    else:
        mismatches = ~np.isclose(actual, expected, rtol=RTOL, atol=0,
                                 equal_nan=True)

    indices = np.argwhere(mismatches)
    lines = [f"{case.name}: {variable} differs at {len(indices)} elements.",
             f"First {min(len(indices), MAX_REPORTED_MISMATCHES)} "
             "mismatches (index: actual, expected):"]
    for index in indices[:MAX_REPORTED_MISMATCHES]:
        index = tuple(index)
        lines.append(f"  {index}: {actual[index]!r}, {expected[index]!r}")
    return "\n".join(lines)


def _assert_array(case, variable, actual, expected):
    # Compare variable actual value to expected value. Type of comparison (exact, absolute, relative)
    # depends on the variable being tested.
    assert actual.shape == expected.shape, (
        f"{case.name}: {variable} shape changed from {expected.shape} "
        f"to {actual.shape}."
    )
    try:
        if variable in case.exact_variables:
            if np.issubdtype(actual.dtype, np.inexact):
                np.testing.assert_allclose(
                    actual, expected, rtol=0, atol=0, equal_nan=True,
                    err_msg=f"{case.name}: {variable}")
            else:
                np.testing.assert_array_equal(
                    actual, expected, err_msg=f"{case.name}: {variable}")
        elif variable in case.atol_variables:
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=CENTROID_ATOL, equal_nan=True,
                err_msg=f"{case.name}: {variable}")
        else:
            np.testing.assert_allclose(
                actual, expected, rtol=RTOL, atol=0, equal_nan=True,
                err_msg=f"{case.name}: {variable}")
    except AssertionError as error:
        raise AssertionError(
            f"{error}\n\n{_mismatch_details(case, variable, actual, expected)}"
        ) from None


def _assert_mad(case, actual_meta, expected):
    """Compare the Stage 3 MAD stored in the SpecData metadata."""
    assert "mad_s3" in expected.attrs, \
        f"{case.name}: missing reference mad_s3"
    actual_mad = np.atleast_1d(np.asarray(actual_meta.mad_s3))
    reference_mad = np.atleast_1d(np.asarray(expected.attrs["mad_s3"]))
    assert actual_mad.shape == reference_mad.shape, (
        f"{case.name}: mad_s3 shape changed from {reference_mad.shape} "
        f"to {actual_mad.shape}."
    )
    np.testing.assert_allclose(actual_mad, reference_mad, rtol=MAD_RTOL,
                               atol=0, equal_nan=True,
                               err_msg=f"{case.name}: mad_s3")


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
def test_s3_science_products(case, run_s3):
    """Stage 3 outputs must match their approved instrument/mode baseline."""
    actual, actual_meta = run_s3(case)
    specdata_path = _reference_paths(case)
    assert specdata_path.is_file(), f"Missing reference: {specdata_path}"

    expected = xrio.readXR(str(specdata_path), verbose=False)
    for variable in case.variables: # loop over cases defined in cases.py and test
        assert variable in actual, f"{case.name}: missing output {variable}"
        assert variable in expected, f"{case.name}: missing reference {variable}"
        _assert_array(case, variable, actual[variable].values,
                      expected[variable].values)

    _assert_mad(case, actual_meta, expected)
    _assert_case_semantics(case, actual, expected)
