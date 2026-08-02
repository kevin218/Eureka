"""Science-product regression tests for Eureka Stage 3."""

import astraeus.xarrayIO as xrio
import numpy as np
import pytest

from .cases import CASES
from .conftest import REFERENCE_ROOT

# Test tolerances.
RTOL = 1e-4  # Relative tolerance for most arrays.
MAD_RTOL = 1e-3  # Relative tolerance for MAD values.
CENTROID_ATOL = 1e-2  # Absolute tolerance for centroid positions.

MAX_REPORTED_MISMATCHES = 10


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
    """Assert an array matches its reference using the case's tolerance."""
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


def _assert_mad(case, actual, expected):
    """Compare the Stage 3 MAD stored in the SpecData metadata."""
    assert "mad_s3" in actual.attrs, f"{case.name}: missing output mad_s3"
    assert "mad_s3" in expected.attrs, \
        f"{case.name}: missing reference mad_s3"
    actual_mad = np.atleast_1d(np.asarray(actual.attrs["mad_s3"]))
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


def _overwrite_reference(case, actual, specdata_path):
    """Replace one approved reference with the current Stage 3 product."""
    assert specdata_path.is_file(), (
        f"{case.name}: refusing to create an untracked reference: "
        f"{specdata_path}"
    )
    reference = actual.copy()
    temporary_path = specdata_path.with_name(
        specdata_path.stem + ".tmp" + specdata_path.suffix)
    success = xrio.writeXR(str(temporary_path), reference, verbose=False)
    if not success:
        raise OSError(f"Failed to write updated reference: {specdata_path}")
    temporary_path.replace(specdata_path)
    print(f"Updated Stage 3 reference: {specdata_path}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s3_science_products(case, run_s3, overwrite_ref_files):
    """Stage 3 outputs must match their approved instrument/mode baseline."""
    actual, _ = run_s3(case)
    specdata_path = _reference_paths(case)
    assert specdata_path.is_file(), f"Missing reference: {specdata_path}"

    if overwrite_ref_files:
        _overwrite_reference(case, actual, specdata_path)
        return

    expected = xrio.readXR(str(specdata_path), verbose=False)
    for variable in case.variables:
        assert variable in actual, f"{case.name}: missing output {variable}"
        assert variable in expected, (
            f"{case.name}: missing reference {variable}"
        )
        _assert_array(case, variable, actual[variable].values,
                      expected[variable].values)

    _assert_mad(case, actual, expected)
    _assert_case_semantics(case, actual, expected)
