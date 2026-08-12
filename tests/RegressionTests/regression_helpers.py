"""Shared assertions and reference-file utilities for regression tests."""

import astraeus.xarrayIO as xrio
import numpy as np


MAX_REPORTED_MISMATCHES = 10


def _label(product, variable):
    """Return a variable label, optionally qualified by its data product."""
    return f"{product}.{variable}" if product else variable


def _mismatch_details(case, product, variable, actual, expected, rtol,
                      centroid_atol):
    """Return the first mismatched array elements for a failed comparison."""
    if variable in case.exact_variables:
        if np.issubdtype(actual.dtype, np.inexact):
            mismatches = ~np.isclose(actual, expected, rtol=0, atol=0,
                                     equal_nan=True)
        else:
            mismatches = actual != expected
    elif variable in case.atol_variables:
        mismatches = ~np.isclose(actual, expected, rtol=0,
                                 atol=centroid_atol, equal_nan=True)
    else:
        mismatches = ~np.isclose(actual, expected, rtol=rtol, atol=0,
                                 equal_nan=True)

    indices = np.argwhere(mismatches)
    label = _label(product, variable)
    lines = [f"{case.name}: {label} differs at {len(indices)} elements.",
             f"First {min(len(indices), MAX_REPORTED_MISMATCHES)} "
             "mismatches (index: actual, expected):"]
    for index in indices[:MAX_REPORTED_MISMATCHES]:
        index = tuple(index)
        lines.append(f"  {index}: {actual[index]!r}, {expected[index]!r}")
    return "\n".join(lines)


def assert_array(case, variable, actual, expected, *, product=None,
                 rtol=1e-4, centroid_atol=1e-2):
    """Compare a science array using the case's configured tolerance."""
    label = _label(product, variable)
    assert actual.shape == expected.shape, (
        f"{case.name}: {label} shape changed from {expected.shape} to "
        f"{actual.shape}."
    )
    try:
        if variable in case.exact_variables:
            if np.issubdtype(actual.dtype, np.inexact):
                # treats NaNs as equal for floating point arrays
                np.testing.assert_allclose(
                    actual, expected, rtol=0, atol=0, equal_nan=True,
                    err_msg=f"{case.name}: {label}")
            else:
                np.testing.assert_array_equal(
                    actual, expected, err_msg=f"{case.name}: {label}")
        elif variable in case.atol_variables:
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=centroid_atol, equal_nan=True,
                err_msg=f"{case.name}: {label}")
        else:
            np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=0, equal_nan=True,
                err_msg=f"{case.name}: {label}")
    except AssertionError as error:
        details = _mismatch_details(case, product, variable, actual, expected,
                                    rtol, centroid_atol)
        raise AssertionError(
            f"{error}\n\n{details}"
        ) from None


def assert_attribute(case, product, actual, expected, attribute, *, rtol=0,
                     atleast_1d=False):
    """Compare metadata attribute in HDF5 files"""
    assert attribute in actual.attrs, (
        f"{case.name}: missing {product} output attribute {attribute}"
    )
    assert attribute in expected.attrs, (
        f"{case.name}: missing {product} reference attribute {attribute}"
    )
    actual_value = actual.attrs[attribute]
    expected_value = expected.attrs[attribute]
    if atleast_1d:
        actual_value = np.atleast_1d(np.asarray(actual_value))
        expected_value = np.atleast_1d(np.asarray(expected_value))
        assert actual_value.shape == expected_value.shape, (
            f"{case.name}: {attribute} shape changed from "
            f"{expected_value.shape} to {actual_value.shape}."
        )
    np.testing.assert_allclose(actual_value, expected_value, rtol=rtol, atol=0,
                               equal_nan=True,
                               err_msg=f"{case.name}: {product} {attribute}")


def overwrite_reference(case, product, actual, reference_path, stage):
    """Safely replace one approved reference with its current product."""
    assert reference_path.is_file(), (
        f"{case.name}: refusing to create an untracked reference: "
        f"{reference_path}"
    )
    temporary_path = reference_path.with_name(
        reference_path.stem + ".tmp" + reference_path.suffix)
    success = xrio.writeXR(str(temporary_path), actual.copy(), verbose=False)
    if not success:
        raise OSError(f"Failed to write updated reference: {reference_path}")
    temporary_path.replace(reference_path)
    print(f"Updated Stage {stage} {product} reference: {reference_path}")
