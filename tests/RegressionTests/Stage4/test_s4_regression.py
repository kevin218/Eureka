import astraeus.xarrayIO as xrio
import numpy as np
import pytest

from .cases import CASES
from .conftest import REFERENCE_ROOT


# Most science arrays use a relative comparison. Each case declares its
# exact-value and centroid-position exceptions alongside its product manifest.
RTOL = 1e-4
MAD_RTOL = 1e-3
CENTROID_ATOL = 1e-2
MAX_REPORTED_MISMATCHES = 10


def _reference_paths(case):
    """Return the two approved Stage 4 reference files for one case."""
    reference_dir = REFERENCE_ROOT / case.reference_dir
    return reference_dir / "SpecData.h5", reference_dir / "LCData.h5"


def _mismatch_details(case, product, variable, actual, expected):
    """Return the first mismatched elements for a failed comparison."""
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
    label = f"{product}.{variable}"
    lines = [f"{case.name}: {label} differs at {len(indices)} elements.",
             f"First {min(len(indices), MAX_REPORTED_MISMATCHES)} "
             "mismatches (index: actual, expected):"]
    for index in indices[:MAX_REPORTED_MISMATCHES]:
        index = tuple(index)
        lines.append(f"  {index}: {actual[index]!r}, {expected[index]!r}")
    return "\n".join(lines)


def _assert_array(case, product, variable, actual, expected):
    """Compare one science array using its appropriate numerical tolerance."""
    assert actual.shape == expected.shape, (
        f"{case.name}: {variable} shape changed from {expected.shape} to "
        f"{actual.shape}."
    )
    try:
        if variable in case.exact_variables:
            if np.issubdtype(actual.dtype, np.inexact):
                np.testing.assert_allclose(
                    actual, expected, rtol=0, atol=0, equal_nan=True,
                    err_msg=f"{case.name}: {product}.{variable}")
            else:
                message = f"{case.name}: {product}.{variable}"
                np.testing.assert_array_equal(
                    actual, expected, err_msg=message)
        elif variable in case.atol_variables:
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=CENTROID_ATOL, equal_nan=True,
                err_msg=f"{case.name}: {product}.{variable}")
        else:
            np.testing.assert_allclose(
                actual, expected, rtol=RTOL, atol=0, equal_nan=True,
                err_msg=f"{case.name}: {product}.{variable}")
    except AssertionError as error:
        raise AssertionError(
            f"{error}\n\n"
            f"{_mismatch_details(case, product, variable, actual, expected)}"
        ) from None


def _assert_attribute(case, product, actual, expected, attribute, rtol=0):
    """Compare one science-product attribute saved with its data product."""
    assert attribute in actual.attrs, (
        f"{case.name}: missing {product} output attribute "
        f"{attribute}"
    )
    assert attribute in expected.attrs, (
        f"{case.name}: missing {product} reference attribute "
        f"{attribute}"
    )
    np.testing.assert_allclose(actual.attrs[attribute],
                               expected.attrs[attribute], rtol=rtol, atol=0,
                               equal_nan=True,
                               err_msg=(f"{case.name}: {product} "
                                        f"{attribute}"))


def _overwrite_reference(case, product, actual, reference_path):
    """Replace one approved Stage 4 reference with its current product."""
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
    print(f"Updated Stage 4 {product} reference: {reference_path}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s4_science_products(case, run_s4, overwrite_ref_files):
    """Run standalone S4 and compare its selected products to the baseline."""
    spec, lc, _ = run_s4(case)
    spec_path, lc_path = _reference_paths(case)
    assert spec_path.is_file(), f"Missing reference: {spec_path}"
    assert lc_path.is_file(), f"Missing reference: {lc_path}"

    if overwrite_ref_files:
        _overwrite_reference(case, "SpecData", spec, spec_path)
        _overwrite_reference(case, "LCData", lc, lc_path)
        return

    # Compare the configured science arrays from both Stage 4 products.
    expected_spec = xrio.readXR(str(spec_path), verbose=False)
    expected_lc = xrio.readXR(str(lc_path), verbose=False)
    for variable in case.spec_variables:
        assert variable in spec, (
            f"{case.name}: missing SpecData output {variable}"
        )
        assert variable in expected_spec, (
            f"{case.name}: missing SpecData reference {variable}"
        )
        _assert_array(case, "SpecData", variable, spec[variable].values,
                      expected_spec[variable].values)
    for variable in case.lc_variables:
        assert variable in lc, (
            f"{case.name}: missing LCData output {variable}"
        )
        assert variable in expected_lc, (
            f"{case.name}: missing LCData reference {variable}"
        )
        _assert_array(case, "LCData", variable, lc[variable].values,
                      expected_lc[variable].values)

    _assert_attribute(case, "SpecData", spec, expected_spec, "mad_s4",
                      rtol=MAD_RTOL)
    _assert_attribute(case, "SpecData", spec, expected_spec, "mask_columns")
    _assert_attribute(case, "LCData", lc, expected_lc, "mad_s4_binned",
                      rtol=MAD_RTOL)
    _assert_attribute(case, "LCData", lc, expected_lc,
                      "mad_s4_binned_bg", rtol=MAD_RTOL)
