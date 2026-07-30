"""Science-product regression tests for Eureka Stage 4."""
import json

import numpy as np
import pytest
import astraeus.xarrayIO as xrio

from .cases import CASES
from .conftest import REFERENCE_ROOT


# Most science arrays use a relative comparison; centroid locations use a
# small absolute tolerance, while masks, bin edges, and count-like arrays
# must be unchanged.
RTOL = 1e-4
MAD_RTOL = 1e-3
CENTROID_ATOL = 1e-2
EXACT_VARIABLES = {
    "optmask", "mask", "mask_white", "driftmask", "scandir", "flatmask",
    "wave_1d", "wave", "wave_low", "wave_hi", "wave_mid", "wave_err",
    "nappix", "nskypix", "nskyideal", "status",
}
CENTROID_VARIABLES = {"centroid_x", "centroid_y"}


def _reference_paths(case):
    """Return the three approved Stage 4 reference files for one case."""
    reference_dir = REFERENCE_ROOT / case.reference_dir
    return (reference_dir / "SpecData.h5", reference_dir / "LCData.h5",
            reference_dir / "metadata.json")


def _assert_array(case, variable, actual, expected):
    """Compare one science array using its appropriate numerical tolerance."""
    assert actual.shape == expected.shape, (
        f"{case.name}: {variable} shape changed from {expected.shape} to "
        f"{actual.shape}."
    )
    if variable in EXACT_VARIABLES:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                                   equal_nan=True,
                                   err_msg=f"{case.name}: {variable}")
    elif variable in CENTROID_VARIABLES:
        np.testing.assert_allclose(actual, expected, rtol=0,
                                   atol=CENTROID_ATOL, equal_nan=True,
                                   err_msg=f"{case.name}: {variable}")
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0,
                                   equal_nan=True,
                                   err_msg=f"{case.name}: {variable}")


def _to_json_comparable(value):
    """Convert NumPy values and masked entries to JSON-comparable values."""
    if value is np.ma.masked:
        return None
    if isinstance(value, np.ndarray):
        return _to_json_comparable(value.tolist())
    if isinstance(value, np.generic):
        return _to_json_comparable(value.item())
    if isinstance(value, (list, tuple)):
        return [_to_json_comparable(item) for item in value]
    return value


def _assert_metadata(case, actual_meta, metadata_path):
    """Compare saved Stage 4 metadata metrics and bin definitions."""
    expected = json.loads(metadata_path.read_text())
    for key, expected_value in expected.items():
        if key == "reference_schema_version":
            continue
        assert hasattr(actual_meta, key), f"{case.name}: missing metadata {key}"
        actual_value = _to_json_comparable(getattr(actual_meta, key))
        if isinstance(expected_value, list):
            assert len(actual_value) == len(expected_value), (
                f"{case.name}: {key} length changed."
            )
            for actual_item, expected_item in zip(actual_value, expected_value):
                if expected_item is None:
                    assert actual_item is None, f"{case.name}: {key} mask changed."
                elif key.startswith("mad_s4"):
                    np.testing.assert_allclose(actual_item, expected_item,
                                               rtol=MAD_RTOL, atol=0,
                                               equal_nan=True,
                                               err_msg=f"{case.name}: {key}")
                else:
                    np.testing.assert_allclose(actual_item, expected_item,
                                               rtol=0, atol=0, equal_nan=True,
                                               err_msg=f"{case.name}: {key}")
        elif key.startswith("mad_s4"):
            np.testing.assert_allclose(actual_value, expected_value,
                                       rtol=MAD_RTOL, atol=0, equal_nan=True,
                                       err_msg=f"{case.name}: {key}")
        else:
            assert actual_value == expected_value, f"{case.name}: {key} changed."


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s4_science_products(case, run_s4):
    """Run standalone S4 and compare its selected products to the baseline."""
    spec, lc, meta = run_s4(case)
    spec_path, lc_path, metadata_path = _reference_paths(case)
    assert spec_path.is_file(), f"Missing reference: {spec_path}"
    assert lc_path.is_file(), f"Missing reference: {lc_path}"
    assert metadata_path.is_file(), f"Missing reference: {metadata_path}"

    # Compare the configured science arrays from both Stage 4 products.
    expected_spec = xrio.readXR(str(spec_path), verbose=False)
    expected_lc = xrio.readXR(str(lc_path), verbose=False)
    for variable in case.spec_variables:
        assert variable in spec, f"{case.name}: missing SpecData output {variable}"
        assert variable in expected_spec, f"{case.name}: missing SpecData reference {variable}"
        _assert_array(case, variable, spec[variable].values,
                      expected_spec[variable].values)
    for variable in case.lc_variables:
        assert variable in lc, f"{case.name}: missing LCData output {variable}"
        assert variable in expected_lc, f"{case.name}: missing LCData reference {variable}"
        _assert_array(case, variable, lc[variable].values,
                      expected_lc[variable].values)

    # Check the non-array reference metadata.
    _assert_metadata(case, meta, metadata_path)
