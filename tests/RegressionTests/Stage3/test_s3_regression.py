"""Science-product regression tests for Eureka Stage 3."""
import json

import numpy as np
import pytest
import astraeus.xarrayIO as xrio

from .cases import CASES
from .conftest import REFERENCE_ROOT

# Define tolerances for tests
RTOL = 1e-4 # relative tolerance, used for most arrays in general
MAD_RTOL = 1e-3 # relative tolerance for MAD value
CENTROID_ATOL = 1e-2 # absolute tolerance for centroid position


def _reference_paths(case):
    reference_dir = REFERENCE_ROOT / case.reference_dir
    return reference_dir / "SpecData.h5", reference_dir / "metadata.json"


def _assert_array(case, variable, actual, expected):
    assert actual.shape == expected.shape, (
        f"{case.name}: {variable} shape changed from {expected.shape} "
        f"to {actual.shape}."
    )
    if variable in case.exact_variables:
        if np.issubdtype(actual.dtype, np.inexact):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                                       equal_nan=True,
                                       err_msg=f"{case.name}: {variable}")
        else:
            np.testing.assert_array_equal(actual, expected,
                                          err_msg=f"{case.name}: {variable}")
    elif variable in case.atol_variables:
        np.testing.assert_allclose(actual, expected, rtol=0,
                                   atol=CENTROID_ATOL, equal_nan=True,
                                   err_msg=f"{case.name}: {variable}")
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0,
                                   equal_nan=True,
                                   err_msg=f"{case.name}: {variable}")


def _assert_mad(case, actual_meta, metadata_path):
    reference_mad = np.asarray(json.loads(metadata_path.read_text())["mad_s3"])
    actual_mad = np.asarray(actual_meta.mad_s3)
    assert actual_mad.shape == reference_mad.shape, (
        f"{case.name}: mad_s3 shape changed from {reference_mad.shape} "
        f"to {actual_mad.shape}."
    )
    np.testing.assert_allclose(actual_mad, reference_mad, rtol=MAD_RTOL,
                               atol=0, equal_nan=True,
                               err_msg=f"{case.name}: mad_s3")


def _assert_case_semantics(case, actual, expected):
    if case.check_niriss_orders:
        np.testing.assert_array_equal(actual["order"].values,
                                      expected["order"].values,
                                      err_msg=f"{case.name}: order coordinate")
        assert actual.optspec.ndim == 3, (
            f"{case.name}: expected time, wavelength, and order dimensions."
        )

    if case.check_miri_orientation:
        actual_order = np.argsort(actual.wave_1d.values)
        expected_order = np.argsort(expected.wave_1d.values)
        np.testing.assert_allclose(actual.wave_1d.values[actual_order],
                                   expected.wave_1d.values[expected_order],
                                   rtol=0, atol=0, equal_nan=True,
                                   err_msg=f"{case.name}: sorted wavelengths")
        np.testing.assert_allclose(actual.optspec.values[:, actual_order],
                                   expected.optspec.values[:, expected_order],
                                   rtol=RTOL, atol=0, equal_nan=True,
                                   err_msg=f"{case.name}: sorted spectrum")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s3_science_products(case, run_s3):
    """Stage 3 outputs must match their approved instrument/mode baseline."""
    actual, actual_meta = run_s3(case)
    specdata_path, metadata_path = _reference_paths(case)
    assert specdata_path.is_file(), f"Missing reference: {specdata_path}"
    assert metadata_path.is_file(), f"Missing reference: {metadata_path}"

    expected = xrio.readXR(str(specdata_path), verbose=False)
    for variable in case.variables:
        assert variable in actual, f"{case.name}: missing output {variable}"
        assert variable in expected, f"{case.name}: missing reference {variable}"
        _assert_array(case, variable, actual[variable].values,
                      expected[variable].values)

    _assert_mad(case, actual_meta, metadata_path)
    _assert_case_semantics(case, actual, expected)
