import astraeus.xarrayIO as xrio
import pytest

from ..regression_helpers import (assert_array, assert_attribute,
                                  overwrite_reference)
from .cases import CASES
from .conftest import REFERENCE_ROOT

# Most science arrays use a relative comparison. Each case declares its
# exact-value and centroid-position exceptions alongside its product manifest.
RTOL = 1e-4
MAED_RTOL = 1e-3
CENTROID_ATOL = 1e-2


def _reference_paths(case):
    """Return the two approved Stage 4 reference files for one case."""
    reference_dir = REFERENCE_ROOT / case.reference_dir
    return reference_dir / "SpecData.h5", reference_dir / "LCData.h5"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s4_science_products(case, run_s4, overwrite_ref_files):
    """Run standalone S4 and compare its selected products to the baseline."""
    spec, lc, _ = run_s4(case)
    spec_path, lc_path = _reference_paths(case)
    assert spec_path.is_file(), f"Missing reference: {spec_path}"
    assert lc_path.is_file(), f"Missing reference: {lc_path}"

    if overwrite_ref_files:
        overwrite_reference(case, "SpecData", spec, spec_path, stage=4)
        overwrite_reference(case, "LCData", lc, lc_path, stage=4)
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
        assert_array(case, variable, spec[variable].values,
                     expected_spec[variable].values, product="SpecData",
                     rtol=RTOL, centroid_atol=CENTROID_ATOL)
    for variable in case.lc_variables:
        assert variable in lc, (
            f"{case.name}: missing LCData output {variable}"
        )
        assert variable in expected_lc, (
            f"{case.name}: missing LCData reference {variable}"
        )
        assert_array(case, variable, lc[variable].values,
                     expected_lc[variable].values, product="LCData",
                     rtol=RTOL, centroid_atol=CENTROID_ATOL)

    # Assertions for HDF5 metadata
    assert_attribute(case, "SpecData", spec, expected_spec, "maed_s4",
                     rtol=MAED_RTOL)
    assert_attribute(case, "SpecData", spec, expected_spec, "mask_columns")
    assert_attribute(case, "LCData", lc, expected_lc, "maed_s4_binned",
                     rtol=MAED_RTOL)
    assert_attribute(case, "LCData", lc, expected_lc, "maed_s4_binned_bg",
                     rtol=MAED_RTOL)
