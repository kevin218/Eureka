"""Shared pytest configuration for Eureka tests."""
import sys

import pytest


def pytest_addoption(parser):
    """Register options used by Eureka's integration tests."""
    parser.addoption(
        "--keep-s3-output",
        action="store_true",
        default=False,
        help=("Preserve Stage 3 output directories after integration tests. "
              "Use this only when creating or inspecting regression "
              "reference data."),
    )
    parser.addoption(
        "--keep-s2-output",
        action="store_true",
        default=False,
        help=("Preserve Stage 2 output directories after integration tests. "
              "Use this only when preparing fixed Stage 2 inputs for Stage "
              "3 regression tests."),
    )
    parser.addoption(
        "--overwrite-ref-files",
        "--overwrite_ref_files",
        action="store_true",
        dest="overwrite_ref_files",
        default=False,
        help=("Replace the selected regression-test reference files with "
              "their current outputs after an interactive confirmation."),
    )
    parser.addoption(
        "--keep-s4-output",
        action="store_true",
        default=False,
        help=("Preserve Stage 4 output directories after integration tests. "
              "Use this only when creating or inspecting regression "
              "reference data."),
    )


@pytest.fixture
def keep_s3_output(pytestconfig):
    """Whether a test should retain its generated Stage 3 products."""
    return pytestconfig.getoption("--keep-s3-output")


@pytest.fixture
def keep_s2_output(pytestconfig):
    """Whether a test should retain its generated Stage 2 products."""
    return pytestconfig.getoption("--keep-s2-output")


@pytest.fixture(scope="session")
def overwrite_ref_files(pytestconfig):
    """Confirm whether selected regression references may be overwritten."""
    if not pytestconfig.getoption("overwrite_ref_files"):
        return False

    stdin = sys.__stdin__
    stderr = sys.__stderr__
    if not stdin.isatty():
        raise pytest.UsageError(
            "--overwrite-ref-files requires an interactive terminal so its "
            "confirmation prompt cannot be bypassed."
        )

    stderr.write(
        "\nWARNING: This run will replace the Git-tracked reference file(s) "
        "for every selected regression case. Review the resulting Git diff "
        "before committing.\n"
        "Type OVERWRITE to continue: "
    )
    stderr.flush()
    if stdin.readline().strip() != "OVERWRITE":
        raise pytest.UsageError("Reference overwrite cancelled.")
    return True


@pytest.fixture
def keep_s4_output(pytestconfig):
    """Whether a test should retain its generated Stage 4 products."""
    return pytestconfig.getoption("--keep-s4-output")


def pytest_collection_modifyitems(session, config, items):
    """Modifies test items to ensure test functions run in a given order

    Parameters
    ----------
    session : pytest.Session
        The pytest session object.
    config : pytest.Config
        The pytest config object.
    items : List[pytest.Item]
        List of item objects.
    """
    function_order = ["test_trim", "test_medstddev",
                      "test_parameter", "test_parameters", "test_model",
                      "test_compositemodel", "test_polynomialmodel",
                      "test_transitmodel", "test_eclipsemodel",
                      "test_sinsoidalmodel", "test_poettr_model",
                      "test_poetecl_model", "test_poetpc_model",
                      "test_lorentzian_model", "test_exponentialmodel",
                      "test_simulation",
                      "test_MIRI", "test_NIRCam", "test_NIRSpec",
                      "test_NIRCamPhotometry", "test_NIRCamPhotometry_hex",
                      "test_WFC3"]
    item_names = [item.name for item in items]
    function_mapping = {item.name: item for item in items
                        if item.name in function_order}
    extra_functions = [item for item in items
                       if item.name not in function_order]

    sorted_items = []
    for func_ in function_order:
        if func_ in item_names:
            sorted_items.append(function_mapping[func_])
    sorted_items.extend(extra_functions)

    items[:] = sorted_items
