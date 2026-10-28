"""Shared pytest configuration for Eureka tests."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Import this checkout before regression fixtures and test modules load.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

# Set this before test modules and regression fixtures import Matplotlib.
# Respect an explicit configuration directory supplied by the caller.
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

# Import Matplotlib only after MPLCONFIGDIR is set, then render test figures
# without requiring a GUI or a working Tk installation.
import matplotlib  # noqa: E402

matplotlib.use("Agg")


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


def is_unit_test(item):
    """Return True for tests stored under tests/unit."""
    return 'unit' in Path(str(item.fspath)).parts


def pytest_collection_modifyitems(items):
    """Run unit tests, other general tests, stage regressions, then
    instrument end-to-end tests."""
    instrument_order = [
        "test_MIRI", "test_NIRCam", "test_NIRSpec",
        "test_NIRCamPhotometry", "test_NIRCamPhotometry_hex",
        "test_WFC3", "test_NIRISS", "test_S3opt", "test_S4cal",
    ]
    instrument_rank = {name: rank
                       for rank, name in enumerate(instrument_order)}

    def test_order(item):
        if is_unit_test(item):
            return (0, 0)
        name = getattr(item, "originalname", None) or item.name
        if name in instrument_rank:
            return (3, instrument_rank[name])
        if "RegressionTests" in item.path.parts:
            return (2, 0)
        return (1, 0)

    # Stable sorting preserves collection order within each group and keeps
    # every parametrized case, even when multiple items share a test name.
    items.sort(key=test_order)
