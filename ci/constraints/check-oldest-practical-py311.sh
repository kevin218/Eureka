#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  check-oldest-practical-py311.sh <conda-env> [constraints-file] [--with-hst] [--full-pytest] [--fresh]

Examples:
  ci/constraints/check-oldest-practical-py311.sh eureka_temp
  ci/constraints/check-oldest-practical-py311.sh eureka_temp --fresh
  ci/constraints/check-oldest-practical-py311.sh eureka_temp ci/constraints/oldest-practical-py311.txt --with-hst --full-pytest --fresh

What it does:
  1. Optionally recreates the named conda environment with Python 3.11
  2. Upgrades pip inside the named conda environment
  3. Installs Eureka in editable mode with the selected extras
  4. Runs key import smoke checks
  5. Runs the focused pytest subset
  6. Optionally runs tests/test_WFC3.py when --with-hst is used
  7. Optionally runs full pytest when --full-pytest is used

Notes:
  - The default constraints file is ci/constraints/oldest-practical-py311.txt.
  - The default extras are "jwst,test".
  - With --with-hst, the extras become "jwst,hst,test".
  - Pass a different constraints file to probe an alternate candidate.
  - Use --fresh for clean comparison runs.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ENV_NAME="$1"
shift

CONSTRAINTS_FILE=""
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    CONSTRAINTS_FILE="$1"
    shift
fi

WITH_HST=false
FULL_PYTEST=false
FRESH_ENV=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-hst)
            WITH_HST=true
            ;;
        --full-pytest)
            FULL_PYTEST=true
            ;;
        --fresh)
            FRESH_ENV=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONSTRAINTS_FILE="${REPO_ROOT}/ci/constraints/oldest-practical-py311.txt"
if [[ -z "${CONSTRAINTS_FILE}" ]]; then
    CONSTRAINTS_FILE="${DEFAULT_CONSTRAINTS_FILE}"
fi

if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
    echo "Constraints file not found: ${CONSTRAINTS_FILE}" >&2
    exit 1
fi

EXTRAS="jwst,test"
IMPORTS=("eureka" "astraeus.xarrayIO" "jwst" "stcal" "stdatamodels")

if [[ "${WITH_HST}" == true ]]; then
    EXTRAS="jwst,hst,test"
    IMPORTS+=("image_registration")
fi

PIP_INSTALL_ARGS=(
    python -m pip install
    --timeout 60
    --retries 5
    --upgrade
    -c "${CONSTRAINTS_FILE}"
)

PACKAGE_SPEC="${REPO_ROOT}[${EXTRAS}]"
PIP_INSTALL_ARGS+=(-e "${PACKAGE_SPEC}")

LIGHTCURVE_K="parameter or parameters or model or compositemodel or polynomialmodel or transitmodel or eclipsemodel or sinsoidalmodel or poettr_model or poetecl_model or poetpc_model or lorentzian_model or exponentialmodel or simulation"

echo
echo "==> Using repo root: ${REPO_ROOT}"
echo "==> Using conda env: ${ENV_NAME}"
echo "==> Using extras: ${EXTRAS}"
echo "==> Recreate env first: ${FRESH_ENV}"
echo "==> Using constraints: ${CONSTRAINTS_FILE}"

if [[ "${FRESH_ENV}" == true ]]; then
    echo
    echo "==> Recreating conda env: ${ENV_NAME}"
    conda remove -n "${ENV_NAME}" --all -y >/dev/null 2>&1 || true
    conda create -n "${ENV_NAME}" python=3.11.0 -y
fi

echo
echo "==> Upgrading pip"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo
echo "==> Installing Eureka candidate"
conda run -n "${ENV_NAME}" "${PIP_INSTALL_ARGS[@]}"

echo
echo "==> Running import smoke checks"
IMPORTS_CSV="$(IFS=,; echo "${IMPORTS[*]}")"
conda run -n "${ENV_NAME}" env IMPORTS_CSV="${IMPORTS_CSV}" python - <<'PY'
import importlib
import os

modules = os.environ["IMPORTS_CSV"].split(",")
for name in modules:
    importlib.import_module(name)
    print(f"ok: {name}")
PY

echo
echo "==> Running focused smoke subset"
conda run -n "${ENV_NAME}" python -m pytest tests/test_general.py
conda run -n "${ENV_NAME}" python -m pytest tests/test_lightcurve_fitting.py -k "${LIGHTCURVE_K}"

if [[ "${WITH_HST}" == true ]]; then
    echo
    echo "==> Running HST smoke"
    conda run -n "${ENV_NAME}" python -m pytest tests/test_WFC3.py
fi

if [[ "${FULL_PYTEST}" == true ]]; then
    echo
    echo "==> Running full pytest suite"
    conda run -n "${ENV_NAME}" python -m pytest tests
fi

echo
echo "==> Oldest-practical check completed"
