# Experimental Constraints For Phase 0

This directory is reserved for maintainer-only `py311` exploration artifacts
used to evaluate an `oldest-practical` dependency floor.

These files are intentionally separate from:

- `pyproject.toml`, which remains the canonical dependency metadata source;
- `environment.yml`, which is generated from `pyproject.toml` via UniDep; and
- any future permanent compatibility-testing workflow.

## Why this directory exists

Phase 0 needed a place for candidate pip constraints files without implying
that the repository already promises support for those exact versions.

Constraints files are preferred here because they:

- compose with extras, for example
  `pip install -c ci/constraints/experiments/<candidate>.txt '.[jwst,test]'`;
- avoid churn in `pyproject.toml` while experiments are still in progress; and
- let maintainers pin only the direct dependencies that are being explored.

## Retained Phase 0 recipe

The retained Phase 0 constraints artifact is:

- `ci/constraints/experiments/py311-practical-spec0-window.txt`

This broader recipe was kept after exploration because it successfully
installed, passed import smoke, and passed the full pytest suite with
`.[jwst,hst,test,docs,jupyter,dev]`. It is retained as a tested exploratory
recipe for a broad py311 install surface, not as proof of literal lower bounds
and not as the automatic permanent CI contract.

This file also reflects the project's SPEC 0-inspired Phase 0 approach: prefer
realistic, recently supportable dependency windows over broad untested lower
bounds, while still allowing Eureka-specific exceptions where the astronomy
stack or contributor workflow makes that necessary.

Although this file was validated with a broad install surface, it can still be
used as a constraints file for narrower installs when maintainers want to probe
subsets such as `.[jwst,test]` or `.[jwst,hst,test]`.

Superseded exploratory files were removed once the broader recipe also worked,
including the earlier narrower `py311-practical-spec0-window-r1.txt`.

## Helper script

This directory also includes a small runner script:

- `ci/constraints/experiments/run-phase0-candidate.sh`

Typical usage:

1. Baseline install with no candidate constraints:
   `ci/constraints/experiments/run-phase0-candidate.sh eureka_temp --fresh`
2. Retained recipe with a narrower `jwst,test` install:
   `ci/constraints/experiments/run-phase0-candidate.sh eureka_temp ci/constraints/experiments/py311-practical-spec0-window.txt --fresh`
3. Retained recipe with `jwst,hst,test`:
   `ci/constraints/experiments/run-phase0-candidate.sh eureka_temp ci/constraints/experiments/py311-practical-spec0-window.txt --with-hst --fresh`
4. Full-suite run with `jwst,hst,test`:
   `ci/constraints/experiments/run-phase0-candidate.sh eureka_temp ci/constraints/experiments/py311-practical-spec0-window.txt --with-hst --full-pytest --fresh`

For real candidate comparisons, `--fresh` is recommended so each run starts
from a clean Python 3.11 environment instead of layering installs.

## Suggested validation order

For future exploratory candidates, or when rechecking the retained recipe:

1. Install and resolve with `.[jwst,test]`
2. Run key import smoke checks
3. Run the focused smoke subset
4. Only for finalists, repeat with `.[jwst,hst,test]`
5. Only for finalists, consider full `pytest tests`

That keeps this directory aligned with exploration rather than policy.
