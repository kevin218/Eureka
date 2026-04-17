# Phase 0: Oldest-Practical Dependency Floor Exploration

This note is for maintainer-facing exploration only. It does not define a
user-facing support policy, and it should not be read as a promise that
Eureka! supports every version allowed by `pyproject.toml`.

## Verified baseline in the repository

As of the packaging and CI modernization work in PRs #835 and #836:

- `pyproject.toml` is the canonical source of dependency metadata.
- `environment.yml` is generated from `pyproject.toml` with UniDep and is
  validated in CI.
- Eureka! has mixed conda and pip dependency handling, including pip-only and
  VCS dependencies.
- Optional dependency groups currently include `jwst`, `hst`, `docs`, `test`,
  `jupyter`, and `dev`.
- Pull request CI already validates generated environments on Linux/macOS for
  Python 3.11-3.12, runs pytest with `.[jwst,hst,test]`, runs build/install
  smoke checks, and keeps `pre-commit` configured separately.

## What "oldest practical" should mean here

For Eureka!, an `oldest-practical-py311` floor should mean:

- a Python 3.11 dependency set that can still be installed and resolved
  without unusual solver gymnastics;
- a small set of direct dependencies pinned low enough to be meaningfully old,
  but not so low that they predate realistic Python 3.11 support;
- a set that passes explicit validation chosen by the project; and
- a floor we can describe honestly as "tested" rather than merely "declared."

This is intentionally different from a literal interpretation of all lower
bounds in `pyproject.toml`. Some declared lower bounds exist to document API
requirements, but they are not necessarily a realistic `py311` compatibility
floor on their own.

SPEC 0 is useful as guidance for what counts as a conservative scientific
Python baseline, but Eureka! should still allow project-specific exceptions for
JWST/HST and other astronomy-specific packages.

## How SPEC 0 informs this phase

Scientific Python [SPEC 0](https://scientific-python.org/specs/spec-0000/)
defines a common time-based minimum-support policy across the ecosystem. In
short:

- Python versions are generally supported for 3 years after their initial
  release.
- Core scientific-package dependency versions are generally supported for
  2 years after their initial release.

That ethos is valuable here because it gives Eureka! a pragmatic way to talk
about "old enough to be conservative" without drifting into indefinite support
for stale combinations that are expensive to test and maintain. For a project
with a scientific stack as broad as Eureka's, that kind of time-window framing
helps reduce maintenance burden, keeps compatibility claims more honest, and
encourages us to test realistic environments rather than theoretical lower
bounds.

Eureka! is not treating SPEC 0 as a hard contract, however. This project has
astronomy-specific dependencies, VCS pins, and workflow expectations that do
not map cleanly onto a single generic support rule. That is why this Phase 0
work is framed as "informed by SPEC 0" rather than "strictly adopts SPEC 0."

One concrete example is Python 3.11: under a strict SPEC 0-style policy, a
Python 3.11 support window would end 3 years after its initial release on
October 24, 2022, which is October 23, 2025. Eureka! has already decided to
continue supporting Python 3.11 for now, so the project is intentionally making
a project-specific exception while still using the broader SPEC 0 maintenance
ethos.

## Phase 0 outcome

Phase 0 stayed exploratory and manual, and it converged on a single retained
constraints artifact:

- `ci/constraints/experiments/py311-practical-spec0-window.txt`

This broader recipe was kept after confirming that it successfully installed,
passed import smoke, and passed the full pytest suite with
`.[jwst,hst,test,docs,jupyter,dev]`.

The narrower earlier runtime-focused recipe was removed after the broader file
also worked. That choice is intentional: a substantial fraction of Eureka
users are also contributors or developer-adjacent users, so the broader install
surface is a useful exploratory artifact to retain.

This retained file is:

- a tested exploratory recipe for a broad py311 dependency surface;
- not proof of literal lower bounds for every package listed in
  `pyproject.toml`; and
- not automatically the permanent required CI lane for compatibility testing.

The later dependency-policy and compatibility-testing PR should still make an
explicit choice about what permanent CI contract, if any, is appropriate.

At the time of Phase 0 testing, this recipe also resolved below NumPy 2. The
only identified direct blockers were `fleck` and `pastasoss`. That is useful
as a description of the current practical Phase 0 outcome, but it should not be
over-read as a durable long-term truth about Eureka!'s support policy.

Both blockers have active upstream work in progress: `fleck` PR #28 is still a
draft PR, and `pastasoss` PR #20 is still open. Local testing against the
`pastasoss` PR also suggested that the package worked after removing its
`numpy<2` cap.

A strict SPEC 0-style reading would already put pressure on Eureka! to move
past this outcome, because `numpy<2` now implies support for dependency
versions that are older than the normal recent-window ethos SPEC 0 is trying to
encourage. That tension is part of why Eureka! is not being strict about SPEC 0
yet: the project still has real astronomy-stack constraints that are moving, and
it is better to describe the current tested outcome honestly than to claim a
clean SPEC 0-aligned position before the surrounding ecosystem is ready. In
other words, this is exactly the kind of ecosystem detail that should keep
support language honest and current rather than hardening a temporary
constraint into project policy.

## Candidate-spec strategy

Use pip constraints files for candidate specs.

Why constraints are the best fit here:

- they compose cleanly with existing extras, for example
  `pip install -c <constraints> '.[jwst,test]'`;
- they do not require changing canonical dependency metadata in
  `pyproject.toml`;
- they let us pin only the packages we are intentionally exploring while
  leaving the rest solver-selected; and
- they keep exploratory "tested floors" separate from declared dependency
  ranges and from the generated conda environment.

Each candidate constraints file should pin the explored direct dependencies
rather than every transitive package. In practice, the retained Phase 0 recipe
is broader than a minimal runtime-only file because it intentionally records a
tested exploratory environment that includes contributor-facing extras.

## Direct dependency groups to explore first

Phase 0 should start with the direct dependencies most likely to define a real
`py311` floor:

- Core scientific stack:
  `numpy`, `scipy`, `matplotlib`, `astropy`, `h5py`, `pandas`, `photutils`
- Eureka!-specific runtime drivers:
  `astraeus-io`, `batman-package`, `dynesty`, `emcee`, `lmfit`, `fleck`,
  `george`, `celerite2`, `svo_filters`
- JWST stack:
  `jwst`, `stcal`, `stdatamodels`, `crds`, `pastasoss`
- HST-specific package:
  `image_registration`

The current hard `jwst==1.20.2` pin should be treated as part of the baseline
unless and until a later focused change is made to that compatibility story.

## Extras included during exploration

The exploration started with narrower `jwst,test` and `jwst,hst,test` passes,
but the retained recipe ultimately broadened to cover
`.[jwst,hst,test,docs,jupyter,dev]`.

That broader retention choice reflects how Eureka is actually used: many users
also touch contributor-facing extras, documentation tooling, or notebook
workflows. Keeping one broader tested exploratory recipe is therefore more
useful than retaining a narrower runtime-only artifact.

## Validation ladder for each candidate

Each candidate should move through the cheapest checks first:

1. Resolve/install:
   `python -m pip install -c <constraints> '.[jwst,test]'`
2. Key imports:
   `python -c "import eureka, astraeus.xarrayIO, jwst, stcal, stdatamodels"`
3. Focused smoke subset:
   `pytest tests/test_general.py`
   `pytest tests/test_lightcurve_fitting.py -k 'parameter or parameters or model or compositemodel or polynomialmodel or transitmodel or eclipsemodel or sinsoidalmodel or poettr_model or poetecl_model or poetpc_model or lorentzian_model or exponentialmodel or simulation'`
4. Finalist-only broader pass:
   `python -m pip install -c <constraints> '.[jwst,hst,test]'`
   `pytest tests/test_WFC3.py`
5. Full `pytest tests` only for one or two finalists that survive the earlier
   steps.

This keeps Phase 0 honest without turning every candidate into a full-matrix
CI job.

## What is retained in the repo

The retained Phase 0 footprint is:

- this maintainer note;
- the lightweight `ci/constraints/experiments/` location and usage notes;
- the single retained broad py311 exploratory recipe,
  `ci/constraints/experiments/py311-practical-spec0-window.txt`; and
- the helper script for rerunning exploratory installs and smoke/full-test
  checks against that constraints file or future candidates.

That keeps the useful Phase 0 evidence in the repo without prematurely
hardening a permanent workflow or support promise.

## What is intentionally deferred

The later dependency-policy + compatibility-testing PR can decide:

- the final `oldest-practical-py311` candidate that Eureka! will actually
  advertise as tested;
- any permanent CI lane for that floor;
- policy language about declared ranges versus tested support;
- whether any `pyproject.toml` lower bounds should be tightened to match the
  tested floor;
- how, if at all, conda-specific floor testing should be represented; and
- any broader packaging cleanup beyond what is needed to support the chosen
  floor honestly.
