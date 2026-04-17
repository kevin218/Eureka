# Dependency Policy

Eureka! distinguishes between dependency metadata, tested compatibility
recipes, and the generated conda environment. This page describes how those
pieces fit together and what the project currently means by "supported".

## Canonical dependency metadata

`pyproject.toml` is the canonical source of Eureka!'s dependency metadata.
Direct dependency declarations, optional dependency groups, and packaging
configuration should be maintained there.

Eureka currently defines optional dependency groups including:

- `jwst`
- `hst`
- `docs`
- `test`
- `jupyter`
- `dev`

Those groups describe install surfaces that users and maintainers may choose to
combine. They do not, by themselves, mean that every theoretically valid
version combination inside the declared metadata ranges is tested.

## Generated conda environment

The repository's `environment.yml` is generated from `pyproject.toml` via
UniDep. It is a repository-managed conda environment artifact and should not be
hand-edited.

That file is useful for contributors and for users who want the maintained
repository environment, but it serves a different purpose from package metadata
and from the tested oldest-practical recipe described below.

## How Eureka uses SPEC 0

Eureka is informed by [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/),
but it is not strictly bound to it.

SPEC 0 provides a useful maintenance ethos: prefer realistic, recently
supportable dependency windows, test them explicitly, and avoid claiming broad
compatibility that is not exercised in CI. That is especially helpful for a
scientific Python package with a large ecosystem footprint.

Eureka still needs project-specific exceptions. The astronomy stack includes
dependencies, VCS pins, and ecosystem transitions that do not always line up
cleanly with a strict generic support window. For that reason, the project uses
SPEC 0 as guidance rather than as a rigid contract.

## What "supported" means

For Eureka, "supported" means "tested in CI".

In practical terms, the repository distinguishes between three different kinds
of statements:

1. Declared dependency ranges in `pyproject.toml`
2. The tested oldest-practical recipe
3. The generated conda environment in `environment.yml`

These are related, but they are not identical.

### Declared dependency ranges

The version ranges in `pyproject.toml` are package metadata. They express the
current install intent of the project and sometimes reflect API assumptions or
known packaging constraints.

They are not a promise that every resolver-valid in-range combination is known
to work.

### Tested oldest-practical recipe

The current tested oldest-practical recipe is:

- `ci/constraints/oldest-practical-py311.txt`

This file records one evidence-backed Python 3.11 dependency set that the
project intentionally exercises in CI.

This recipe is important because it gives the project a concrete tested floor
story. It also helps keep support language honest by tying the idea of
"oldest practical" to an actual reproducible environment rather than to
declared lower bounds alone.

At the same time, this file should not be over-read:

- it is not proof that every lower bound in `pyproject.toml` is literal;
- it is not a claim that every in-range solve is supported; and
- it is not the same thing as the generated conda environment.

### Generated conda environment

`environment.yml` captures the repository-managed conda environment derived from
`pyproject.toml`. It is meant to provide a maintained environment for users and
contributors who prefer that route.

Because pip and conda solve differently, Eureka does not require those two
paths to be identical at every version edge. What matters is that both are
maintained honestly and tested in the ways the project explicitly documents.

## Current NumPy story

For Python 3.11, `numpy>=1.25` is the honest current floor in Eureka's package
metadata.

The current tested oldest-practical outcome still resolves below NumPy 2. That
is the practical tested result today, not a permanent long-term truth about
Eureka's policy. Upstream blockers are changing, so this should be revisited as
the surrounding ecosystem moves.

## CI expectations

The permanent compatibility lane is intentionally small. The current tested
floor is exercised in CI with:

- Python 3.11.0
- the oldest-practical constraints file
- `.[jwst,hst,test,docs,jupyter,dev]`
- the full `pytest tests` suite

That single permanent lane is intentional. It keeps the CI contract anchored to
one tested oldest-practical solve rather than to every resolver-valid
combination.
