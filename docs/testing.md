# Testing

This document is the source of truth for verification commands and completion
expectations in `cubesim`.

## Shared Validation

Use the shared base testing guidance in `astro-agents/validation/base-testing.md`.

## Repo-Local Verification

Use the repo-local verification commands and completion expectations below.

## Environment

Use the local `./.conda` environment for Python commands and test runs unless a
task explicitly requires something else.

For fresh clones, create or activate a Python 3.12+ environment at `./.conda`
with your preferred environment manager, then install `cubesim` in editable
mode with the `dev` extra.

Refresh the local editable install when dependencies or metadata change with:

```bash
./.conda/bin/python -m pip install -e ".[dev]"
```

## Canonical Verification Commands

Run the full test suite with:

```bash
./.conda/bin/python -m pytest -q
```

## Completion Expectations

Add or adjust tests with every behavior change.

Finish with the full test suite for changes that affect:

- package-root exports
- ETC input or PSF contract validation
- artifact loading or normalization behavior
- compute preconditions or result behavior
- README or docs examples that describe public API use

Prefer tests of externally visible behavior over tests coupled to internal
structure.

Preserve science-visible behavior unless a change is explicitly intended and
documented.
