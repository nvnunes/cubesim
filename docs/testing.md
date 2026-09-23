# Testing

This document is the source of truth for verification commands and completion
expectations in `cubesim`.

## Environment

Use the local `./.conda` environment for Python commands and test runs unless a
specific workflow requires something else.

For fresh clones, create or activate a Python 3.12+ environment at `./.conda`
with your preferred environment manager, then install `cubesim` in editable
mode with the `dev` and `docs` extras.

Refresh the local editable install when dependencies or metadata change with:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

## Canonical Verification Commands

Run the full test suite with:

```bash
./.conda/bin/python -m pytest -q
```

Build the complete public documentation surface with strict navigation,
cross-reference, and generated-reference validation:

```bash
./.conda/bin/mkdocs build --strict
```

The real Hybrid integration test in `tests/test_hybrid.py` runs when the
optional `cubesim[hybrid]` extra is installed; otherwise that one test is
skipped. The remaining Hybrid boundary tests run in the base environment.
AO Stats integration tests and execution of `example/etc.ipynb` require the
optional `cubesim[stats]` extra; they are skipped when AO Stats is absent.
The missing-dependency boundary test still runs in the base environment.

## Completion Expectations

Add or adjust tests with every behavior change.

Finish with the full test suite for changes that affect:

- package-root exports
- ETC input or PSF contract validation
- instrument-data loading or PSF normalization behavior
- compute preconditions or result behavior
- README or docs examples that describe public API use
- generated API-reference docstrings or MkDocs navigation

Prefer tests of externally visible behavior over tests coupled to internal
structure.

The full suite executes the root README quickstart and, when their optional
dependencies are installed, every code cell in the bundled example notebooks
from isolated temporary copies. Keep those examples runnable without relying
on generated files or state left in the repository.

Preserve science-visible behavior unless a change is explicitly intended and
documented.
