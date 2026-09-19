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

Preserve science-visible behavior unless a change is explicitly intended and
documented.
