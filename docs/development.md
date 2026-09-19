# Development Setup

This document covers local environment setup and daily commands. For repo
structure and contract ownership, use
[`architecture.md`](architecture.md). For canonical verification commands and
completion expectations, use [`testing.md`](testing.md).

## Environment

Use the local `./.conda` environment for Python commands and test runs unless a
specific workflow requires something else.

For fresh clones, create or activate a Python 3.12+ environment at `./.conda`
with your preferred environment manager. This repo does not prescribe one
specific manager yet.

After the environment is ready, install the package in editable mode with the
`dev` and `docs` extras:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

## Daily Commands

After the editable install is current, prefer commands from the local
environment instead of bare `python` or `pip` invocations:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
./.conda/bin/python -m pytest -q
./.conda/bin/mkdocs build --strict
./.conda/bin/mkdocs serve
```

The strict docs build is included in canonical verification. The local server is
for previewing the same navigation and generated API reference while editing.

## Pre-commit Hook

The repo includes a versioned hook at `.githooks/pre-commit`. On each commit,
it runs the full test suite and the strict MkDocs build.

Activate the versioned hooks path once per clone:

```bash
git config core.hooksPath .githooks
```
