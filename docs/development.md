# Development Setup

This document covers local environment setup and daily commands. For repo
structure and contract ownership, use `docs/architecture.md`. For canonical
verification commands and completion expectations, use `docs/testing.md`.

## Shared Guidance

This repo adopts the shared guidance in
`astro-agents/guidance/public-python-projects.md` and
`astro-agents/guidance/python-development.md`.

Repo-local environment setup and daily commands in this document remain the
source of truth for this repo.

## Environment

Use the local `./.conda` environment for Python commands and test runs unless a
task explicitly requires something else.

For fresh clones, create or activate a Python 3.12+ environment at `./.conda`
with your preferred environment manager. This repo does not prescribe one
specific manager yet.

After the environment is ready, install the package in editable mode with the
`dev` extra:

```bash
./.conda/bin/python -m pip install -e ".[dev]"
```

## Daily Commands

After the editable install is current, prefer commands from the local
environment instead of bare `python` or `pip` invocations:

```bash
./.conda/bin/python -m pip install -e ".[dev]"
./.conda/bin/python -m pytest -q
```
