# AGENTS.md

## Scope
- Documentation surface profile: public-python.

## Source Of Truth Docs
- Follow `README.md` for the repo's public summary and starting docs.
- Follow `docs/architecture.md` for package shape, public API boundaries, artifact and ETC-input contract ownership, and compute lifecycle.
- Follow `docs/testing.md` for canonical verification commands and completion expectations.
- Follow `docs/development.md` for local environment setup and daily commands.

## Shared Validation
- Use `$agent-surface-review` for shared agent-surface review.
- Use `$documentation-surface-review` for documentation-surface review with the `public-python` profile.
- Use `$code-quality-review` for source-code quality review.

## Skill Requirements
- For Python code, use `$python-code-writing`.
- For project docs such as `docs/architecture.md`, `docs/testing.md`, `docs/development.md`, and similar long-lived project documents, use `$project-docs-writing`.
- For `README.md`, use `$readme-writing`.
- For plan documents or phased execution docs when they are created or revised, use `$plan-writing`.

## Working Rules
- For package structure, public API boundaries, persisted contracts, and compute-lifecycle-sensitive changes, consult `docs/architecture.md` before editing.
- Before concluding substantial work, satisfy the verification expectations in `docs/testing.md`.
- Use the local `./.conda` environment and the workflow in `docs/development.md` for Python commands and test runs unless a task explicitly requires something else.

## Review Lens
- Favor contract ownership, lifecycle clarity, and preservation of science-visible behavior in review.
