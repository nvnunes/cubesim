# Architecture

This document is the source of truth for `cubesim` package shape, public API
boundaries, artifact and ETC-input contract ownership, and compute lifecycle.

## Shared Validation And Skills

This project uses runtime-discovered `astro-agents` skills for shared review and authoring support:

- `$agent-surface-review`
- `$documentation-surface-review` with the `public-python` profile
- `$code-quality-review`
- `$python-code-writing`

Repo-local package boundaries, public API choices, contract rules, and
exceptions in this document remain the source of truth for this repo.

## Package Surface

`cubesim` is the deliberate public Python package boundary.

- Re-export only supported user-facing entrypoints from `cubesim.__init__`.
- Keep lower-level ETC implementation and validation helpers behind the package
  root.
- Keep CLI or scripts as thin wrappers over the Python API if they are added.
- Keep project-specific behavior out of `cubesim`.

## Current Public API

The current package-root API exports:

- `__version__`
- `EtcOptions`
- `EtcResult`
- `compute`

Do not expand the public API casually. Keep docs and examples aligned with the
supported package-root imports.

## Core Boundaries

Keep clear boundaries between:

- instrument config loading
- target and source model construction
- PSF artifact ingestion and validation
- background, throughput, QE, detector, signal, noise, and SNR models
- ETC compute and result assembly
- reusable analysis helpers

The current scaffold keeps early API, option state, PSF contract handling, and
compute precondition checks in `cubesim/etc.py`. As the package grows, split
coherent concerns into narrower modules that preserve the boundaries above
rather than turning `etc.py` into the permanent owner of every concern.

## Contract Ownership

Treat instrument configs, PSF artifact references, PSF payload keys, and ETC
inputs as explicit contracts.

- Validate early with actionable errors.
- Avoid silent coercions and hidden fallback behavior.
- Keep one obvious owner per contract.
- Define stable keys and field names as named constants in the narrowest module
  that owns the contract.

The current PSF payload contract requires:

- `psf`
- `pixel_scale`

Current payload validation rules are:

- `psf` must normalize to a 2D finite non-negative array with strictly
  positive total flux
- `pixel_scale` must be scalar-like and strictly positive

## Compute Lifecycle

Keep ETC setup, artifact loading, validation, and compute clearly separated.

The current lifecycle is:

- configure options through `config_*` methods
- attach or load PSF data through `set_psf()` or `load_psf()`
- mark the setup complete with `config_finish()`
- run `compute()`

Preserve that lifecycle clarity as the repo grows. If a module has a strong
lifecycle or execution flow, prefer method order that follows that lifecycle.
