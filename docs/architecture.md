# Architecture

This document is the source of truth for `cubesim` package shape, public API
boundaries, data and ETC-input contract ownership, and compute lifecycle. The
detailed instrument directory and file formats are defined in
[`instrument-data.md`](instrument-data.md).

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
- `Etc`

Do not expand the public API casually. Keep docs and examples aligned with the
supported package-root imports.

## Core Boundaries

Keep clear boundaries between:

- instrument config loading
- target and source model construction
- PSF input ingestion and validation
- background, throughput, QE, detector, signal, noise, and SNR models
- ETC compute and result assembly
- reusable analysis helpers

The current implementation keeps the public lifecycle in `cubesim/etc.py`,
instrument-definition and ECSV loading in `cubesim/_instrument.py`, and direct
PSF loading in `cubesim/_psf.py`. Keep those ownership boundaries as compute
components are added.

## Contract Ownership

Treat instrument configs, their file references, direct PSF inputs, and ETC
inputs as explicit contracts.

- Validate early with actionable errors.
- Avoid silent coercions and hidden fallback behavior.
- Keep one obvious owner per contract.
- Define stable keys and field names as named constants in the narrowest module
  that owns the contract.

An instrument-data directory is explicit caller input and must contain
`etc.ini` at its root. INI file references are relative, must remain within the
directory after symlink resolution, and receive no package-data, current
directory, environment-variable, registry, or network fallback. Detector QE,
atmospheric transmission, and atmospheric background files use the canonical
two-column ECSV schemas and units defined in
[`instrument-data.md`](instrument-data.md).

The direct PSF contract accepts only:

- a FITS image with `PIXSCALE` in mas per pixel
- an NPY 2D array with an explicit positive angular `pixel_scale`
- an in-memory 2D array with an explicit positive angular `pixel_scale`

PSF file paths may be absolute or relative to the instrument-data directory.
Relative paths are never resolved against the current working directory. FITS
inputs take their pixel scale only from `PIXSCALE` and reject an explicit
`pixel_scale`; NPY and in-memory inputs require an explicit angular quantity.

NPY loading uses `allow_pickle=False`. PSFs are copied, validated as finite,
non-negative, nonempty 2D numeric arrays with positive total flux, centred
using the retained legacy ETC operation, and normalized to unit sum.

## Compute Lifecycle

Keep ETC setup, data loading, validation, and compute clearly separated.

The current setup lifecycle is:

- construct `Etc` with an explicit instrument-data directory
- select an exact scale, selectable disperser leaf, and atmosphere
  with `configure()`
- configure a direct PSF with `set_psf()`

The current implementation provides construction, instrument-option selection,
and direct PSF input. The reserved `run()` entrypoint raises `NotImplementedError`
because forward computation is not implemented.

Preserve that lifecycle clarity as the repo grows. If a module has a strong
lifecycle or execution flow, prefer method order that follows that lifecycle.
