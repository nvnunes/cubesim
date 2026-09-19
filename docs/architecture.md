# Architecture

This document is the source of truth for `cubesim` package shape, public API
boundaries, data and ETC-input contract ownership, and compute lifecycle. The
detailed instrument directory and file formats are defined in
[`instrument-data.md`](instrument-data.md). The caller-facing lifecycle and
result contract are defined in [`api.md`](api.md) and the linked reference
pages.

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
- `Point`
- `Gaussian`
- `GaussianLines`
- `Sersic`
- `SpatialImage`
- `TabulatedSpectrum`
- `Uniform`
- `ConstantVelocity`
- `RotatingDisk`
- `VelocityField`

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
target-model contracts in `cubesim/models.py`, numerical calculation in
`cubesim/_calculation.py`, immutable result structures in
`cubesim/_result.py`, instrument-definition and ECSV loading in
`cubesim/_instrument.py`, and direct PSF loading in `cubesim/_psf.py`.

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
using the established PSF centering operation, and normalized to unit sum.

## Compute Lifecycle

Keep ETC setup, data loading, validation, and compute clearly separated.

The current setup lifecycle is:

- construct `Etc` with an explicit instrument-data directory
- select an exact scale, selectable disperser leaf, and atmosphere
  with `configure()`
- add one or more composed spatial and spectral targets with `add_target()`
- configure a direct PSF with `set_psf()`
- configure target and sky integrations with `set_exposure()` and optional
  `set_sky_subtraction()`
- optionally register apertures with `add_aperture()`
- execute the deterministic calculation or request noisy realizations with
  `run()`

`run()` always returns immutable S/N and wavelength arrays plus a structured
snapshot of resolved inputs. Models, detector signals, detector variances, and
Poisson-plus-Gaussian noisy cubes are opt-in result groups. Results save as a
complete trusted pickle or as portable FITS datacubes and metadata.

The current implementation covers point, Gaussian, Sersic, supplied
image, and uniform spatial profiles; Gaussian lines in air or vacuum and
tabulated spectra; constant, rotating-disk, and supplied velocity fields;
achromatic direct PSFs; nodding and in-field sky subtraction; and rectangular
or custom apertures. Spatially varying velocity is applied on the
high-resolution model before PSF and line-spread-function convolution.

Non-uniform spatial models represent globally normalized integrated-flux
distributions. Their high- and detector-resolution grids retain only the flux
that falls within the corresponding sampled field; cubesim does not
renormalize a profile after field clipping. For in-field subtraction, the
shared sky estimate is also removed from the expected target signal used for
cube and aperture S/N. The cube-level signal breakdown continues to describe
the raw detected components that determine Poisson noise, while aperture-level
signals describe the sky-subtracted reduction.

Preserve that lifecycle clarity as the repo grows. If a module has a strong
lifecycle or execution flow, prefer method order that follows that lifecycle.
