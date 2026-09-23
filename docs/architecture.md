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
- PSF measurement diagnostics
- result visualization

The current implementation keeps the public lifecycle in `cubesim/etc.py`,
target-model contracts in `cubesim/models.py`, numerical calculation in
`cubesim/_calculation.py`, immutable result structures in
`cubesim/_result.py`, shared in-field variance propagation in
`cubesim/_variance.py`, instrument-definition and ECSV loading in
`cubesim/_instrument.py`, Hybrid orchestration in `cubesim/_hybrid.py`, direct
PSF loading in `cubesim/_psf.py`, and public
result visualization in `cubesim/plotting.py`. Plotting is a submodule API and
is intentionally not re-exported from the package root. PSF measurements live
in the separate public `cubesim/diagnostics.py` submodule and delegate their
numerical algorithms to the optional `ao-stats` dependency, loaded only when
`psf_stats()` is called. Plotting without supplied statistics needs no AO Stats.

## Contract Ownership

Treat instrument configs, their file references, direct PSF inputs, and ETC
inputs as explicit contracts.

The public [coordinate and array conventions](api.md#coordinate-and-array-conventions)
distinguish sky offsets, detector Cartesian axes, and NumPy index order. Keep
their conversion at an explicit boundary when extending positioning or PSF
integration.

- Validate early with actionable errors.
- Avoid silent coercions and hidden fallback behavior.
- Keep one obvious owner per contract.
- Define stable keys and field names as named constants in the narrowest module
  that owns the contract.

An instrument-data directory is explicit caller input and must contain
`etc.ini` at its root. Relative input paths resolve from the current working
directory, then from the nearest `pyproject.toml` project root when the current
path does not exist. INI file references are relative to the instrument-data
directory; symbolic links are followed normally. References receive no further
package-data, current-directory, environment-variable, registry, or network
fallback. Detector QE, atmospheric transmission, and atmospheric background
files use the canonical two-column ECSV schemas and units defined in
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
An optional instrument `[hybrid]` section supplies a magnitude zeropoint and
three instrument-data-relative assets. `cubesim[hybrid]` is imported only for
an active Hybrid-backed `run()`. Hybrid owns magnitude-to-photon conversion,
MASTSEL interpretation, interpolator loading, and AO PSF computation. CubeSim
resolves current pointing geometry, requests one science PSF, rotates its
`[y, x]` image through the relative IFU angle, and passes it through the
existing PSF validation path. Direct PSFs are already detector-oriented.

## Compute Lifecycle

Keep ETC setup, data loading, validation, and compute clearly separated.

The current setup lifecycle is:

- construct `Etc` with an explicit instrument-data directory
- select an exact scale, selectable disperser leaf, and atmosphere
  with `configure()`
- optionally set an absolute telescope pointing and place or rotate the IFU
  within its field of regard
- add one or more composed spatial and spectral targets with `add_target()`
- configure a direct PSF with `set_psf()` or a pending Hybrid PSF with
  `set_hybrid_psf()`; the last setter wins
- configure target and sky integrations with `set_exposure()` and optional
  `set_sky_subtraction()`
- optionally register apertures with `add_aperture()`
- execute the deterministic calculation with `run()`
- draw noisy realizations from the result with `sample()`

`run()` always returns immutable S/N, wavelength, signal, and variance arrays
plus a structured snapshot of resolved inputs. A configured PSF is retained
with its normalized image, angular pixel scale, telescope diameter, and any
available modelling wavelength and pupil. Detailed model
products are opt-in through `include_models=True`. Results can draw
Poisson-plus-Gaussian noisy cubes without mutating the deterministic result.
Cube and aperture sampling return immutable objects that own their data, random
seed, interpretation metadata, and FITS persistence.
Every registered aperture retains integrated signal and variance components,
wavelength and detector-position projections, and compact state for drawing
integrated samples. Results save as a complete trusted pickle or as portable
FITS datacubes and metadata.

The current implementation covers point, Gaussian, Sersic, supplied
image, and uniform spatial profiles; Gaussian lines in air or vacuum and
tabulated spectra; constant, rotating-disk, and supplied velocity fields;
achromatic direct or Hybrid-modelled PSFs; nodding and in-field sky subtraction; and rectangular
or custom apertures. Spatially varying velocity is applied on the
high-resolution model before PSF and line-spread-function convolution.

Non-uniform spatial models represent globally normalized integrated-flux
distributions. Their high- and detector-resolution grids retain only the flux
that falls within the corresponding sampled field; cubesim does not
renormalize a profile after field clipping. For in-field subtraction, the
sky mask declares target-free detector spaxels. Target signal is zero there,
and registered science apertures cannot overlap those spaxels. Sky subtraction
is an IFU-level operation; aperture signals directly reduce the cube-level
signal components, while aperture variance propagation retains covariance from
the shared sky estimate.

Preserve that lifecycle clarity as the repo grows. If a module has a strong
lifecycle or execution flow, prefer method order that follows that lifecycle.
