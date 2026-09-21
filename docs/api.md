# Python API Documentation

This guide describes the primary code-first exposure-time calculation exposed
at `cubesim`. CubeSim separates the reusable calculation framework from
instrument definitions and scientific assets. Begin with an instrument-data
directory that follows the [instrument-data contract](instrument-data.md).

## Configure An ETC

Construct `Etc` with the directory that contains `etc.ini`, then select one
scale, disperser, and atmosphere by their exact configured names:

```python
import astropy.units as u
import cubesim

etc = cubesim.Etc("/path/to/instrument-data")
etc.configure(
    scale="50mas",
    disperser="r3000.yj",
    atmosphere="airmass10_pwv10",
)
```

Relative paths in `etc.ini` and relative PSF paths are resolved against this
directory. CubeSim does not search package data, the current directory,
environment variables, registries, or the network.

Calling `configure()` again replaces the selected instrument state. Existing
targets, exposures, apertures, and the PSF remain configured until explicitly
replaced or a new `Etc` is constructed.

## Set The Pointing

The default pointing uses a position angle of zero and angular offsets rather
than an absolute sky coordinate. Position angles are measured east of north.

```python
from astropy.coordinates import SkyCoord

etc.set_pointing(
    position_angle=30 * u.deg,
    center=SkyCoord("12h00m00s", "+30d00m00s"),
)
```

The optional scalar `SkyCoord` supplies celestial WCS metadata when a result is
saved as FITS. Without it, FITS products use east and north angular offsets.

## Build Targets

A target explicitly combines a position, spatial profile, spectrum, and
optional line-of-sight velocity model:

```python
target = cubesim.Gaussian(
    fwhm=0.2 * u.arcsec,
    axis_ratio=0.7,
    position_angle=15 * u.deg,
)
spectrum = cubesim.GaussianLines(
    rest_wavelength=0.65646 * u.micron,
    redshift=0.8,
    medium="vacuum",
    flux=2e-17 * u.erg / (u.s * u.cm**2),
    dispersion=70 * u.km / u.s,
)
velocity = cubesim.RotatingDisk(
    maximum_velocity=180 * u.km / u.s,
    turnover_radius=0.25 * u.arcsec,
    inclination=50 * u.deg,
    position_angle=65 * u.deg,
)

etc.add_target(
    position=(0.1 * u.arcsec, -0.2 * u.arcsec),
    spatial=target,
    spectrum=spectrum,
    velocity=velocity,
)
```

Target positions are `(east, north)` angular offsets. Add as many targets as
needed; their detector-resolution models are combined into one target cube.

Non-uniform profiles use integrated flux. `Uniform` is the only spatial model
that uses surface-brightness flux. Non-uniform profiles are normalized over the
full model plane before the IFU footprint is applied, so flux outside the field
of view is not renormalized back into the detector.

See the [target-model reference](reference/models.md) for analytic profiles,
FITS and NPY image or velocity inputs, tabulated spectra, and the complete unit
contracts.

File-backed target models resolve relative paths from the process working
directory, rather than from the instrument-data directory. Construct them with
absolute paths when the working directory is not controlled. Their required
array dimensions, FITS headers, table columns, units, wavelength-medium
metadata, and validation rules are defined in the
[target-model reference](reference/models.md).

## Set The PSF

Every calculation containing a non-uniform target requires one achromatic PSF:

```python
etc.set_psf("psf.fits")
```

FITS PSFs carry `PIXSCALE` in milliarcseconds per pixel. NPY and in-memory
two-dimensional arrays require an explicit angular pixel scale:

```python
etc.set_psf("psf.npy", pixel_scale=10 * u.mas)
```

The PSF must be finite, nonnegative, and have positive total flux. CubeSim
measures its centroid from a copy smoothed by a 0.5-pixel Gaussian, using pixel
centers as coordinates. If either centroid coordinate is at least 0.01 pixel
from the geometric center, the original array is shifted with cubic
interpolation and zero-filled boundaries. Negative interpolation residuals are
clipped, and the centered PSF is normalized to unit total. Calling
`set_psf()` again replaces the previous PSF.

## Configure Sky Subtraction

Nodding with an `AB` sequence is the default. A sequence consists of `A` target
and `B` sky exposures and must contain both symbols:

```python
etc.set_sky_subtraction(method="nodding", sequence="ABBA")
```

For in-field subtraction, supply a nonempty two-dimensional Boolean mask with
the selected detector shape. `True` spaxels define the sky sample:

```python
etc.set_sky_subtraction(method="in_field", mask=sky_mask)
```

In-field subtraction removes a mean sky estimate wavelength by wavelength.
The mask declares target-free spaxels: their target signal is zero in the
detector calculation, and cubesim does not model target contamination of the
sky estimate. A registered science aperture cannot overlap the sky mask.
Signals have the same detector-component meaning at cube and aperture level.

## Set The Exposure

Supply the duration of one frame and exactly one count:

```python
etc.set_exposure(time=600 * u.s, n_target=8)
```

For nodding, `n` counts all frames and `n_target` counts only `A` frames. The
chosen count must represent a whole number of configured sequences. For
in-field subtraction, every frame is a target frame, so either count resolves
to the same total and there are no separate sky frames.

The resolved result options contain the single-frame time, total frame count,
target and sky frame counts, on-target time, sky time, and total integration
time.

## Add Apertures

Apertures are evaluated during the run so they are retained in saved results.
They follow the public cube axis order `(y, x, wavelength)`.

Create a rectangular aperture with integer-pixel size and either a center or a
start coordinate:

```python
etc.add_aperture(
    name="line-core",
    size=(5, 5, 7),
    center=(19.5, 19.5, 1.18 * u.micron),
)
```

The spatial coordinates are detector-pixel numbers. The spectral coordinate
may be a detector-pixel number or scalar wavelength. `center` and `start` are
mutually exclusive. Coordinates are converted to pixel bounds using NumPy
round-to-nearest behavior. If neither placement is supplied, the aperture uses
the central detector pixel spatially and the first wavelength of the first
target's spectrum spectrally. A custom aperture instead accepts a nonempty
three-dimensional Boolean mask with the exact output-cube shape:

```python
etc.add_aperture(name="custom", mask=aperture_mask)
```

Any aperture extending outside the result cube is rejected.

## Run The Calculation

The basic calculation returns the S/N cube, the detector signal and variance
components needed to interpret and sample it, and a snapshot of the resolved
configuration:

```python
result = etc.run()
print(result.snr.shape)
print(result.wavelength)
```

Detailed model products are optional:

```python
result = etc.run(
    include_models=True,
)
```

- `include_models` adds per-target high- and low-resolution model components,
  the combined detector-resolution target cube, and the transmission, sky, and
  thermal models sampled on the detector wavelength grid.

Every calculation can draw random Poisson-plus-Gaussian, sky-subtracted data
after the deterministic run:

```python
sampled_cube = result.sample(n=10, seed=42)
noise = sampled_cube.data - result.signals.target
sampled_cube.save("sampled_cube.fits")
```

`result.sample()` defaults to one realization. Its `data` has shape
`(y, x, wavelength)`; values of `n` greater than one add a leading realization
axis. The returned object retains its random seed, wavelength, options, and
FITS persistence. Registered apertures provide the corresponding integrated
operation:

```python
aperture_sample = result.apertures[0].sample(n=1000, seed=91)
aperture_noise = aperture_sample.data - result.apertures[0].signals.target
aperture_sample.save("aperture_samples.fits")
```

When `seed` is omitted, CubeSim generates one and retains it on the returned
sample object.

See [Results And Persistence](reference/results.md) for the complete immutable
result layout, units, aperture products, and save formats.

## Plot Results

Plotting functions live in a separate public submodule and consume only an
immutable result:

```python
import cubesim.plotting as plotting

figure = plotting.plot_snr(result, wavelength=1.1 * u.micron)
```

Each function creates and returns a Matplotlib `Figure`. It does not show,
save, or close the figure, so notebooks and applications retain control of
display and output. Composite diagnostics and their individual panels cover
the configured PSF, target and background models, detected signal components,
S/N, and aperture signal and S/N projections. Functions that need optional
model products report that `Etc.run(include_models=True)` is required.

Detector positions use `(y, x)` integer coordinates. Inclusive
`((y_min, y_max), (x_min, x_max))` bounds sum a rectangular detector region.
Detector maps use edge coordinates from zero to the number of spaxels so all
major and minor ticks lie on pixel boundaries. Zero-based spaxel index `i`
occupies map coordinates `i` through `i + 1`.
Wavelength selectors may be zero-based detector indices or scalar spectral
quantities; quantity values select the nearest detector sample. A two-element
spectral quantity such as `(1.095, 1.105) * u.micron` sums all detector samples
whose central wavelengths fall within that inclusive interval.

When `position` and a wavelength range are supplied together, the position
selects the spaxel or spatial region used to form a spectrum and the wavelength
range limits that spectrum's displayed interval. A scalar wavelength cannot be
combined with `position` because it identifies only one voxel rather than a
spectrum.

Aperture plotting functions select a registered aperture by zero-based index,
unique name, or its retained result object. These calls are equivalent:

```python
plotting.plot_aperture_snr(result, aperture=0)
plotting.plot_aperture_snr(result, aperture="galaxy")
plotting.plot_aperture_snr(result, aperture=result.apertures[0])
```

An aperture object must belong to the supplied result.

The aperture S/N functions also accept a sequence through `aperture`. Multiple
apertures are plotted as a vertical stack, with one spectrum and map per row in
the composite plot. Their maps share one color normalization:

```python
plotting.plot_aperture_snr(
    result,
    aperture=("core", "redshifted", "blueshifted"),
    cbar_range=(0, 50),
    y_range=(0, 100),
)
```

Every map-capable plotting function accepts
`cbar_range=(minimum, maximum)`, and every spectrum-capable function accepts
`y_range=(minimum, maximum)`. Functions that can produce both kinds of output
accept both parameters. In a multi-panel figure, each range applies to all
panels of its corresponding kind. The same colorbar range can therefore be
supplied to an S/N overview and its aperture diagnostics, while one vertical
range makes all aperture spectra directly comparable. Logarithmic maps and
spectra require positive limits.

For mode-dependent signal and S/N functions, a position selection produces a
spectrum and a wavelength selection produces a map. Supplying `cbar_range` for
spectrum output or `y_range` for map output is an error because the requested
axis is not present.

Signal and S/N plotters accept an optional sequence of registered apertures as
an annotation layer:

```python
plotting.plot_snr(
    result,
    wavelength=(1.095, 1.105) * u.micron,
    apertures=(0, "redshifted", result.apertures[2]),
)
```

Map output outlines each aperture's spatial support at exact spaxel
boundaries. Spectrum output marks the start and end of each aperture's
wavelength support with vertical lines. Aperture colors remain stable according
to their order in `result.apertures`. When neither `position` nor `wavelength`
is supplied, selected apertures with the same contiguous spectral support
automatically select that wavelength range and produce a map. If their spectral
support differs, provide a wavelength range explicitly. An explicit `position`
still produces a spectrum and does not require matching aperture support. A map
omits any aperture that has no spectral overlap with its displayed wavelength
selection. Aperture overlays do not otherwise change the plotted calculation
or reduction.

Signal plots use the always-present detector signal cubes. Scalar S/N plots use
`result.snr`; pairing a scalar position with a wavelength range limits that
existing S/N spectrum. S/N plots that reduce a position or wavelength range
recompute S/N from the summed target signal and propagated variance. See the
[plotting reference](reference/plotting.md) for all functions and signatures.

## Save Results

Use the filename suffix to select a supported format:

```python
result.save("calculation.pkl")
result.save("calculation.fits")
```

Pickle preserves the complete result object and is appropriate only for trusted
files used with compatible Python, dependency, and CubeSim versions. FITS
stores the S/N, target-signal, background, and total-variance cubes with their
wavelength coordinate, units, WCS, and calculation metadata. Existing files
are rejected unless `overwrite=True`.

## Validation Behavior

CubeSim validates each input at the narrowest useful boundary:

- instrument syntax and referenced files are validated when `Etc` is
  constructed;
- spatial, spectral, and velocity model values are validated when their model
  objects are constructed;
- target unit compatibility is validated by `add_target()`;
- selected-grid coverage, velocity-field coverage, sky-mask shape, exposure
  sequence counts, aperture bounds, and missing lifecycle steps are validated
  by `run()`.

Invalid values raise ordinary Python, NumPy, or Astropy exceptions with a
contract-specific message. CubeSim does not silently clip unsupported inputs,
extrapolate tabulated spectra or instrument tables, or substitute missing
configuration.

## Result Independence

Results are immutable snapshots. Arrays and quantities are read-only, and the
resolved options own their target configuration. Reconfiguring or reusing the
originating `Etc` cannot retroactively change an existing result.

The current API implements forward exposure-time calculations with one
wavelength-independent direct PSF. Inverse calculations and
wavelength-dependent PSFs are outside the current public contract.

See the [ETC reference](reference/etc.md) for method signatures and parameter
details.
