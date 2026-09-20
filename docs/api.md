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
Cube-level signal products remain the raw detected components that determine
Poisson noise, while aperture signal products report sky-subtracted
measurements.

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
target's spectrum spectrally. A custom aperture instead accepts a
three-dimensional
Boolean mask with the exact output-cube shape:

```python
etc.add_aperture(name="custom", mask=aperture_mask)
```

Any aperture extending outside the result cube is rejected.

## Run The Calculation

The basic calculation returns only the S/N cube and what is needed to interpret
it:

```python
result = etc.run()
print(result.snr.shape)
print(result.wavelength)
```

Optional groups expose increasingly detailed products:

```python
import numpy as np

result = etc.run(
    include_models=True,
    include_signals=True,
    include_variances=True,
    include_data=True,
    n_cubes=10,
    rng=np.random.default_rng(42),
)
```

- `include_models` adds per-target high- and low-resolution model components,
  the combined detector-resolution target cube, and the transmission, sky, and
  thermal models sampled on the detector wavelength grid.
- `include_signals` adds target, sky, thermal, dark, background, and total
  detected signals.
- `include_variances` adds target, sky, thermal, dark, read, and total detector
  variances.
- `include_data` adds random noisy, sky-subtracted detector cubes.

`n_cubes` and `rng` are valid only when data are requested. With one
realization, `result.data` is a three-dimensional cube. Multiple realizations
add a leading axis with shape `(n_cubes, y, x, wavelength)`.

See [Results And Persistence](reference/results.md) for the complete immutable
result layout, units, aperture products, and save formats.

## Save Results

Use the filename suffix to select a supported format:

```python
result.save("calculation.pkl")
result.save("calculation.fits")
```

Pickle preserves the complete result object and is appropriate only for trusted
files used with compatible Python, dependency, and CubeSim versions. FITS
preserves the science datacubes and their units, WCS, aperture products, and
interpretive metadata; it is not a complete reconstruction of every Python
model object. Existing files are rejected unless `overwrite=True`.

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
