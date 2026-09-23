# Results And Persistence

`Etc.run()` returns an immutable result that is independent of the originating
`Etc`. Arrays and quantities are copied into read-only storage. Reconfiguring
the ETC does not change an existing result.

## Core Products

Every result contains:

| Attribute | Meaning | Shape |
| --- | --- | --- |
| `snr` | Dimensionless signal-to-noise cube | `(y, x, wavelength)` |
| `wavelength` | Detector wavelength coordinate | `(wavelength,)` |
| `options` | Structured snapshot of configured inputs and resolved geometry | scalar object |
| `signals` | Target, sky, thermal, dark, and total electron cubes | `(y, x, wavelength)` |
| `variances` | Target, sky, thermal, dark, read, and total variance cubes | `(y, x, wavelength)` |
| `apertures` | Registered aperture results | tuple |

When a PSF is configured, the result also contains:

| Attribute | Meaning | Shape |
| --- | --- | --- |
| `psf` | Applied PSF image and physical metadata | structured object |

`result.psf.data` is the centered, unit-normalized 2D image in `[y, x]`
order. `result.psf.pixel_scale` is its positive angular pixel scale, and
`result.psf.telescope_diameter` is the telescope diameter used by the PSF
model or instrument. `result.psf.wavelength` and `result.psf.pupil` are
populated for Hybrid-modelled PSFs; direct PSFs leave them as `None`. The pupil
is a dimensionless 2D quantity. The image and metadata are owned, read-only
snapshots; they do not depend on `include_models`.

A calculation containing only uniform targets may omit a PSF; in that case,
`result.psf` is absent. If a PSF is configured for a
uniform-only calculation, it is retained in the result even though spatial
convolution is unnecessary and is not applied.

The cube axis order is always `(y, x, wavelength)` in Python; see the
[coordinate and array conventions](../api.md#coordinate-and-array-conventions)
for how these indices relate to sky offsets. The optional
`models` group is absent when not requested; it is not present with a value of
`None`.

## Options

`result.options` records instrument, scale, detector spaxel scale, disperser,
atmosphere, PSF pixel scale and source path when applicable, and resolved
exposure and sky-subtraction configuration. Its geometry fields are:

| Field | Meaning |
| --- | --- |
| `position_angle` | Telescope pointing `+y` angle east of north |
| `pointing_center` | Supplied absolute telescope `SkyCoord`, or `None` |
| `ifu_position` | Supplied pointing offset or sky position and relative `rotation` |
| `ifu_center` | Resolved IFU center in ICRS, or `None` without an absolute pointing |
| `ifu_position_angle` | Detector `+y` angle: telescope angle plus IFU rotation |
| `targets` | Owned requests retaining each target's supplied position form and models |

Supplied `SkyCoord` frames are retained in `pointing_center`, `ifu_position`,
and `targets`; CubeSim uses ICRS to resolve their geometry.

`result.options.exposure` contains `time`, `n`, `n_target`, `n_sky`,
`target_time`, `sky_time`, and `total_time`. The sky-subtraction options contain
the method, optional nodding sequence, and optional read-only in-field mask.
For a Hybrid-backed run, `result.options.hybrid` records:

| Field | Meaning |
| --- | --- |
| `coordinate_form` | Supplied NGS form: `pointing_offsets` or `sky_positions` |
| `ngs_pointing_offsets` | Resolved `(x, y)` NGS offsets in telescope pointing axes, in angular units |
| `ngs_magnitudes` | Supplied NGS magnitudes, in `mag` |
| `ngs_magnitude_zeropoint` | Dataset zeropoint, in `photon / (m2 s)` |
| `ngs_flux` | Hybrid-resolved NGS rates, in `photon / s` |
| `science_position` | IFU center `(x, y)` in telescope pointing axes, in angular units |
| `wavelength` | PSF modelling wavelength, in microns |
| `zenith_angle` | PSF modelling zenith angle, in degrees |
| `mastsel_ini_file` | Resolved absolute MASTSEL configuration path |
| `science_ho_interpolator_file` | Resolved absolute science interpolator path |
| `ngs_ho_interpolator_file` | Resolved absolute NGS interpolator path |

The paths follow any symbolic links in the instrument data. For a run without
Hybrid modelling, `result.options.hybrid` is `None`. These values are an owned
snapshot, independent of later ETC changes. The full Hybrid diagnostics are
not retained; pickle persistence includes the options snapshot, while compact
FITS exports omit it.

## Models

With `include_models=True`, `result.models` contains:

- `targets`: one entry per configured target
- `combined`: the single detector-resolution target cube formed from all
  targets after spatial and spectral processing
- `transmission`: dimensionless atmospheric transmission on the detector
  wavelength grid
- `sky`: atmospheric sky spectral radiance on the detector wavelength grid
- `thermal`: instrument thermal spectral radiance on the detector wavelength
  grid

Each target has `high` and `low` model grids. A model grid contains
`wavelength`, `spatial`, `spectrum`, and optional `velocity` components.
Convolved high-resolution spatial and spectral components are also retained
when those operations apply.

`models.combined`, `models.transmission`, `models.sky`, and `models.thermal`
are the models that feed the detector calculation. The sky and thermal models
have units of spectral radiance. The signal products described below are
detected electrons after atmosphere, collecting area, throughput, QE, and
exposure accounting; they are not duplicate model products.

## Signals

`result.signals` always contains electron cubes:

| Attribute | Meaning |
| --- | --- |
| `target` | Detected target electrons |
| `sky` | Detected atmospheric sky electrons |
| `thermal` | Detected thermal electrons |
| `dark` | Detector dark-current electrons |
| `total` | `target + sky + thermal + dark` |

For in-field subtraction, the mask declares target-free spaxels. The target
signal is zero within that mask. Sky, thermal, and dark remain the detected
electron expectations that contribute shot noise, even though their expected
values cancel during sky subtraction.

## Variances

`result.variances` always contains electron-squared cubes named `target`,
`sky`, `thermal`, `dark`, `read`, and `total`.

## Noisy Samples

`result.sample(n=1, seed=None)` returns an immutable sampled-cube object without
changing the deterministic result. Its `data` contains noisy, sky-subtracted
detector electrons. One realization has shape `(y, x, wavelength)`; multiple
realizations have shape `(n, y, x, wavelength)`. Subtract
`result.signals.target` from `sample.data` to isolate the sampled noise.

Sampling uses Poisson target and background counts plus Gaussian read noise.
It applies the configured nodding or in-field subtraction operation to every
realization. Supply a non-negative integer `seed` for reproducible data. When
it is omitted, CubeSim generates one. The sample always exposes the seed used
through `sample.seed` and retains `wavelength` and `options` for interpretation
and persistence.

## Aperture Results

Each entry in `result.apertures` always contains:

- `name`: configured aperture name
- `mask`: read-only three-dimensional Boolean mask
- `snr`: scalar integrated signal to noise
- `signals`: integrated target, sky, thermal, dark, and total signals
- `variances`: integrated target, sky, thermal, dark, read, and total variances
- `spectra.snr`: S/N after spatially summing selected voxels at each wavelength
- `maps.snr`: S/N after spectrally summing selected voxels at each position

Aperture signals are direct reductions of the corresponding cube-level
components. `aperture.sample(n=1, seed=None)` draws integrated aperture data
without allocating full IFU cubes. The returned immutable object's `data` is a
scalar electron quantity for one realization and a one-dimensional quantity
for multiple realizations. Subtract `aperture.signals.target` from
`sample.data` to isolate sampled aperture noise. The sample retains its seed,
aperture name, three-dimensional mask, and wavelength coordinate.

For in-field subtraction, an aperture that overlaps the sky mask is rejected.
Sampling and variance reduction both propagate the shared IFU sky estimate;
the aperture does not perform a separate sky-subtraction operation.

Both `spectra.signals` and `maps.signals` contain the same target, sky, thermal,
dark, and total fields as the integrated aperture signal group. Spectra have
shape `(wavelength,)`, maps have shape `(y, x)`, and summing either projection
reproduces the corresponding integrated signal.

The equivalent projection groups contain target, sky, thermal, dark, read, and
total variances. Spectral variances are covariance-aware and sum to the
integrated aperture variances. Map variances are marginal per-spaxel
reductions. Under in-field subtraction, their sum does not generally reproduce
the integrated variance because a two-dimensional map cannot encode covariance
between spaxels that share a sky estimate. Under nodding, where selected voxels
are independent, it does.

Projection samples outside the three-dimensional aperture support contain
zero. Plotting functions mask those samples using the retained aperture mask.

## Saving

Use `result.save(path, overwrite=False)`. The suffix selects the format, and an
existing path raises `FileExistsError` unless overwrite is explicitly enabled.

### Pickle

`.pkl` stores the complete result object, including the structured options,
per-target model products, and aperture results. Load it with Python's `pickle`
module.

Pickle can execute code during loading. Load only trusted files, and treat the
format as coupled to compatible Python, dependency, and CubeSim versions.

### FITS

`.fits`, `.fit`, and `.fts` store a fixed portable science product containing
the wavelength coordinate and four detector cubes:

- `SNR`: signal to noise
- `SIGNAL`: expected sky-subtracted target electrons
- `BACKGROUND`: atmospheric sky, thermal, and dark electrons before subtraction
- `VARIANCE`: total variance after applying the configured sky subtraction

The background is `result.signals.sky + result.signals.thermal +
result.signals.dark`. The total detected signal remains derivable as `SIGNAL +
BACKGROUND` and is not stored separately.

FITS does not store models, component signals or variances, the PSF, registered
apertures, masks, or random samples. Use pickle when the complete result object
is required.

Image units are written in FITS metadata. Cube WCS uses celestial coordinates
when `set_pointing(sky_position=...)` supplied an absolute telescope center and
`XOFFSET` and `YOFFSET` otherwise. Those non-celestial world coordinates are
east and north angular offsets from the IFU center, not detector `(x, y)`
indices; the WCS rotates detector axes by the combined IFU position angle. For
an off-axis IFU, celestial WCS uses the IFU center, not the telescope center.

The primary header records the resolved calculation metadata:

| Keyword | Meaning |
| --- | --- |
| `INSTRUME` | Instrument name |
| `SCALE` | Selected scale |
| `DISPERSE` | Selected disperser |
| `ATMOS` | Selected atmosphere |
| `POSANGLE` | IFU position angle in degrees |
| `EXPTIME` | Single-frame time in seconds |
| `NEXP` | Total frame count |
| `NTARGET` | Target-frame count |
| `NSKY` | Sky-frame count |
| `TONTARG` | On-target integration in seconds |
| `TINT` | Total integration in seconds |
| `SKYMETH` | Sky-subtraction method |
| `SKYSEQ` | Nodding sequence, when configured |

The image extension layout is fixed:

| Extension | Presence | Contents |
| --- | --- | --- |
| `WAVELEN` | always | Detector wavelength coordinate |
| `SNR` | always | Signal-to-noise cube |
| `SIGNAL` | always | Expected sky-subtracted target-electron cube |
| `BACKGROUND` | always | Sky, thermal, and dark electron cube |
| `VARIANCE` | always | Total electron-squared variance cube |

Science images include `BUNIT`. FITS WCS axis 1 is wavelength, and axes 2 and 3
are the spatial coordinates.

### Sampled FITS

Cube and aperture sample objects save directly to FITS:

```python
cube_sample = result.sample(n=10, seed=42)
cube_sample.save("cube_samples.fits")

aperture_sample = result.apertures[0].sample(n=1000, seed=91)
aperture_sample.save("aperture_samples.fits")
```

A sampled-cube file contains `WAVELEN` and `DATA` extensions. `DATA` uses the
same cube WCS and calculation metadata as deterministic result FITS files. A
single realization is three-dimensional; multiple realizations use one
four-dimensional image whose leading Python axis is the realization axis.

A sampled-aperture file contains `WAVELEN`, `DATA`, and `MASK` extensions. The
primary header records the aperture name. A single integrated realization is
stored as a one-element `DATA` image, while multiple realizations form a
one-dimensional image. `MASK` stores the registered three-dimensional aperture
definition.

Both sample formats record `RNGSEED` and `NREAL` in the primary header. Sample
files accept `.fits`, `.fit`, or `.fts`; existing files require
`overwrite=True`.
