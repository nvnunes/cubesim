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
| `options` | Structured snapshot of resolved inputs | scalar object |
| `apertures` | Registered aperture results | tuple |

The cube axis order is always `(y, x, wavelength)` in Python. Optional result
groups are absent when not requested; they are not present with a value of
`None`.

## Options

`result.options` records:

- instrument, scale, detector spaxel scale, disperser, and atmosphere names
- pointing position angle and optional absolute center
- an owned snapshot of every target
- PSF pixel scale and source path when applicable
- resolved exposure and sky-subtraction configuration
- requested number of noisy cubes

`result.options.exposure` contains `time`, `n`, `n_target`, `n_sky`,
`target_time`, `sky_time`, and `total_time`. The sky-subtraction options contain
the method, optional nodding sequence, and optional read-only in-field mask.

## Models

With `include_models=True`, `result.models` contains:

- `targets`: one entry per configured target
- `combined`: the single detector-resolution target cube formed from all
  targets after spatial and spectral processing

Each target has `high` and `low` model grids. A model grid contains
`wavelength`, `spatial`, `spectrum`, and optional `velocity` components.
Convolved high-resolution spatial and spectral components are also retained
when those operations apply.

`models.combined` is the model that feeds the detector calculation. The signal
products described below are detected electrons after atmosphere, collecting
area, throughput, QE, and exposure accounting; they are not duplicate model
products.

## Signals

With `include_signals=True`, `result.signals` contains electron cubes:

| Attribute | Meaning |
| --- | --- |
| `target` | Detected target electrons |
| `sky` | Detected atmospheric sky electrons |
| `thermal` | Detected thermal electrons |
| `dark` | Detector dark and light-leak electrons |
| `background` | `sky + thermal + dark` |
| `total` | `target + background` |

For in-field subtraction, these cube-level products remain the raw detected
components used to construct Poisson noise. They do not replace the target with
its sky-subtracted expectation.

## Variances

With `include_variances=True`, `result.variances` contains electron-squared
cubes named `target`, `sky`, `thermal`, `dark`, `read`, and `total`.

## Noisy Data

With `include_data=True`, `result.data` contains noisy, sky-subtracted detector
data in electrons. For `n_cubes=1`, its shape is `(y, x, wavelength)`. For
multiple realizations, its shape is `(n_cubes, y, x, wavelength)`.

## Aperture Results

Each entry in `result.apertures` always contains:

- `name`: configured aperture name
- `mask`: read-only three-dimensional Boolean mask
- `snr`: scalar integrated signal to noise

Requested signal, variance, and noisy-data summaries appear under `signals`,
`variances`, and `data`, matching the groups requested from `run()`. Aperture
data has one value per noisy realization. For in-field subtraction, aperture
signals describe the sky-subtracted reduction.

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

`.fits`, `.fit`, and `.fts` store portable science datacubes and interpretive
metadata. Every FITS result contains wavelength and S/N image extensions.
Requested combined-model, signal, variance, and noisy-data cubes are included
when present. Sky masks and aperture products are also included when
configured.

Image units are written in FITS metadata. Cube WCS uses celestial coordinates
when `set_pointing(center=...)` supplied an absolute center and angular offsets
otherwise. Multiple noisy realizations add a realization axis.

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
| `NCUBES` | Number of requested noisy cubes |
| `SKYSEQ` | Nodding sequence, when configured |
| `PSFSCALE` | PSF pixel scale in milliarcseconds, when configured |
| `PSFFILE` | PSF source path, when file-backed |

The image and table extension layout is:

| Extension | Presence | Contents |
| --- | --- | --- |
| `WAVELEN` | always | Detector wavelength coordinate |
| `SNR` | always | Signal-to-noise cube |
| `MODEL` | `include_models` | Combined detector-resolution target model |
| `SIGTARG` | `include_signals` | Target signal |
| `SIGSKY` | `include_signals` | Atmospheric sky signal |
| `SIGTHERM` | `include_signals` | Thermal signal |
| `SIGDARK` | `include_signals` | Dark and light-leak signal |
| `SIGBKG` | `include_signals` | Combined background signal |
| `SIGTOTAL` | `include_signals` | Total detected signal |
| `VARTARG` | `include_variances` | Target variance |
| `VARSKY` | `include_variances` | Atmospheric sky variance |
| `VARTHERM` | `include_variances` | Thermal variance |
| `VARDARK` | `include_variances` | Dark and light-leak variance |
| `VARREAD` | `include_variances` | Read variance |
| `VARTOTAL` | `include_variances` | Total variance |
| `DATA` | `include_data` | Noisy sky-subtracted detector data |
| `SKYMASK` | in-field subtraction | Two-dimensional sky mask |
| `APMASK<n>` | each aperture | Three-dimensional aperture mask |
| `APERTURE` | any aperture | Aperture names, S/N, and requested summaries |

Science images include `BUNIT`. FITS WCS axis 1 is wavelength, axes 2 and 3
are the spatial coordinates, and axis 4 is realization number when `DATA`
contains multiple cubes. The `APERTURE` table always contains `NAME` and `SNR`;
requested signal columns use a `SIG` prefix, requested variance columns use a
`VAR` prefix, and requested aperture realizations use `DATA`.

FITS intentionally does not reconstruct the complete Python object graph or
every per-target high- and low-resolution intermediate. Use pickle when exact
object reconstruction is required.
