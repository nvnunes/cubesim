# cubesim

`cubesim` is an exposure time calculator framework for integral-field
spectrographs.

The current implementation loads and validates an instrument-data directory,
configures targets, exposure accounting, and a direct PSF, then calculates
signal-to-noise datacubes for an integral-field spectrograph.

Its intended calculation scope is:

- instrument configuration and derived ETC state
- target and source model construction
- PSF input ingestion and application
- background, throughput, detector, signal, noise, and SNR computation

An instrument integration consists only of an `etc.ini` file and the data
files it references. Real instrument datasets are distributed separately from
the Python package.

`cubesim` does not own PSF simulation. Direct PSFs may be supplied as FITS,
NPY, or in-memory two-dimensional arrays. FITS files carry `PIXSCALE` in mas
per pixel; NPY and in-memory inputs require an explicit angular pixel scale.

## Installation

To install the package from a local checkout:

```bash
python -m pip install .
```

That path is intended for package use. For local development in this repo, use
the canonical workflow in `Local Development Setup` below.

CubeSim does not include instrument definitions or scientific data. A
calculation requires a separately supplied instrument-data directory containing
`etc.ini` and its referenced files.

## Quickstart: Python API

```python
import astropy.units as u
import cubesim

etc = cubesim.Etc("/path/to/instrument-data")
etc.configure(
    scale="50mas",
    disperser="r3000.yj",
    atmosphere="airmass10_pwv10",
)
etc.set_psf("psf.npy", pixel_scale=10 * u.mas)
etc.add_target(
    position=(0 * u.arcsec, 0 * u.arcsec),
    spatial=cubesim.Point(),
    spectrum=cubesim.GaussianLines(
        wavelength=1.1 * u.micron,
        flux=1e-17 * u.erg / (u.s * u.cm**2),
        dispersion=40 * u.km / u.s,
    ),
)
etc.set_exposure(time=600 * u.s, n_target=10)

result = etc.run()
print(result.snr.shape)
```

The instrument-data directory must contain `etc.ini` at its root. Paths in
that file are relative to the same directory; cubesim does not search the
current directory, package data, environment variables, or the network.
Direct PSF paths may be absolute or relative to the instrument-data directory.

`result.snr`, `result.wavelength`, and `result.options` are always present.
Pass `include_models`, `include_signals`, `include_variances`, or
`include_data` to `run()` for the corresponding optional immutable groups.
The model group includes the target models and the detector-grid atmospheric
transmission, sky-radiance, and thermal-radiance models.
Call `result.save("result.pkl")` for a complete trusted Python round trip or
`result.save("result.fits")` for portable datacubes, metadata, masks, aperture
measurements, units, and WCS.

The current implementation covers analytic and supplied spatial
profiles, Gaussian-line and tabulated spectra, constant velocity, rotating
disks, supplied velocity fields, nodding and in-field sky subtraction, and
achromatic direct PSFs. Spatially varying velocities are applied on the
high-resolution model before PSF and line-spread-function convolution.

Python API guide: [`docs/api.md`](docs/api.md)

## Documentation

- Documentation home: [`docs/index.md`](docs/index.md)
- Python API guide: [`docs/api.md`](docs/api.md)
- Instrument-data contract: [`docs/instrument-data.md`](docs/instrument-data.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Testing: [`docs/testing.md`](docs/testing.md)
- Development setup: [`docs/development.md`](docs/development.md)

## Local Development Setup

The canonical local development workflow uses the repo-local `./.conda`
environment.

For a fresh clone, create or activate a Python 3.12+ environment at `./.conda`
with your preferred environment manager, then install the package in editable
mode with the `dev` and `docs` extras:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

After the editable install is current, run the test suite and strict docs build
from the same environment:

```bash
./.conda/bin/python -m pytest -q
./.conda/bin/mkdocs build --strict
```

Pre-commit checks are versioned in `.githooks/pre-commit`. Activate them once
per clone with:

```bash
git config core.hooksPath .githooks
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
