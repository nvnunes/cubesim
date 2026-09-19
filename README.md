# cubesim

`cubesim` is an exposure time calculator framework for integral-field
spectrographs.

The current implementation loads and validates an instrument-data directory,
selects its scale, disperser, and atmosphere, and accepts a direct PSF. The
forward ETC calculation is not implemented yet.

Its intended calculation scope is:

- instrument configuration and derived ETC state
- target and source model construction
- PSF input ingestion and application
- background, throughput, detector, signal, noise, and SNR computation

An instrument integration consists only of an `etc.ini` file and the data
files it references. Configuration, ECSV table loading, and direct PSF input
are the portions implemented currently. Real instrument datasets
are distributed separately from the Python package.

`cubesim` does not own PSF simulation. Direct PSFs may be supplied as FITS,
NPY, or in-memory two-dimensional arrays. FITS files carry `PIXSCALE` in mas
per pixel; NPY and in-memory inputs require an explicit angular pixel scale.

## Current API

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
```

The instrument-data directory must contain `etc.ini` at its root. Paths in
that file are relative to the same directory; cubesim does not search the
current directory, package data, environment variables, or the network.
Direct PSF paths may be absolute or relative to the instrument-data directory.

## Project Docs

- [`docs/architecture.md`](docs/architecture.md)
  - package shape, public API boundaries, data and ETC-input ownership, and
    compute lifecycle
- [`docs/instrument-data.md`](docs/instrument-data.md)
  - instrument directory, INI schema, path resolution, and ECSV table contracts
- [`docs/testing.md`](docs/testing.md)
  - verification commands and completion expectations
- [`docs/development.md`](docs/development.md)
  - local environment setup and daily commands

## Local Development Setup

The canonical local development workflow uses the repo-local `./.conda`
environment.

For a fresh clone, create or activate a Python 3.12+ environment at `./.conda`
with your preferred environment manager, then install the package in editable
mode with the `dev` extra:

```bash
./.conda/bin/python -m pip install -e ".[dev]"
```

After the editable install is current, run the test suite from the same
environment:

```bash
./.conda/bin/python -m pytest -q
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
