# cubesim

`cubesim` is an exposure time calculator framework for integral-field
spectrographs.

The current implementation loads and validates an instrument-data directory,
configures targets, exposure accounting, and a direct or optional Hybrid AO PSF, then calculates
signal-to-noise datacubes for an integral-field spectrograph.

Its intended calculation scope is:

- instrument configuration and derived ETC state
- target and source model construction
- PSF input ingestion and application
- background, throughput, detector, signal, noise, and SNR computation
- result-based scientific validation plots

An instrument integration consists only of an `etc.ini` file and the data
files it references. The repository includes a redistributable example
instrument; real instrument datasets are distributed separately from the
Python package.

`cubesim` delegates optional AO PSF modelling to `hybrid-ao-psf`; it does not
implement that simulation. Direct PSFs may be supplied as FITS,
NPY, or in-memory two-dimensional arrays. FITS files carry `PIXSCALE` in mas
per pixel; NPY and in-memory inputs require an explicit angular pixel scale.

## Installation

To install the package from a local checkout:

```bash
python -m pip install .
```

For instrument datasets with Hybrid AO PSF assets, install the optional
dependency with `python -m pip install '.[hybrid]'` from this checkout.

That path is intended for package use. For local development in this repo, use
the canonical workflow in `Local Development Setup` below.

The bundled [`example/instrument_data`](example/instrument_data) directory can
be used immediately. A real calculation requires a separately supplied
instrument-data directory containing `etc.ini` and its referenced files.

## Quickstart: Python API

```python
import astropy.units as u
import cubesim

etc = cubesim.Etc("example/instrument_data")
etc.configure(
    scale="50mas",
    disperser="r3000.yj",
    atmosphere="airmass10",
)
etc.set_psf("psf.fits")
etc.add_target(
    ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
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
that file are relative to the same directory. A relative instrument-data path
may resolve from the working directory or the nearest `pyproject.toml` project
root; referenced files receive no additional search fallback.
Direct PSF paths may be absolute or relative to the instrument-data directory.

`result.snr`, `result.wavelength`, `result.options`, `result.signals`, and
`result.variances` are always present. Pass `include_models=True` to `run()`
to retain target models and the detector-grid atmospheric transmission,
sky-radiance, and thermal-radiance models. Draw noisy, sky-subtracted detector
cubes with `sampled_cube = result.sample(seed=42)`, inspect them through
`sampled_cube.data`, and save them with `sampled_cube.save("sampled_cube.fits")`.
Call `result.save("result.pkl")` for a complete trusted Python round trip or
`result.save("result.fits")` for portable S/N, target-signal, background, and
total-variance cubes with wavelength, units, metadata, and WCS.

Plot immutable results through the dedicated submodule. Functions return
Matplotlib figures without displaying or saving them:

```python
import cubesim.plotting as plotting

figure = plotting.plot_snr(result, wavelength=1.1 * u.micron)
```

The current implementation covers analytic and supplied spatial
profiles, Gaussian-line and tabulated spectra, constant velocity, rotating
disks, supplied velocity fields, nodding and in-field sky subtraction, and
achromatic direct or Hybrid-modelled PSFs. Spatially varying velocities are applied on the
high-resolution model before PSF and line-spread-function convolution.

Python API guide: [`docs/api.md`](docs/api.md)

## Example Instrument

[`example/README.md`](example/README.md) describes the bundled, redistributable
example instrument and the provenance of its Gemini North Maunakea atmosphere
tables. Its rounded instrument properties and synthetic PSF are intended to
demonstrate and validate cubesim, not to predict GIRMOS or another real
instrument.

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
