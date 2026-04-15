# cubesim

`cubesim` is a simple exposure time calculator framework for quick setup and
use.

It is intended for the stage before a full ETC is built: enough structure to
configure an instrument, ingest PSF artifacts, and compute useful signal and
noise estimates without building a complete end-user ETC.

It provides a focused core for:
- instrument configuration and derived ETC state
- target and source model construction
- PSF artifact ingestion and application
- background, throughput, detector, signal, noise, and SNR computation

`cubesim` is intentionally not a full-featured ETC framework. It is designed
to stay lightweight, direct, and easy to wire into early instrument and
simulation workflows.

`cubesim` does not own PSF simulation. It consumes persisted PSF artifacts,
with pickle as the initial supported format.

## Project Docs

- `docs/architecture.md`
  - package shape, public API boundaries, artifact and ETC-input contracts, and
    compute lifecycle
- `docs/testing.md`
  - verification commands and completion expectations
- `docs/development.md`
  - local environment setup and daily commands
- `docs/plan.md`
  - near-term phased work and deferred planning context

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
