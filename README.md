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

Planned early work:
- define the initial public API
- add config-driven instrument definitions
- implement the minimal ETC compute path
- establish artifact handling for atmospheric and instrument-specific inputs

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
