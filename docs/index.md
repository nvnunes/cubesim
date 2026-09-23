# cubesim Docs

CubeSim is an exposure time calculator framework for integral-field
spectrographs. It provides:

- explicit instrument selection from an external instrument-data directory
- composable spatial, spectral, and velocity target models
- direct achromatic PSF input or optional Hybrid AO PSF modelling
- nodding and in-field sky subtraction
- signal-to-noise datacubes with detector signal and variance components
- reproducible sampled cubes and integrated aperture measurements
- three-dimensional aperture measurements with spectrum and map diagnostics
- result-based plotting for model, signal, S/N, PSF, and aperture validation
- FITS and pickle result persistence, plus FITS sample persistence

CubeSim supplies the calculation framework and data formats. The repository
includes one redistributable example at `example/instrument_data`; real
instrument definitions and scientific assets are distributed separately.

## Start Here

- Guides:
  - [Python API](api.md)
  - [Instrument Data](instrument-data.md)
  - [Architecture](architecture.md)
- API reference:
  - [ETC](reference/etc.md)
  - [Target Models](reference/models.md)
  - [Results And Persistence](reference/results.md)
  - [Plotting](reference/plotting.md)
- Contributor documentation:
  - [Testing and verification](testing.md)
  - [Development setup](development.md)
