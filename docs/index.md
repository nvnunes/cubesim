# cubesim Docs

CubeSim is an exposure time calculator framework for integral-field
spectrographs. It provides:

- explicit instrument selection from an external instrument-data directory
- composable spatial, spectral, and velocity target models
- direct achromatic PSF input
- nodding and in-field sky subtraction
- signal-to-noise datacubes with optional model, signal, variance, and noisy
  data products
- three-dimensional aperture measurements
- FITS and pickle result persistence

CubeSim supplies the calculation framework and data formats. Instrument
definitions and scientific assets are distributed separately.

## Start Here

- Guides:
  - [Python API](api.md)
  - [Instrument Data](instrument-data.md)
  - [Architecture](architecture.md)
- API reference:
  - [ETC](reference/etc.md)
  - [Target Models](reference/models.md)
  - [Results And Persistence](reference/results.md)
- Contributor documentation:
  - [Testing and verification](testing.md)
  - [Development setup](development.md)
