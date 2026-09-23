# Example Instrument

This directory contains a small, redistributable IFU definition for learning
and validating the public cubesim workflow. Run the notebooks with `example/`
as the working directory. From there, construct the ETC with the
`instrument_data` directory:

```python
import cubesim
etc = cubesim.Etc("instrument_data")
```

[`etc.ipynb`](etc.ipynb) demonstrates a complete calculation for an inclined
emission-line galaxy and the result-based plotting interface.
[`sampling.ipynb`](sampling.ipynb) saves a sampled IFU cube, draws integrated
Monte Carlo aperture measurements, and compares their empirical variance with
the predicted detector variance.
Both notebooks use the shared configuration in
[`inclined_galaxy.py`](inclined_galaxy.py).

The configuration is inspired by the final GIRMOS design described by
[Sivanandam et al. (2024)](https://doi.org/10.1117/12.3020305), but it is not a
GIRMOS performance model.

## Selectable Options

Three spatial scales are available:

| Selection | Spaxel scale | Field of view |
| --- | --- | --- |
| `25mas` | 25 mas | 1 x 1 arcsec |
| `50mas` | 50 mas | 2 x 2 arcsec |
| `100mas` | 100 mas | 4 x 4 arcsec |

Six dispersers are available. Wavelength ranges are vacuum wavelengths, and
each resolution element is sampled by two detector pixels.

| Selection | Resolving power | Wavelength range |
| --- | ---: | --- |
| `r3000.yj` | 3000 | 0.95-1.35 microns |
| `r3000.jh` | 3000 | 1.25-1.80 microns |
| `r3000.hk` | 3000 | 1.63-2.35 microns |
| `r8000.j` | 8000 | 1.19-1.35 microns |
| `r8000.h` | 8000 | 1.50-1.71 microns |
| `r8000.k` | 8000 | 2.11-2.38 microns |

The `airmass10` and `airmass20` atmosphere selections both use 1 mm of
precipitable water vapour, at airmasses 1.0 and 2.0 respectively.

The included `psf.fits` provides a smooth synthetic adaptive-optics PSF for
example calculations.

For example:

```python
etc.configure(
    scale="50mas",
    disperser="r3000.yj",
    atmosphere="airmass10",
)
etc.set_psf("psf.fits")
```

## Atmosphere Data

The bundled transmission and background tables are adapted from Gemini North
Maunakea atmosphere files for 1 mm precipitable water vapour at airmasses 1.0
and 2.0. The source files are published on the
[Gemini Observatory sites page](https://www.gemini.edu/observing/telescopes-and-sites/sites#IRSky):

- `mktrans_zm_10_10.dat` and `mk_skybg_zm_10_10_ph.dat`
- `mktrans_zm_10_20.dat` and `mk_skybg_zm_10_20_ph.dat`

The underlying ATRAN model is described by
[Lord (1992), *A New Software Tool for Computing Earth's Atmospheric
Transmission of Near- and Far-Infrared Radiation*, NASA Technical Memorandum
103957](https://ntrs.nasa.gov/citations/19930010877).

The bundled adaptations are intended only for software examples and
validation, not as a calibrated instrument model or a substitute for the
current Gemini ITC.
