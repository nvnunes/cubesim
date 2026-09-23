"""Measure the PSF retained by a CubeSim calculation."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u

from cubesim._result import PsfResult

__all__ = ["PsfStats", "psf_stats"]


@dataclass(frozen=True, slots=True)
class PsfStats:
    """Measurements of one retained PSF.

    Attributes:
        sr: Dimensionless Strehl ratio, or ``None`` without wavelength and pupil.
        fwhm: Contour-derived geometric-mean FWHM in angular units.
        ee_apertures: Full side lengths of the square EE apertures.
        ee: Ensquared-energy fractions at ``ee_apertures``.
    """

    sr: u.Quantity | None
    fwhm: u.Quantity
    ee_apertures: u.Quantity
    ee: u.Quantity

    def __post_init__(self) -> None:
        for name in ("sr", "fwhm", "ee_apertures", "ee"):
            value = getattr(self, name)
            if value is not None:
                owned = u.Quantity(value, copy=True)
                owned.setflags(write=False)
                object.__setattr__(self, name, owned)


def psf_stats(psf: PsfResult, *, ee_apertures: u.Quantity) -> PsfStats:
    """Measure Strehl when possible, plus FWHM and ensquared energy.

    ``psf`` is the structured ``result.psf`` from an ETC calculation.
    ``ee_apertures`` is a nonempty one-dimensional angular Quantity of full
    square-aperture widths. The optional ``cubesim[stats]`` extra is required.
    Strehl is unavailable for a direct PSF without modelling wavelength and
    telescope pupil metadata.
    """

    if not isinstance(psf, PsfResult):
        raise TypeError("psf must be the structured result.psf from Etc.run().")
    if not isinstance(ee_apertures, u.Quantity):
        raise TypeError("ee_apertures must be an angular Quantity.")
    if ee_apertures.ndim != 1:
        raise ValueError("ee_apertures must be one-dimensional.")

    try:
        from ao_stats import PsfMetadata, compute_psf_stats
    except ImportError as exc:
        if exc.name == "ao_stats":
            raise ImportError(
                "PSF diagnostics require the optional 'cubesim[stats]' extra. "
                "Install it from the CubeSim checkout with pip install '.[stats]'."
            ) from exc
        raise

    apertures = ee_apertures.to(u.mas)
    if psf.wavelength is not None and psf.pupil is not None:
        sr, ee, fwhm = compute_psf_stats(
            psf.data,
            PsfMetadata(
                wavelength=psf.wavelength,
                pixel_scale=psf.pixel_scale,
                tel_diameter=psf.telescope_diameter,
                tel_pupil=psf.pupil,
            ),
            ee_apertures=apertures,
            ee_geometry="ensquared",
        )
    else:
        ee, fwhm = compute_psf_stats(
            psf.data,
            pixel_scale=psf.pixel_scale,
            ee_apertures=apertures,
            ee_geometry="ensquared",
            metrics=("ee", "fwhm"),
        )
        sr = None

    return PsfStats(sr=sr, fwhm=fwhm, ee_apertures=apertures, ee=ee)
