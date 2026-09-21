"""Shared inclined-galaxy configuration for the example notebooks."""

from __future__ import annotations

import astropy.units as u

import cubesim


def configure(etc: cubesim.Etc) -> None:
    """Configure an ETC for the shared inclined emission-line galaxy example."""

    etc.configure(
        scale="50mas",
        disperser="r3000.hk",
        atmosphere="airmass10",
    )
    etc.set_psf("psf.fits")
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Sersic(
            effective_radius=0.15 * u.arcsec,
            index=1.0,
            axis_ratio=0.5,
            position_angle=35 * u.deg,
        ),
        spectrum=cubesim.GaussianLines(
            wavelength=2.2 * u.micron,
            flux=5e-16 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
        velocity=cubesim.RotatingDisk(
            maximum_velocity=200 * u.km / u.s,
            turnover_radius=0.1 * u.arcsec,
            inclination=60 * u.deg,
            position_angle=35 * u.deg,
        ),
    )
    etc.add_aperture(
        name="core",
        size=(4, 4, 13),
        center=(19, 19, 2.2 * u.micron),
    )
    etc.add_aperture(
        name="redshifted",
        size=(4, 4, 13),
        center=(23, 22, 2.2 * u.micron),
    )
    etc.add_aperture(
        name="blueshifted",
        size=(4, 4, 13),
        center=(15, 16, 2.2 * u.micron),
    )
    etc.set_exposure(time=600 * u.s, n_target=4)
