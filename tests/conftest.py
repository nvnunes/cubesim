"""Shared instrument-bundle fixtures."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable


@pytest.fixture
def instrument_data(tmp_path: Path) -> Path:
    wavelength = np.linspace(0.9, 1.4, 6) * u.micron
    QTable(
        [wavelength, np.linspace(0.7, 0.8, 6) * u.dimensionless_unscaled],
        names=("wavelength", "quantum_efficiency"),
    ).write(tmp_path / "qe.ecsv", format="ascii.ecsv")
    QTable(
        [wavelength, np.linspace(0.8, 0.9, 6) * u.dimensionless_unscaled],
        names=("wavelength", "transmission"),
    ).write(tmp_path / "transmission.ecsv", format="ascii.ecsv")
    QTable(
        [
            wavelength,
            np.ones(6) * u.photon / (u.s * u.m**2 * u.arcsec**2 * u.micron),
        ],
        names=("wavelength", "background"),
    ).write(tmp_path / "background.ecsv", format="ascii.ecsv")

    (tmp_path / "etc.ini").write_text(
        """\
[system]
name = Test IFU

[telescope]
primary_diameter = 8.0
central_obscuration = 1.0
f_number = 16.0

[detector]
read_noise = 5.0
dark_current = 0.05
light_leak = 0.01
quantum_efficiency_file = qe.ecsv

[scale.50mas]
spaxels_x = 4
spaxels_y = 3
spaxel_scale = 50

[disperser.r3000]
resolving_power = 3000
pixels_per_resolution_element = 2

[disperser.r3000.yj]
wavelength_min = 0.95
wavelength_max = 1.35

[atmosphere.airmass10_pwv10]
pwv = 1.0
airmass = 1.0
transmission_file = transmission.ecsv
background_file = background.ecsv

[optics.telescope]
order = 1
throughput = 0.9
emissivity = 0.1
temperature = 275

[optics.spectrograph.r3000]
order = 2
throughput = 0.7
emissivity = 0.0
""",
        encoding="utf-8",
    )
    return tmp_path
