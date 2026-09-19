"""Instrument configuration and table-loading contract tests."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

import cubesim


def test_configure_resolves_modes_and_ordered_optical_path(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    etc.configure(
        spatial_mode="50mas",
        spectral_mode="r3000_yj",
        atmosphere_mode="pwv10_airmass10",
    )

    assert etc._selection is not None
    assert etc._selection.spatial_mode.name == "50mas"
    assert [item.name for item in etc._selection.optical_components] == [
        "telescope",
        "spectrograph",
    ]


def test_configure_rejects_unknown_exact_mode(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(
        ValueError,
        match=r"Unknown spatial mode '50MAS'; available modes: \['50mas'\]",
    ):
        etc.configure(
            spatial_mode="50MAS",
            spectral_mode="r3000_yj",
            atmosphere_mode="pwv10_airmass10",
        )


def test_constructor_rejects_unknown_ini_key(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "f_number = 16.0", "f_number = 16.0\nfield_of_view = 2.0"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys"):
        cubesim.Etc(instrument_data)


def test_constructor_rejects_reference_outside_bundle(instrument_data, tmp_path) -> None:
    outside = tmp_path.parent / "outside.ecsv"
    outside.write_text("not used", encoding="utf-8")
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "quantum_efficiency_file = qe.ecsv",
            "quantum_efficiency_file = ../outside.ecsv",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the instrument bundle"):
        cubesim.Etc(instrument_data)


def test_configure_rejects_incomplete_table_coverage(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "wavelength_max = 1.35", "wavelength_max = 1.45"
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match="does not cover spectral mode"):
        etc.configure(
            spatial_mode="50mas",
            spectral_mode="r3000_yj",
            atmosphere_mode="pwv10_airmass10",
        )


def test_constructor_rejects_nonuniform_atmosphere_grid(instrument_data) -> None:
    path = instrument_data / "transmission.ecsv"
    table = QTable.read(path, format="ascii.ecsv")
    table["wavelength"] = np.array([0.9, 1.0, 1.1, 1.21, 1.3, 1.4]) * u.micron
    table.write(path, format="ascii.ecsv", overwrite=True)

    with pytest.raises(ValueError, match="uniformly sampled"):
        cubesim.Etc(instrument_data)


def test_constructor_rejects_incompatible_table_unit(instrument_data) -> None:
    path = instrument_data / "background.ecsv"
    table = QTable.read(path, format="ascii.ecsv")
    table["background"] = np.ones(6) * u.W
    table.write(path, format="ascii.ecsv", overwrite=True)

    with pytest.raises(ValueError, match="units must be convertible"):
        cubesim.Etc(instrument_data)


def test_constant_detector_qe_needs_no_qe_table_coverage(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "quantum_efficiency_file = qe.ecsv", "quantum_efficiency = 0.9"
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    etc.configure(
        spatial_mode="50mas",
        spectral_mode="r3000_yj",
        atmosphere_mode="pwv10_airmass10",
    )

    assert etc._selection is not None


def test_configure_rejects_duplicate_component_order(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[optical_component.r3000_yj.spectrograph]\norder = 2",
            "[optical_component.r3000_yj.spectrograph]\norder = 1",
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match="duplicate order values"):
        etc.configure(
            spatial_mode="50mas",
            spectral_mode="r3000_yj",
            atmosphere_mode="pwv10_airmass10",
        )


def test_constructor_requires_etc_ini(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="has no etc.ini"):
        cubesim.Etc(tmp_path)
