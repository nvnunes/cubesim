"""Instrument configuration and table-loading contract tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

import cubesim


def test_constructor_resolves_relative_path_from_project_root(
    instrument_data,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    nested = project / "notebooks"
    destination = project / "example" / "instrument_data"
    nested.mkdir(parents=True)
    destination.mkdir(parents=True)
    for source in instrument_data.iterdir():
        if source.is_file():
            shutil.copy2(source, destination)
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    monkeypatch.chdir(nested)

    etc = cubesim.Etc("example/instrument_data")

    assert etc._instrument.root == destination.resolve()


def test_configure_resolves_options_and_ordered_optical_path(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )

    assert etc._selection is not None
    assert etc._selection.scale.name == "50mas"
    assert etc._selection.disperser.name == "r3000.yj"
    assert etc._selection.disperser.resolving_power == 3000
    assert [item.name for item in etc._selection.optical_components] == [
        "telescope",
        "spectrograph",
    ]


def test_configure_rejects_unknown_exact_option(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(
        ValueError,
        match=r"Unknown scale '50MAS'; available options: \['50mas'\]",
    ):
        etc.configure(
            scale="50MAS",
            disperser="r3000.yj",
            atmosphere="airmass10_pwv10",
        )


def test_configure_accepts_hyphenated_option_name(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[scale.50mas]", "[scale.girmos-kmos]"
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    etc.configure(
        scale="girmos-kmos",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )

    assert etc._selection is not None
    assert etc._selection.scale.name == "girmos-kmos"


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


def test_constructor_follows_relative_symlink_outside_data_directory(
    instrument_data, tmp_path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-qe.ecsv"
    shutil.copy2(instrument_data / "qe.ecsv", outside)
    (instrument_data / "qe-link.ecsv").symlink_to(Path("..") / outside.name)
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "quantum_efficiency_file = qe.ecsv",
            "quantum_efficiency_file = qe-link.ecsv",
        ),
        encoding="utf-8",
    )

    cubesim.Etc(instrument_data)


def test_constructor_rejects_absolute_ini_reference(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "quantum_efficiency_file = qe.ecsv",
            f"quantum_efficiency_file = {instrument_data / 'qe.ecsv'}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="INI file references must be relative"):
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

    with pytest.raises(ValueError, match="does not cover disperser"):
        etc.configure(
            scale="50mas",
            disperser="r3000.yj",
            atmosphere="airmass10_pwv10",
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
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )

    assert etc._selection is not None


def test_configure_rejects_duplicate_component_order(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[optics.spectrograph.r3000]\norder = 2",
            "[optics.spectrograph.r3000]\norder = 1",
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match="duplicate order values"):
        etc.configure(
            scale="50mas",
            disperser="r3000.yj",
            atmosphere="airmass10_pwv10",
        )


def test_disperser_group_is_not_selectable(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(
        ValueError,
        match=r"Unknown disperser 'r3000'; available options: \['r3000.yj'\]",
    ):
        etc.configure(
            scale="50mas",
            disperser="r3000",
            atmosphere="airmass10_pwv10",
        )


def test_disperser_leaf_can_override_parent(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[disperser.r3000.yj]\n",
            "[disperser.r3000.yj]\nresolving_power = 3500\n",
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )

    assert etc._selection is not None
    assert etc._selection.disperser.resolving_power == 3500


def test_constructor_requires_explicit_disperser_parent(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[disperser.r3000]", "[disperser.other]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires parent section"):
        cubesim.Etc(instrument_data)


def test_constructor_rejects_incomplete_disperser_leaf(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "pixels_per_resolution_element = 2\n", ""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required keys after inheritance"):
        cubesim.Etc(instrument_data)


def test_constructor_rejects_unknown_optical_scope(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[optics.spectrograph.r3000]",
            "[optics.spectrograph.unknown]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown disperser scope 'unknown'"):
        cubesim.Etc(instrument_data)


def test_flat_disperser_remains_supported(instrument_data) -> None:
    config = instrument_data / "etc.ini"
    config.write_text(
        config.read_text(encoding="utf-8")
        .replace(
            """\
[disperser.r3000]
resolving_power = 3000
pixels_per_resolution_element = 2

[disperser.r3000.yj]
wavelength_min = 0.95
wavelength_max = 1.35
""",
            """\
[disperser.prism]
resolving_power = 3000
pixels_per_resolution_element = 2
wavelength_min = 0.95
wavelength_max = 1.35
""",
        )
        .replace(
            "[optics.spectrograph.r3000]",
            "[optics.spectrograph.prism]",
        ),
        encoding="utf-8",
    )
    etc = cubesim.Etc(instrument_data)

    etc.configure(
        scale="50mas",
        disperser="prism",
        atmosphere="airmass10_pwv10",
    )

    assert etc._selection is not None
    assert etc._selection.disperser.name == "prism"


def test_constructor_requires_etc_ini(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="has no etc.ini"):
        cubesim.Etc(tmp_path)
