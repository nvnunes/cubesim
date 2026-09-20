"""Instrument-definition and canonical table loading."""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import astropy.units as u
import numpy as np
from astropy.table import QTable

_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")
_UNIFORM_RTOL = 1e-7
_DISPERSER_KEYS = {
    "resolving_power",
    "pixels_per_resolution_element",
    "wavelength_min",
    "wavelength_max",
}
_Option = TypeVar("_Option")


@dataclass(frozen=True, slots=True)
class SpectralTable:
    """One validated two-column wavelength table in canonical units."""

    path: Path
    wavelength: u.Quantity
    values: u.Quantity


@dataclass(frozen=True, slots=True)
class TelescopeDefinition:
    primary_diameter: u.Quantity
    central_obscuration: u.Quantity
    f_number: float


@dataclass(frozen=True, slots=True)
class DetectorDefinition:
    read_noise: u.Quantity
    dark_current: u.Quantity
    quantum_efficiency: float | None
    quantum_efficiency_table: SpectralTable | None


@dataclass(frozen=True, slots=True)
class Scale:
    name: str
    spaxels_x: int
    spaxels_y: int
    spaxel_scale: u.Quantity


@dataclass(frozen=True, slots=True)
class Disperser:
    name: str
    resolving_power: float
    pixels_per_resolution_element: float
    wavelength_min: u.Quantity
    wavelength_max: u.Quantity


@dataclass(frozen=True, slots=True)
class Atmosphere:
    name: str
    pwv: u.Quantity
    airmass: float
    transmission: SpectralTable
    background: SpectralTable


@dataclass(frozen=True, slots=True)
class OpticalComponent:
    name: str
    disperser_scope: str | None
    order: int
    throughput: float
    emissivity: float
    temperature: u.Quantity | None


@dataclass(frozen=True, slots=True)
class InstrumentSelection:
    scale: Scale
    disperser: Disperser
    atmosphere: Atmosphere
    optical_components: tuple[OpticalComponent, ...]


@dataclass(frozen=True, slots=True)
class InstrumentDefinition:
    root: Path
    name: str
    telescope: TelescopeDefinition
    detector: DetectorDefinition
    scales: dict[str, Scale]
    dispersers: dict[str, Disperser]
    atmospheres: dict[str, Atmosphere]
    optical_components: tuple[OpticalComponent, ...]

    def select(
        self,
        *,
        scale: str,
        disperser: str,
        atmosphere: str,
    ) -> InstrumentSelection:
        """Resolve exact option names and validate their combined data."""

        selected_scale = _select_option("scale", scale, self.scales)
        selected_disperser = _select_option("disperser", disperser, self.dispersers)
        selected_atmosphere = _select_option(
            "atmosphere", atmosphere, self.atmospheres
        )

        components = tuple(
            component
            for component in self.optical_components
            if component.disperser_scope is None
            or disperser == component.disperser_scope
            or disperser.startswith(f"{component.disperser_scope}.")
        )
        names = [component.name for component in components]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Optical path for disperser {disperser!r} has duplicate "
                "component names."
            )
        orders = [component.order for component in components]
        if len(orders) != len(set(orders)):
            raise ValueError(
                f"Optical path for disperser {disperser!r} has duplicate "
                "order values."
            )
        components = tuple(
            sorted(components, key=lambda component: component.order)
        )

        tables = [selected_atmosphere.transmission, selected_atmosphere.background]
        if self.detector.quantum_efficiency_table is not None:
            tables.append(self.detector.quantum_efficiency_table)
        for table in tables:
            _validate_coverage(table, selected_disperser)

        return InstrumentSelection(
            scale=selected_scale,
            disperser=selected_disperser,
            atmosphere=selected_atmosphere,
            optical_components=components,
        )


def load_instrument(instrument_data: str | Path) -> InstrumentDefinition:
    """Load and validate an instrument-data directory and all INI references."""

    root = _resolve_root(instrument_data)
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    with (root / "etc.ini").open(encoding="utf-8") as stream:
        parser.read_file(stream)

    if parser.defaults():
        raise ValueError("etc.ini must not define DEFAULT values.")
    _validate_section_inventory(parser)

    system = _section(parser, "system", required={"name"})
    name = system["name"].strip()
    if not name:
        raise ValueError("[system] name must be non-empty.")

    telescope_values = _section(
        parser,
        "telescope",
        required={"primary_diameter", "central_obscuration", "f_number"},
    )
    telescope = TelescopeDefinition(
        primary_diameter=_positive_float(
            telescope_values["primary_diameter"], "telescope.primary_diameter"
        )
        * u.m,
        central_obscuration=_positive_float(
            telescope_values["central_obscuration"],
            "telescope.central_obscuration",
        )
        * u.m,
        f_number=_positive_float(telescope_values["f_number"], "telescope.f_number"),
    )
    if telescope.central_obscuration >= telescope.primary_diameter:
        raise ValueError(
            "telescope.central_obscuration must be smaller than primary_diameter."
        )

    detector = _load_detector(parser, root)
    scales = _load_scales(parser)
    dispersers, disperser_scopes = _load_dispersers(parser)
    atmospheres = _load_atmospheres(parser, root)
    optical_components = _load_optical_components(parser, disperser_scopes)

    return InstrumentDefinition(
        root=root,
        name=name,
        telescope=telescope,
        detector=detector,
        scales=scales,
        dispersers=dispersers,
        atmospheres=atmospheres,
        optical_components=optical_components,
    )


def _resolve_root(instrument_data: str | Path) -> Path:
    root = Path(instrument_data).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Instrument-data directory does not exist: {root}")
    config_path = root / "etc.ini"
    if not config_path.is_file():
        raise FileNotFoundError(f"Instrument-data directory has no etc.ini: {root}")
    return root


def _validate_section_inventory(parser: configparser.ConfigParser) -> None:
    required = {"system", "telescope", "detector"}
    missing = required.difference(parser.sections())
    if missing:
        raise ValueError(f"etc.ini is missing required sections: {sorted(missing)}")

    counts = {"scale": 0, "disperser": 0, "atmosphere": 0}
    for section in parser.sections():
        if section in required:
            continue
        parts = section.split(".")
        if parts[0] in {"scale", "atmosphere"} and len(parts) == 2:
            _validate_name(parts[1], section)
            counts[parts[0]] += 1
            continue
        if parts[0] == "disperser" and len(parts) >= 2:
            for name in parts[1:]:
                _validate_name(name, section)
            counts[parts[0]] += 1
            continue
        if parts[0] == "optics" and len(parts) >= 2:
            for name in parts[1:]:
                _validate_name(name, section)
            continue
        raise ValueError(f"Unknown etc.ini section: [{section}]")

    empty = [namespace for namespace, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"etc.ini must define at least one option for: {empty}")


def _section(
    parser: configparser.ConfigParser,
    name: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, str]:
    optional = optional or set()
    values = dict(parser[name])
    missing = required.difference(values)
    unknown = set(values).difference(required | optional)
    if missing:
        raise ValueError(f"[{name}] is missing required keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"[{name}] has unknown keys: {sorted(unknown)}")
    return values


def _load_detector(
    parser: configparser.ConfigParser, root: Path
) -> DetectorDefinition:
    values = _section(
        parser,
        "detector",
        required={"read_noise", "dark_current"},
        optional={
            "quantum_efficiency",
            "quantum_efficiency_file",
        },
    )
    qe_keys = {"quantum_efficiency", "quantum_efficiency_file"}.intersection(values)
    if len(qe_keys) != 1:
        raise ValueError(
            "[detector] must define exactly one of quantum_efficiency and "
            "quantum_efficiency_file."
        )

    qe: float | None = None
    qe_table: SpectralTable | None = None
    if "quantum_efficiency" in values:
        qe = _bounded_float(
            values["quantum_efficiency"], "detector.quantum_efficiency"
        )
    else:
        path = _resolve_reference(root, values["quantum_efficiency_file"])
        qe_table = _load_table(
            path,
            value_column="quantum_efficiency",
            value_unit=u.dimensionless_unscaled,
            bounded=True,
            uniform=False,
        )

    return DetectorDefinition(
        read_noise=_nonnegative_float(values["read_noise"], "detector.read_noise")
        * u.electron,
        dark_current=_nonnegative_float(
            values["dark_current"], "detector.dark_current"
        )
        * u.electron
        / u.s,
        quantum_efficiency=qe,
        quantum_efficiency_table=qe_table,
    )


def _load_scales(
    parser: configparser.ConfigParser,
) -> dict[str, Scale]:
    scales: dict[str, Scale] = {}
    for section_name in parser.sections():
        if not section_name.startswith("scale."):
            continue
        name = section_name.removeprefix("scale.")
        values = _section(
            parser,
            section_name,
            required={"spaxels_x", "spaxels_y", "spaxel_scale"},
        )
        scales[name] = Scale(
            name=name,
            spaxels_x=_positive_int(values["spaxels_x"], f"{section_name}.spaxels_x"),
            spaxels_y=_positive_int(values["spaxels_y"], f"{section_name}.spaxels_y"),
            spaxel_scale=_positive_float(
                values["spaxel_scale"], f"{section_name}.spaxel_scale"
            )
            * u.mas,
        )
    return scales


def _load_dispersers(
    parser: configparser.ConfigParser,
) -> tuple[dict[str, Disperser], frozenset[str]]:
    section_values: dict[str, dict[str, str]] = {}
    section_names: dict[str, str] = {}
    for section_name in parser.sections():
        if not section_name.startswith("disperser."):
            continue
        name = section_name.removeprefix("disperser.")
        section_names[name] = section_name
        section_values[name] = _section(
            parser,
            section_name,
            required=set(),
            optional=_DISPERSER_KEYS,
        )

    scopes = frozenset(section_values)
    for name, section_name in section_names.items():
        for ancestor in _name_ancestors(name)[:-1]:
            if ancestor not in scopes:
                raise ValueError(
                    f"[{section_name}] requires parent section "
                    f"[disperser.{ancestor}]."
                )

    leaf_names = [
        name
        for name in section_values
        if not any(other.startswith(f"{name}.") for other in section_values)
    ]
    dispersers: dict[str, Disperser] = {}
    for name in leaf_names:
        section_name = section_names[name]
        values: dict[str, str] = {}
        for ancestor in _name_ancestors(name):
            values.update(section_values[ancestor])
        missing = _DISPERSER_KEYS.difference(values)
        if missing:
            raise ValueError(
                f"[{section_name}] is missing required keys after inheritance: "
                f"{sorted(missing)}"
            )
        wavelength_min = _positive_float(
            values["wavelength_min"], f"{section_name}.wavelength_min"
        ) * u.micron
        wavelength_max = _positive_float(
            values["wavelength_max"], f"{section_name}.wavelength_max"
        ) * u.micron
        if wavelength_max <= wavelength_min:
            raise ValueError(
                f"[{section_name}] wavelength_max must exceed wavelength_min."
            )
        dispersers[name] = Disperser(
            name=name,
            resolving_power=_positive_float(
                values["resolving_power"], f"{section_name}.resolving_power"
            ),
            pixels_per_resolution_element=_positive_float(
                values["pixels_per_resolution_element"],
                f"{section_name}.pixels_per_resolution_element",
            ),
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
    return dispersers, scopes


def _load_atmospheres(
    parser: configparser.ConfigParser, root: Path
) -> dict[str, Atmosphere]:
    atmospheres: dict[str, Atmosphere] = {}
    for section_name in parser.sections():
        if not section_name.startswith("atmosphere."):
            continue
        name = section_name.removeprefix("atmosphere.")
        values = _section(
            parser,
            section_name,
            required={"pwv", "airmass", "transmission_file", "background_file"},
        )
        transmission_path = _resolve_reference(root, values["transmission_file"])
        background_path = _resolve_reference(root, values["background_file"])
        atmospheres[name] = Atmosphere(
            name=name,
            pwv=_positive_float(values["pwv"], f"{section_name}.pwv") * u.mm,
            airmass=_positive_float(values["airmass"], f"{section_name}.airmass"),
            transmission=_load_table(
                transmission_path,
                value_column="transmission",
                value_unit=u.dimensionless_unscaled,
                bounded=True,
                uniform=True,
            ),
            background=_load_table(
                background_path,
                value_column="background",
                value_unit=u.photon / (u.s * u.m**2 * u.arcsec**2 * u.micron),
                bounded=False,
                uniform=True,
            ),
        )
    return atmospheres


def _load_optical_components(
    parser: configparser.ConfigParser,
    disperser_scopes: frozenset[str],
) -> tuple[OpticalComponent, ...]:
    components: list[OpticalComponent] = []
    for section_name in parser.sections():
        if not section_name.startswith("optics."):
            continue
        parts = section_name.split(".")
        name = parts[1]
        disperser_scope = ".".join(parts[2:]) if len(parts) > 2 else None
        if disperser_scope is not None and disperser_scope not in disperser_scopes:
            raise ValueError(
                f"[{section_name}] refers to unknown disperser scope "
                f"{disperser_scope!r}."
            )
        values = _section(
            parser,
            section_name,
            required={"order", "throughput", "emissivity"},
            optional={"temperature"},
        )
        emissivity = _bounded_float(
            values["emissivity"], f"{section_name}.emissivity"
        )
        if emissivity > 0 and "temperature" not in values:
            raise ValueError(
                f"[{section_name}] requires temperature when emissivity is non-zero."
            )
        temperature = None
        if "temperature" in values:
            temperature = _positive_float(
                values["temperature"], f"{section_name}.temperature"
            ) * u.K
        components.append(
            OpticalComponent(
                name=name,
                disperser_scope=disperser_scope,
                order=_integer(values["order"], f"{section_name}.order"),
                throughput=_bounded_float(
                    values["throughput"], f"{section_name}.throughput"
                ),
                emissivity=emissivity,
                temperature=temperature,
            )
        )
    return tuple(components)


def _name_ancestors(name: str) -> tuple[str, ...]:
    parts = name.split(".")
    return tuple(".".join(parts[:index]) for index in range(1, len(parts) + 1))


def _resolve_reference(root: Path, value: str) -> Path:
    value = value.strip()
    if not value:
        raise ValueError("Referenced filenames must be non-empty.")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"INI file references must be relative: {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"INI file reference escapes the instrument bundle: {value!r}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Referenced instrument file does not exist: {path}")
    return path


def _load_table(
    path: Path,
    *,
    value_column: str,
    value_unit: u.UnitBase,
    bounded: bool,
    uniform: bool,
) -> SpectralTable:
    try:
        table = QTable.read(path, format="ascii.ecsv")
    except Exception as exc:
        raise ValueError(f"Could not read ECSV table {path}: {exc}") from exc
    expected = {"wavelength", value_column}
    if len(table.colnames) != 2 or set(table.colnames) != expected:
        raise ValueError(
            f"{path} must contain exactly these columns: {sorted(expected)}"
        )

    wavelength = _table_quantity(table, "wavelength", u.micron, path)
    values = _table_quantity(table, value_column, value_unit, path)
    if wavelength.ndim != 1 or values.ndim != 1 or len(wavelength) != len(values):
        raise ValueError(f"{path} columns must be one-dimensional and equal length.")
    if len(wavelength) < 2:
        raise ValueError(f"{path} must contain at least two wavelength samples.")
    if not np.isfinite(wavelength.value).all() or np.any(wavelength.value <= 0):
        raise ValueError(f"{path} wavelength must be finite and positive.")
    spacing = np.diff(wavelength.value)
    if np.any(spacing <= 0):
        raise ValueError(f"{path} wavelength must be unique and strictly increasing.")
    if uniform and not np.allclose(
        spacing, spacing[0], rtol=_UNIFORM_RTOL, atol=0.0
    ):
        raise ValueError(
            f"{path} wavelength must be uniformly sampled within rtol={_UNIFORM_RTOL}."
        )
    if not np.isfinite(values.value).all() or np.any(values.value < 0):
        raise ValueError(f"{path} {value_column} must be finite and non-negative.")
    if bounded and np.any(values.value > 1):
        raise ValueError(f"{path} {value_column} must not exceed one.")

    wavelength.value.setflags(write=False)
    values.value.setflags(write=False)
    return SpectralTable(path=path, wavelength=wavelength, values=values)


def _table_quantity(
    table: QTable, column: str, unit: u.UnitBase, path: Path
) -> u.Quantity:
    if table[column].unit is None:
        raise ValueError(f"{path} column {column!r} must declare units.")
    try:
        quantity = u.Quantity(table[column], copy=True).to(unit)
    except u.UnitConversionError as exc:
        raise ValueError(
            f"{path} column {column!r} units must be convertible to {unit}."
        ) from exc
    if quantity.value.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{path} column {column!r} must contain real numbers.")
    return quantity


def _validate_coverage(table: SpectralTable, disperser: Disperser) -> None:
    if (
        table.wavelength[0] > disperser.wavelength_min
        or table.wavelength[-1] < disperser.wavelength_max
    ):
        raise ValueError(
            f"{table.path} does not cover disperser {disperser.name!r} "
            f"({disperser.wavelength_min} to {disperser.wavelength_max})."
        )


def _select_option(kind: str, name: str, options: dict[str, _Option]) -> _Option:
    try:
        return options[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {kind} {name!r}; available options: {sorted(options)}"
        ) from exc


def _validate_name(name: str, section: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Section [{section}] names must use lowercase ASCII letters, digits, "
            "hyphens, and underscores."
        )


def _number(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _positive_float(value: str, field: str) -> float:
    number = _number(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive.")
    return number


def _nonnegative_float(value: str, field: str) -> float:
    number = _number(value, field)
    if number < 0:
        raise ValueError(f"{field} must be non-negative.")
    return number


def _bounded_float(value: str, field: str) -> float:
    number = _number(value, field)
    if not 0 <= number <= 1:
        raise ValueError(f"{field} must be between zero and one inclusive.")
    return number


def _integer(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer.") from exc


def _positive_int(value: str, field: str) -> int:
    number = _integer(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive.")
    return number
