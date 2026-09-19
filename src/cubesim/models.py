"""Public target-model configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS


@dataclass(frozen=True, slots=True)
class Point:
    """An unresolved spatial profile.

    A calculation containing this profile requires a configured PSF.
    """


@dataclass(frozen=True, slots=True)
class Uniform:
    """A uniform surface-brightness spatial profile.

    Its spectrum must use surface-brightness flux units. A calculation in which
    every target is uniform does not require a PSF.
    """


@dataclass(frozen=True, slots=True)
class Gaussian:
    """A normalized elliptical Gaussian spatial profile.

    Args:
        fwhm: Positive angular major-axis full width at half maximum.
        axis_ratio: Minor-to-major axis ratio in ``(0, 1]``.
        position_angle: Angular position angle measured east of north.
    """

    fwhm: u.Quantity
    axis_ratio: float = 1.0
    position_angle: u.Quantity = 0 * u.deg

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fwhm", _positive_quantity(self.fwhm, u.arcsec, "fwhm")
        )
        axis_ratio = _real_scalar(self.axis_ratio, "axis_ratio")
        if not 0 < axis_ratio <= 1:
            raise ValueError("axis_ratio must be greater than zero and at most one.")
        object.__setattr__(self, "axis_ratio", axis_ratio)
        object.__setattr__(
            self,
            "position_angle",
            _finite_quantity(self.position_angle, u.deg, "position_angle"),
        )


@dataclass(frozen=True, slots=True)
class Sersic:
    """A normalized elliptical Sersic spatial profile.

    Args:
        effective_radius: Positive angular effective radius.
        index: Positive Sersic index.
        axis_ratio: Minor-to-major axis ratio in ``(0, 1]``.
        position_angle: Angular position angle measured east of north.
    """

    effective_radius: u.Quantity
    index: float
    axis_ratio: float = 1.0
    position_angle: u.Quantity = 0 * u.deg

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_radius",
            _positive_quantity(
                self.effective_radius,
                u.arcsec,
                "effective_radius",
            ),
        )
        index = _real_scalar(self.index, "index")
        if index <= 0:
            raise ValueError("index must be positive.")
        object.__setattr__(self, "index", index)
        axis_ratio = _real_scalar(self.axis_ratio, "axis_ratio")
        if not 0 < axis_ratio <= 1:
            raise ValueError("axis_ratio must be greater than zero and at most one.")
        object.__setattr__(self, "axis_ratio", axis_ratio)
        object.__setattr__(
            self,
            "position_angle",
            _finite_quantity(self.position_angle, u.deg, "position_angle"),
        )


@dataclass(frozen=True, slots=True, init=False)
class SpatialImage:
    """A normalized intrinsic spatial profile with a sky orientation.

    Args:
        source: In-memory two-dimensional array, NPY filename, or FITS
            filename. Values must be finite and nonnegative with positive
            total flux.
        pixel_scale: Positive angular pixel scale required for in-memory and
            NPY inputs. FITS input reads ``PIXSCALE`` in milliarcseconds per
            pixel.
        position_angle: Optional sky position angle east of north for
            in-memory and NPY inputs. FITS input owns its orientation and may
            derive it from celestial WCS.

    The input is copied, normalized to unit total, and stored read-only.
    Celestial WCS scale, when present in FITS, must agree with ``PIXSCALE``.
    """

    data: np.ndarray
    pixel_scale: u.Quantity
    position_angle: u.Quantity
    path: Path | None

    def __init__(
        self,
        source: Any,
        *,
        pixel_scale: Any | None = None,
        position_angle: Any | None = None,
    ) -> None:
        path = None
        fits_header = None
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Spatial image does not exist: {path}")
            suffix = path.suffix.lower()
            if suffix == ".npy":
                data = np.load(path, allow_pickle=False)
                if pixel_scale is None:
                    raise ValueError("NPY spatial images require pixel_scale.")
            elif suffix in {".fit", ".fits", ".fts"}:
                if pixel_scale is not None:
                    raise ValueError("FITS spatial images read PIXSCALE from the file.")
                if position_angle is not None:
                    raise ValueError(
                        "FITS spatial images own their orientation; do not pass "
                        "position_angle."
                    )
                data, fits_header = fits.getdata(path, header=True)
                if "PIXSCALE" not in fits_header:
                    raise ValueError("FITS spatial image is missing PIXSCALE.")
                pixel_scale = fits_header["PIXSCALE"] * u.mas
            else:
                raise ValueError("Spatial image files must use FITS or NPY format.")
        else:
            data = source
            if pixel_scale is None:
                raise ValueError("In-memory spatial images require pixel_scale.")

        data = _spatial_array(data, nonnegative=True, positive_sum=True)
        pixel_scale = _positive_quantity(pixel_scale, u.mas, "pixel_scale")
        if fits_header is not None:
            wcs_angle, wcs_scale = _fits_spatial_wcs(fits_header)
            if wcs_scale is not None and not np.isclose(
                wcs_scale.to_value(u.mas),
                pixel_scale.to_value(u.mas),
                rtol=1e-6,
            ):
                raise ValueError("FITS celestial WCS scale conflicts with PIXSCALE.")
            if wcs_angle is not None:
                if position_angle is not None:
                    raise ValueError(
                        "FITS celestial WCS supplies orientation; do not pass "
                        "position_angle."
                    )
                position_angle = wcs_angle
        if position_angle is None:
            position_angle = 0 * u.deg
        position_angle = _finite_quantity(
            position_angle,
            u.deg,
            "position_angle",
        )
        data /= data.sum()
        data.setflags(write=False)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "pixel_scale", pixel_scale)
        object.__setattr__(self, "position_angle", position_angle)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class ConstantVelocity:
    """A constant peculiar line-of-sight velocity offset.

    Args:
        offset: Finite velocity with magnitude below the speed of light.
    """

    offset: u.Quantity

    def __post_init__(self) -> None:
        offset = _finite_quantity(self.offset, u.km / u.s, "offset")
        if abs(offset) >= c:
            raise ValueError("offset magnitude must be less than the speed of light.")
        object.__setattr__(self, "offset", offset)


@dataclass(frozen=True, slots=True)
class RotatingDisk:
    """An arctangent rotating-disk line-of-sight velocity model.

    Args:
        maximum_velocity: Positive asymptotic intrinsic circular velocity.
        turnover_radius: Positive angular turnover radius.
        inclination: Angular inclination in ``[0, 90)`` degrees.
        position_angle: Position angle of the receding major axis, measured
            east of north.
        systemic_velocity: Constant velocity added to the projected field.
    """

    maximum_velocity: u.Quantity
    turnover_radius: u.Quantity
    inclination: u.Quantity
    position_angle: u.Quantity
    systemic_velocity: u.Quantity = 0 * u.km / u.s

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_velocity",
            _positive_quantity(
                self.maximum_velocity,
                u.km / u.s,
                "maximum_velocity",
            ),
        )
        object.__setattr__(
            self,
            "turnover_radius",
            _positive_quantity(self.turnover_radius, u.arcsec, "turnover_radius"),
        )
        inclination = _finite_quantity(self.inclination, u.deg, "inclination")
        if not 0 <= inclination.value < 90:
            raise ValueError("inclination must be at least zero and less than 90 deg.")
        object.__setattr__(self, "inclination", inclination)
        object.__setattr__(
            self,
            "position_angle",
            _finite_quantity(self.position_angle, u.deg, "position_angle"),
        )
        object.__setattr__(
            self,
            "systemic_velocity",
            _finite_quantity(
                self.systemic_velocity,
                u.km / u.s,
                "systemic_velocity",
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class VelocityField:
    """A finite sky-oriented line-of-sight velocity field.

    Args:
        source: In-memory two-dimensional velocity quantity, NPY filename, or
            FITS filename.
        unit: Velocity unit required for NPY input. In-memory quantities carry
            their own unit, and FITS input reads ``BUNIT``.
        pixel_scale: Positive angular pixel scale required for in-memory and
            NPY inputs. FITS input reads ``PIXSCALE`` in milliarcseconds per
            pixel.
        position_angle: Optional sky position angle east of north for
            in-memory and NPY inputs. FITS input owns its orientation and may
            derive it from celestial WCS.

    The field must cover the complete selected IFU after transformation.
    Coverage is validated during the calculation.
    """

    data: u.Quantity
    pixel_scale: u.Quantity
    position_angle: u.Quantity
    path: Path | None

    def __init__(
        self,
        source: Any,
        *,
        unit: Any | None = None,
        pixel_scale: Any | None = None,
        position_angle: Any | None = None,
    ) -> None:
        path = None
        fits_header = None
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Velocity field does not exist: {path}")
            suffix = path.suffix.lower()
            if suffix == ".npy":
                if unit is None or pixel_scale is None:
                    raise ValueError(
                        "NPY velocity fields require unit and pixel_scale."
                    )
                data = np.load(path, allow_pickle=False)
                data = data * _velocity_unit(unit)
            elif suffix in {".fit", ".fits", ".fts"}:
                if any(
                    value is not None for value in (unit, pixel_scale, position_angle)
                ):
                    raise ValueError(
                        "FITS velocity fields own unit, pixel scale, and orientation."
                    )
                data, fits_header = fits.getdata(path, header=True)
                if "BUNIT" not in fits_header:
                    raise ValueError("FITS velocity field is missing BUNIT.")
                if "PIXSCALE" not in fits_header:
                    raise ValueError("FITS velocity field is missing PIXSCALE.")
                data = data * _velocity_unit(fits_header["BUNIT"])
                pixel_scale = fits_header["PIXSCALE"] * u.mas
            else:
                raise ValueError("Velocity field files must use FITS or NPY format.")
        else:
            if unit is not None:
                raise ValueError("In-memory velocity fields carry their own unit.")
            if pixel_scale is None:
                raise ValueError("In-memory velocity fields require pixel_scale.")
            data = source

        data = _velocity_array(data)
        pixel_scale = _positive_quantity(pixel_scale, u.mas, "pixel_scale")
        if fits_header is not None:
            wcs_angle, wcs_scale = _fits_spatial_wcs(fits_header)
            if wcs_scale is not None and not np.isclose(
                wcs_scale.to_value(u.mas),
                pixel_scale.to_value(u.mas),
                rtol=1e-6,
            ):
                raise ValueError("FITS celestial WCS scale conflicts with PIXSCALE.")
            if wcs_angle is not None:
                position_angle = wcs_angle
        if position_angle is None:
            position_angle = 0 * u.deg
        position_angle = _finite_quantity(
            position_angle,
            u.deg,
            "position_angle",
        )
        data.setflags(write=False)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "pixel_scale", pixel_scale)
        object.__setattr__(self, "position_angle", position_angle)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True, init=False)
class TabulatedSpectrum:
    """A validated flux-density spectrum on an independent wavelength grid.

    Args:
        source: Optional ECSV or FITS table filename containing wavelength and
            flux columns.
        wavelength: In-memory one-dimensional wavelength quantity. Required
            with ``flux`` when ``source`` is omitted.
        flux: In-memory one-dimensional flux-density quantity matching
            ``wavelength``.
        medium: ``"air"`` or ``"vacuum"``. File metadata may provide this;
            the default is vacuum.

    Wavelengths must be finite, positive, unique, strictly increasing, and
    contain at least two samples. Air wavelengths are converted to vacuum.
    The tabulation must cover the selected disperser and any velocity-expanded
    margins; extrapolation is rejected during the calculation.
    """

    wavelength: u.Quantity
    flux: u.Quantity
    medium: str
    path: Path | None

    def __init__(
        self,
        source: str | Path | None = None,
        *,
        wavelength: Any | None = None,
        flux: Any | None = None,
        medium: str | None = None,
    ) -> None:
        if source is not None:
            if wavelength is not None or flux is not None:
                raise ValueError(
                    "File-backed spectra do not accept wavelength or flux."
                )
            path = Path(source).expanduser().resolve()
            wavelength, flux, file_medium = _load_spectrum(path)
            if medium is not None and file_medium is not None and medium != file_medium:
                raise ValueError("Explicit medium conflicts with spectrum metadata.")
            medium = medium or file_medium or "vacuum"
        else:
            if wavelength is None or flux is None:
                raise ValueError("In-memory spectra require wavelength and flux.")
            path = None
            medium = medium or "vacuum"
        if medium not in {"air", "vacuum"}:
            raise ValueError("medium must be 'air' or 'vacuum'.")

        wavelength = _positive_array_quantity(wavelength, u.micron, "wavelength")
        if len(wavelength) < 2 or np.any(np.diff(wavelength.value) <= 0):
            raise ValueError("wavelength must be strictly increasing and unique.")
        if medium == "air":
            wavelength = _air_to_vacuum(wavelength)
        flux = _flux_density(flux)
        if flux.isscalar or flux.ndim != 1 or len(flux) != len(wavelength):
            raise ValueError("flux must be one-dimensional and match wavelength.")

        object.__setattr__(self, "wavelength", _readonly_quantity(wavelength))
        object.__setattr__(self, "flux", _readonly_quantity(flux))
        object.__setattr__(self, "medium", medium)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True, init=False)
class GaussianLines:
    """One or more Gaussian emission lines on an optional flat continuum.

    Args:
        wavelength: Observed line-center wavelength or wavelengths. Exactly one
            of ``wavelength`` and ``rest_wavelength`` is required.
        rest_wavelength: Rest-frame line-center wavelength or wavelengths.
            Requires ``redshift``.
        redshift: Redshift greater than ``-1``. Valid only with
            ``rest_wavelength``.
        medium: ``"air"`` or ``"vacuum"``. Air wavelengths are converted to
            vacuum before redshift is applied.
        flux: Integrated line flux for non-uniform spatial models or line
            surface brightness for ``Uniform``.
        flux_ratios: Nonnegative ratios for lines after the first, used with a
            scalar reference ``flux``.
        dispersion: Positive Gaussian standard deviation in velocity or
            wavelength units. Exactly one of ``dispersion`` and ``fwhm`` is
            required.
        fwhm: Positive Gaussian full width at half maximum in velocity or
            wavelength units.
        continuum: Optional flat flux density with units consistent with the
            line-flux form.

    A scalar flux defines one line unless ``flux_ratios`` is supplied. An array
    of fluxes must match the number of line centers. Lines outside the selected
    high-resolution wavelength grid contribute zero.
    """

    wavelength: u.Quantity
    rest_wavelength: u.Quantity | None
    redshift: float | None
    medium: str
    flux: u.Quantity
    flux_ratios: tuple[float, ...] | None
    dispersion: u.Quantity | None
    fwhm: u.Quantity | None
    continuum: u.Quantity | None

    def __init__(
        self,
        *,
        wavelength: Any | None = None,
        rest_wavelength: Any | None = None,
        redshift: float | None = None,
        medium: str = "vacuum",
        flux: Any,
        flux_ratios: Any | None = None,
        dispersion: Any | None = None,
        fwhm: Any | None = None,
        continuum: Any | None = None,
    ) -> None:
        if (wavelength is None) == (rest_wavelength is None):
            raise ValueError("Provide exactly one of wavelength and rest_wavelength.")
        if rest_wavelength is not None and redshift is None:
            raise ValueError("rest_wavelength requires redshift.")
        if wavelength is not None and redshift is not None:
            raise ValueError("redshift is only valid with rest_wavelength.")
        if medium not in {"air", "vacuum"}:
            raise ValueError("medium must be 'air' or 'vacuum'.")
        if (dispersion is None) == (fwhm is None):
            raise ValueError("Provide exactly one of dispersion and fwhm.")

        resolved_redshift: float | None = None
        rest: u.Quantity | None = None
        if rest_wavelength is not None:
            resolved_redshift = _real_scalar(redshift, "redshift")
            if resolved_redshift <= -1:
                raise ValueError("redshift must be greater than -1.")
            rest = _positive_array_quantity(
                rest_wavelength,
                u.micron,
                "rest_wavelength",
            )
            if medium == "air":
                rest = _air_to_vacuum(rest)
            wavelengths = _positive_array_quantity(
                rest * (1 + resolved_redshift),
                u.micron,
                "wavelength",
            )
        else:
            wavelengths = _positive_array_quantity(
                wavelength,
                u.micron,
                "wavelength",
            )
            if medium == "air":
                wavelengths = _air_to_vacuum(wavelengths)

        line_flux = _line_flux(flux)
        ratios: tuple[float, ...] | None = None
        if flux_ratios is not None:
            ratio_array = np.asarray(flux_ratios, dtype=float)
            if ratio_array.ndim != 1 or not np.isfinite(ratio_array).all():
                raise ValueError("flux_ratios must be a finite one-dimensional array.")
            if np.any(ratio_array < 0):
                raise ValueError("flux_ratios must be non-negative.")
            if len(ratio_array) != len(wavelengths) - 1:
                raise ValueError(
                    "flux_ratios must contain one fewer value than wavelength."
                )
            if not line_flux.isscalar:
                raise ValueError("flux_ratios require a scalar reference flux.")
            ratios = tuple(float(value) for value in ratio_array)
        elif not line_flux.isscalar and len(line_flux) != len(wavelengths):
            raise ValueError("flux must be scalar or match the wavelength count.")
        elif line_flux.isscalar and len(wavelengths) != 1:
            raise ValueError("Multiple lines require flux_ratios or one flux per line.")

        resolved_dispersion = None
        resolved_fwhm = None
        if dispersion is not None:
            resolved_dispersion = _positive_width(dispersion, "dispersion")
        else:
            resolved_fwhm = _positive_width(fwhm, "fwhm")

        resolved_continuum = None
        if continuum is not None:
            resolved_continuum = _continuum_flux(continuum, line_flux)

        object.__setattr__(self, "wavelength", _readonly_quantity(wavelengths))
        object.__setattr__(
            self,
            "rest_wavelength",
            _readonly_quantity(rest) if rest is not None else None,
        )
        object.__setattr__(self, "redshift", resolved_redshift)
        object.__setattr__(self, "medium", medium)
        object.__setattr__(self, "flux", _readonly_quantity(line_flux))
        object.__setattr__(self, "flux_ratios", ratios)
        object.__setattr__(
            self,
            "dispersion",
            _readonly_quantity(resolved_dispersion)
            if resolved_dispersion is not None
            else None,
        )
        object.__setattr__(
            self,
            "fwhm",
            _readonly_quantity(resolved_fwhm) if resolved_fwhm is not None else None,
        )
        object.__setattr__(
            self,
            "continuum",
            _readonly_quantity(resolved_continuum)
            if resolved_continuum is not None
            else None,
        )


def _positive_width(value: Any, name: str) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a quantity.")
    if value.unit.is_equivalent(u.km / u.s):
        return _positive_quantity(value, u.km / u.s, name)
    if value.unit.is_equivalent(u.micron):
        return _positive_quantity(value, u.micron, name)
    raise u.UnitConversionError(f"{name} must have velocity or wavelength units.")


def _line_flux(value: Any) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError("flux must be a quantity.")
    for unit in (
        u.erg / (u.s * u.cm**2),
        u.erg / (u.s * u.cm**2 * u.arcsec**2),
    ):
        try:
            return _positive_array_quantity(
                value,
                unit,
                "flux",
                allow_scalar=True,
            )
        except u.UnitConversionError:
            continue
    raise u.UnitConversionError("flux must be integrated flux or surface brightness.")


def _flux_density(value: Any) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError("flux must be a quantity.")
    for unit in (
        u.erg / (u.s * u.cm**2 * u.m),
        u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.m),
    ):
        try:
            quantity = value.to(unit)
        except u.UnitConversionError:
            continue
        if not np.isfinite(quantity.value).all() or np.any(quantity.value < 0):
            raise ValueError("flux must be finite and non-negative.")
        return u.Quantity(quantity, copy=True)
    raise u.UnitConversionError(
        "flux must be flux density or surface-brightness density."
    )


def _continuum_flux(value: Any, line_flux: u.Quantity) -> u.Quantity:
    if line_flux.unit.is_equivalent(u.erg / (u.s * u.cm**2 * u.arcsec**2)):
        unit = u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.m)
    else:
        unit = u.erg / (u.s * u.cm**2 * u.m)
    return _nonnegative_quantity(value, unit, "continuum")


def _air_to_vacuum(wavelength: u.Quantity) -> u.Quantity:
    wavelength = wavelength.to(u.micron)
    sigma_squared = (1 / wavelength.to_value(u.micron)) ** 2
    refractive_index = (
        1
        + 8.34254e-5
        + 2.406147e-2 / (130 - sigma_squared)
        + 1.5998e-4 / (38.9 - sigma_squared)
    )
    return wavelength * refractive_index


def _load_spectrum(path: Path) -> tuple[u.Quantity, u.Quantity, str | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Spectrum file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".ecsv":
        table = QTable.read(path, format="ascii.ecsv")
        file_medium = table.meta.get("medium")
    elif suffix in {".fit", ".fits", ".fts"}:
        table = QTable.read(path, format="fits")
        with fits.open(path) as hdus:
            table_hdu = next(
                (hdu for hdu in hdus if isinstance(hdu, fits.BinTableHDU)),
                None,
            )
            if table_hdu is None:
                raise ValueError("Spectrum FITS must contain a binary table.")
            names = list(table_hdu.columns.names)
            if "wavelength" not in names or "flux" not in names:
                raise ValueError("Spectrum table requires wavelength and flux columns.")
            wavelength_index = names.index("wavelength") + 1
            coordinate_type = table_hdu.header.get(f"TCTYP{wavelength_index}")
            file_medium = {"WAVE": "vacuum", "AWAV": "air"}.get(coordinate_type)
    else:
        raise ValueError("Spectrum files must use ECSV or FITS format.")
    if "wavelength" not in table.colnames or "flux" not in table.colnames:
        raise ValueError("Spectrum table requires wavelength and flux columns.")
    if file_medium is not None and file_medium not in {"air", "vacuum"}:
        raise ValueError("Spectrum medium metadata must be 'air' or 'vacuum'.")
    return u.Quantity(table["wavelength"]), u.Quantity(table["flux"]), file_medium


def _spatial_array(
    values: Any,
    *,
    nonnegative: bool,
    positive_sum: bool,
) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("Spatial data must be a real numeric 2D array.")
    array = np.array(source, dtype=float, copy=True)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Spatial data must be a nonempty 2D array.")
    if not np.isfinite(array).all():
        raise ValueError("Spatial data must contain only finite values.")
    if nonnegative and np.any(array < 0):
        raise ValueError("Spatial data must be non-negative.")
    if positive_sum and array.sum() <= 0:
        raise ValueError("Spatial image total flux must be strictly positive.")
    return array


def _velocity_unit(value: Any) -> u.UnitBase:
    try:
        unit = u.Unit(value)
        unit.to(u.km / u.s)
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise u.UnitConversionError(
            "Velocity-field unit must be a velocity unit."
        ) from exc
    return unit


def _velocity_array(values: Any) -> u.Quantity:
    if not isinstance(values, u.Quantity):
        raise TypeError("Velocity field must be a quantity.")
    try:
        quantity = values.to(u.km / u.s)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError("Velocity field must have velocity units.") from exc
    source = np.asarray(quantity.value)
    if source.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("Velocity field must be a real numeric 2D array.")
    array = np.array(source, dtype=float, copy=True)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Velocity field must be a nonempty 2D array.")
    if not np.isfinite(array).all():
        raise ValueError("Velocity field must contain only finite values.")
    return array * u.km / u.s


def _fits_spatial_wcs(
    header: fits.Header,
) -> tuple[u.Quantity | None, u.Quantity | None]:
    wcs = WCS(header)
    if not wcs.has_celestial:
        return None, None
    matrix = wcs.celestial.pixel_scale_matrix
    scales = np.sqrt(np.sum(matrix**2, axis=0)) * u.deg
    if not np.isclose(scales[0].value, scales[1].value, rtol=1e-6):
        raise ValueError("FITS celestial WCS must use a square pixel scale.")
    east, north = matrix[:, 1]
    angle = np.arctan2(east, north) * u.rad
    return angle.to(u.deg), scales.mean().to(u.mas)


def _positive_array_quantity(
    value: Any,
    unit: u.UnitBase,
    name: str,
    *,
    allow_scalar: bool = False,
) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a quantity.")
    try:
        quantity = value.to(unit)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"{name} must be convertible to {unit}.") from exc
    if quantity.ndim > 1:
        raise TypeError(f"{name} must be scalar or one-dimensional.")
    quantity = quantity if quantity.isscalar else u.Quantity(quantity, copy=True)
    if not quantity.isscalar and quantity.ndim != 1:
        raise TypeError(f"{name} must be one-dimensional.")
    if quantity.isscalar and not allow_scalar:
        quantity = np.atleast_1d(quantity.value) * quantity.unit
    if not np.isfinite(quantity.value).all() or np.any(quantity.value <= 0):
        raise ValueError(f"{name} must be finite and positive.")
    return quantity


def _readonly_quantity(value: u.Quantity) -> u.Quantity:
    quantity = u.Quantity(value, copy=True)
    quantity.setflags(write=False)
    return quantity


def _finite_quantity(value: Any, unit: u.UnitBase, name: str) -> u.Quantity:
    if not isinstance(value, u.Quantity) or not value.isscalar:
        raise TypeError(f"{name} must be a scalar quantity.")
    try:
        quantity = value.to(unit)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"{name} must be convertible to {unit}.") from exc
    if not np.isfinite(quantity.value):
        raise ValueError(f"{name} must be finite.")
    return quantity


def _positive_quantity(value: Any, unit: u.UnitBase, name: str) -> u.Quantity:
    quantity = _finite_quantity(value, unit, name)
    if quantity.value <= 0:
        raise ValueError(f"{name} must be positive.")
    return quantity


def _nonnegative_quantity(value: Any, unit: u.UnitBase, name: str) -> u.Quantity:
    quantity = _finite_quantity(value, unit, name)
    if quantity.value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return quantity


def _real_scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"{name} must be a real scalar.")
    number = float(array)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number
