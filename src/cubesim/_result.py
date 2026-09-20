"""Immutable result structures for forward ETC calculations."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits


@dataclass(frozen=True, slots=True)
class ExposureOptions:
    time: u.Quantity
    n: int
    n_target: int
    n_sky: int
    target_time: u.Quantity
    sky_time: u.Quantity
    total_time: u.Quantity


@dataclass(frozen=True, slots=True)
class SkySubtractionOptions:
    method: str
    sequence: str | None
    mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ResultOptions:
    instrument: str
    scale: str
    spaxel_scale: u.Quantity
    disperser: str
    atmosphere: str
    position_angle: u.Quantity
    pointing_center: Any | None
    targets: tuple[Any, ...]
    psf_pixel_scale: u.Quantity | None
    psf_path: Any | None
    exposure: ExposureOptions
    sky_subtraction: SkySubtractionOptions
    n_cubes: int


@dataclass(frozen=True, slots=True)
class ModelGrid:
    wavelength: u.Quantity
    spatial: np.ndarray
    spectrum: u.Quantity
    velocity: u.Quantity | None
    spatial_convolved: np.ndarray | None = None
    spectrum_convolved: u.Quantity | None = None


@dataclass(frozen=True, slots=True)
class TargetModels:
    high: ModelGrid
    low: ModelGrid


@dataclass(frozen=True, slots=True)
class Models:
    targets: tuple[TargetModels, ...]
    combined: u.Quantity
    transmission: np.ndarray
    sky: u.Quantity
    thermal: u.Quantity


@dataclass(frozen=True, slots=True)
class Signals:
    target: u.Quantity
    sky: u.Quantity
    thermal: u.Quantity
    dark: u.Quantity
    background: u.Quantity
    total: u.Quantity


@dataclass(frozen=True, slots=True)
class Variances:
    target: u.Quantity
    sky: u.Quantity
    thermal: u.Quantity
    dark: u.Quantity
    read: u.Quantity
    total: u.Quantity


class ApertureProjection:
    """One immutable aperture reduction retaining wavelength or position."""

    __slots__ = ("__dict__", "_locked", "snr")

    def __init__(
        self,
        *,
        snr: np.ndarray,
        signals: Signals | None = None,
        variances: Variances | None = None,
    ) -> None:
        object.__setattr__(self, "_locked", False)
        self.snr = _readonly_array(snr)
        if signals is not None:
            self.signals = readonly_signals(signals)
        if variances is not None:
            self.variances = readonly_variances(variances)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("Aperture projections are immutable.")
        object.__setattr__(self, name, value)

    def __getstate__(self) -> dict[str, Any]:
        return {"snr": self.snr, **self.__dict__}

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "snr", _readonly_array(state.pop("snr")))
        if "signals" in state:
            object.__setattr__(
                self,
                "signals",
                readonly_signals(state.pop("signals")),
            )
        if "variances" in state:
            object.__setattr__(
                self,
                "variances",
                readonly_variances(state.pop("variances")),
            )
        object.__setattr__(self, "_locked", True)


class ApertureResult:
    """One immutable registered-aperture reduction."""

    __slots__ = ("__dict__", "_locked", "maps", "mask", "name", "snr", "spectra")

    def __init__(
        self,
        *,
        name: str,
        mask: np.ndarray,
        snr: float,
        spectra: ApertureProjection,
        maps: ApertureProjection,
        signals: Any | None = None,
        variances: Any | None = None,
        data: Any | None = None,
    ) -> None:
        object.__setattr__(self, "_locked", False)
        self.name = name
        self.mask = _readonly_array(mask)
        self.snr = float(snr)
        self.spectra = readonly_aperture_projection(spectra)
        self.maps = readonly_aperture_projection(maps)
        if signals is not None:
            self.signals = readonly_signals(signals)
        if variances is not None:
            self.variances = readonly_variances(variances)
        if data is not None:
            self.data = _readonly_quantity(data)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("Aperture results are immutable.")
        object.__setattr__(self, name, value)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mask": self.mask,
            "snr": self.snr,
            "spectra": self.spectra,
            "maps": self.maps,
            **self.__dict__,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "name", state.pop("name"))
        object.__setattr__(self, "mask", _readonly_array(state.pop("mask")))
        object.__setattr__(self, "snr", float(state.pop("snr")))
        object.__setattr__(
            self,
            "spectra",
            readonly_aperture_projection(state.pop("spectra")),
        )
        object.__setattr__(
            self,
            "maps",
            readonly_aperture_projection(state.pop("maps")),
        )
        if "signals" in state:
            object.__setattr__(
                self,
                "signals",
                readonly_signals(state.pop("signals")),
            )
        if "variances" in state:
            object.__setattr__(
                self,
                "variances",
                readonly_variances(state.pop("variances")),
            )
        if "data" in state:
            object.__setattr__(
                self,
                "data",
                _readonly_quantity(state.pop("data")),
            )
        object.__setattr__(self, "_locked", True)


class EtcResult:
    """Immutable result of one resolved ETC calculation."""

    __slots__ = (
        "__dict__",
        "_locked",
        "apertures",
        "options",
        "snr",
        "wavelength",
    )

    def __init__(
        self,
        *,
        snr: np.ndarray,
        wavelength: u.Quantity,
        options: ResultOptions,
        apertures: tuple[ApertureResult, ...],
        models: Models | None = None,
        signals: Signals | None = None,
        variances: Variances | None = None,
        data: u.Quantity | None = None,
        psf: np.ndarray | None = None,
        psf_pixel_scale: u.Quantity | None = None,
    ) -> None:
        object.__setattr__(self, "_locked", False)
        self.snr = _readonly_array(snr)
        self.wavelength = _readonly_quantity(wavelength)
        self.options = readonly_options(options)
        self.apertures = apertures
        if models is not None:
            self.models = models
        if signals is not None:
            self.signals = signals
        if variances is not None:
            self.variances = variances
        if data is not None:
            self.data = _readonly_quantity(data)
        if (psf is None) != (psf_pixel_scale is None):
            raise ValueError("PSF data and pixel scale must be provided together.")
        if psf is not None:
            self.psf = _readonly_array(psf)
            self.psf_pixel_scale = _readonly_quantity(psf_pixel_scale)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("ETC results are immutable.")
        object.__setattr__(self, name, value)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "snr": self.snr,
            "wavelength": self.wavelength,
            "options": self.options,
            "apertures": self.apertures,
            **self.__dict__,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "snr", _readonly_array(state.pop("snr")))
        object.__setattr__(
            self,
            "wavelength",
            _readonly_quantity(state.pop("wavelength")),
        )
        object.__setattr__(self, "options", readonly_options(state.pop("options")))
        object.__setattr__(self, "apertures", state.pop("apertures"))
        if "models" in state:
            object.__setattr__(self, "models", readonly_models(state.pop("models")))
        if "signals" in state:
            object.__setattr__(
                self,
                "signals",
                readonly_signals(state.pop("signals")),
            )
        if "variances" in state:
            object.__setattr__(
                self,
                "variances",
                readonly_variances(state.pop("variances")),
            )
        if "data" in state:
            object.__setattr__(
                self,
                "data",
                _readonly_quantity(state.pop("data")),
            )
        if "psf" in state:
            object.__setattr__(self, "psf", _readonly_array(state.pop("psf")))
            object.__setattr__(
                self,
                "psf_pixel_scale",
                _readonly_quantity(state.pop("psf_pixel_scale")),
            )
        object.__setattr__(self, "_locked", True)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the complete result as pickle or portable datacubes as FITS."""

        path = Path(path).expanduser()
        if path.exists() and not overwrite:
            raise FileExistsError(f"Result file already exists: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pkl":
            with path.open("wb") as stream:
                pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
            return
        if suffix in {".fit", ".fits", ".fts"}:
            fits.HDUList(_fits_hdus(self)).writeto(
                path, overwrite=overwrite, checksum=True
            )
            return
        raise ValueError("Result files must use PKL or FITS format.")


def readonly_models(models: Models) -> Models:
    targets = tuple(
        TargetModels(
            high=_readonly_model_grid(target.high),
            low=_readonly_model_grid(target.low),
        )
        for target in models.targets
    )
    return Models(
        targets=targets,
        combined=_readonly_quantity(models.combined),
        transmission=_readonly_array(models.transmission),
        sky=_readonly_quantity(models.sky),
        thermal=_readonly_quantity(models.thermal),
    )


def readonly_signals(signals: Signals) -> Signals:
    return Signals(
        **{
            field: _readonly_quantity(getattr(signals, field))
            for field in Signals.__dataclass_fields__
        }
    )


def readonly_variances(variances: Variances) -> Variances:
    return Variances(
        **{
            field: _readonly_quantity(getattr(variances, field))
            for field in Variances.__dataclass_fields__
        }
    )


def readonly_aperture_projection(
    projection: ApertureProjection,
) -> ApertureProjection:
    return ApertureProjection(
        snr=projection.snr,
        signals=getattr(projection, "signals", None),
        variances=getattr(projection, "variances", None),
    )


def readonly_options(options: ResultOptions) -> ResultOptions:
    targets = copy.deepcopy(options.targets)
    for target in targets:
        _freeze_target(target)
    pointing_center = copy.deepcopy(options.pointing_center)
    exposure = ExposureOptions(
        time=_readonly_quantity(options.exposure.time),
        n=options.exposure.n,
        n_target=options.exposure.n_target,
        n_sky=options.exposure.n_sky,
        target_time=_readonly_quantity(options.exposure.target_time),
        sky_time=_readonly_quantity(options.exposure.sky_time),
        total_time=_readonly_quantity(options.exposure.total_time),
    )
    sky_mask = options.sky_subtraction.mask
    sky_subtraction = SkySubtractionOptions(
        method=options.sky_subtraction.method,
        sequence=options.sky_subtraction.sequence,
        mask=_readonly_array(sky_mask) if sky_mask is not None else None,
    )
    return ResultOptions(
        instrument=options.instrument,
        scale=options.scale,
        spaxel_scale=_readonly_quantity(options.spaxel_scale),
        disperser=options.disperser,
        atmosphere=options.atmosphere,
        position_angle=_readonly_quantity(options.position_angle),
        pointing_center=pointing_center,
        targets=targets,
        psf_pixel_scale=(
            _readonly_quantity(options.psf_pixel_scale)
            if options.psf_pixel_scale is not None
            else None
        ),
        psf_path=options.psf_path,
        exposure=exposure,
        sky_subtraction=sky_subtraction,
        n_cubes=options.n_cubes,
    )


def _readonly_model_grid(grid: ModelGrid) -> ModelGrid:
    return ModelGrid(
        wavelength=_readonly_quantity(grid.wavelength),
        spatial=_readonly_array(grid.spatial),
        spectrum=_readonly_quantity(grid.spectrum),
        velocity=(
            _readonly_quantity(grid.velocity) if grid.velocity is not None else None
        ),
        spatial_convolved=(
            _readonly_array(grid.spatial_convolved)
            if grid.spatial_convolved is not None
            else None
        ),
        spectrum_convolved=(
            _readonly_quantity(grid.spectrum_convolved)
            if grid.spectrum_convolved is not None
            else None
        ),
    )


def _readonly_array(values: Any) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _readonly_quantity(values: u.Quantity) -> u.Quantity:
    quantity = u.Quantity(values, copy=True)
    quantity.setflags(write=False)
    return quantity


def _freeze_target(target: Any) -> None:
    for value in target.position:
        value.setflags(write=False)
    spatial = target.spatial
    for name in ("effective_radius", "fwhm", "position_angle"):
        value = getattr(spatial, name, None)
        if isinstance(value, u.Quantity):
            value.setflags(write=False)
    data = getattr(spatial, "data", None)
    if isinstance(data, np.ndarray):
        data.setflags(write=False)
    velocity = target.velocity
    if velocity is not None:
        for name in (
            "offset",
            "maximum_velocity",
            "turnover_radius",
            "inclination",
            "position_angle",
            "systemic_velocity",
            "data",
            "pixel_scale",
        ):
            value = getattr(velocity, name, None)
            if isinstance(value, u.Quantity):
                value.setflags(write=False)
    spectrum = target.spectrum
    for name in (
        "wavelength",
        "rest_wavelength",
        "flux",
        "dispersion",
        "fwhm",
        "continuum",
    ):
        value = getattr(spectrum, name, None)
        if isinstance(value, u.Quantity):
            value.setflags(write=False)


def _fits_hdus(result: EtcResult) -> list[Any]:
    header = _fits_metadata(result)
    hdus: list[Any] = [fits.PrimaryHDU(header=header)]
    hdus.append(_image_hdu("WAVELEN", result.wavelength))
    hdus.append(
        _image_hdu(
            "SNR",
            result.snr * u.dimensionless_unscaled,
            header=_cube_wcs(result),
        )
    )
    if hasattr(result, "models"):
        hdus.append(
            _image_hdu("MODEL", result.models.combined, header=_cube_wcs(result))
        )
        hdus.extend(
            (
                _image_hdu(
                    "TRANSMIS",
                    result.models.transmission * u.dimensionless_unscaled,
                ),
                _image_hdu("SKYMODEL", result.models.sky),
                _image_hdu("THERMAL", result.models.thermal),
            )
        )
    if hasattr(result, "signals"):
        for name, field in (
            ("SIGTARG", "target"),
            ("SIGSKY", "sky"),
            ("SIGTHERM", "thermal"),
            ("SIGDARK", "dark"),
            ("SIGBKG", "background"),
            ("SIGTOTAL", "total"),
        ):
            hdus.append(
                _image_hdu(
                    name, getattr(result.signals, field), header=_cube_wcs(result)
                )
            )
    if hasattr(result, "variances"):
        for name, field in (
            ("VARTARG", "target"),
            ("VARSKY", "sky"),
            ("VARTHERM", "thermal"),
            ("VARDARK", "dark"),
            ("VARREAD", "read"),
            ("VARTOTAL", "total"),
        ):
            hdus.append(
                _image_hdu(
                    name,
                    getattr(result.variances, field),
                    header=_cube_wcs(result),
                )
            )
    if hasattr(result, "data"):
        hdus.append(_image_hdu("DATA", result.data, header=_data_wcs(result)))
    if hasattr(result, "psf"):
        psf_header = fits.Header(
            {"PIXSCALE": result.psf_pixel_scale.to_value(u.mas)}
        )
        hdus.append(
            _image_hdu(
                "PSF",
                result.psf * u.dimensionless_unscaled,
                header=psf_header,
            )
        )
    if result.options.sky_subtraction.mask is not None:
        hdus.append(
            fits.ImageHDU(
                data=result.options.sky_subtraction.mask.astype(np.uint8),
                name="SKYMASK",
            )
        )
    for index, aperture in enumerate(result.apertures):
        aperture_header = fits.Header({"APNAME": aperture.name})
        hdus.append(
            fits.ImageHDU(
                data=aperture.mask.astype(np.uint8),
                header=aperture_header,
                name=f"APMASK{index}",
            )
        )
        hdus.extend(_aperture_projection_hdus(result, aperture, index))
    if result.apertures:
        hdus.append(_aperture_table(result.apertures))
    return hdus


def _fits_metadata(result: EtcResult) -> fits.Header:
    options = result.options
    exposure = options.exposure
    header = fits.Header()
    header["INSTRUME"] = options.instrument
    header["SCALE"] = options.scale
    header["DISPERSE"] = options.disperser
    header["ATMOS"] = options.atmosphere
    header["POSANGLE"] = options.position_angle.to_value(u.deg)
    header["EXPTIME"] = exposure.time.to_value(u.s)
    header["NEXP"] = exposure.n
    header["NTARGET"] = exposure.n_target
    header["NSKY"] = exposure.n_sky
    header["TONTARG"] = exposure.target_time.to_value(u.s)
    header["TINT"] = exposure.total_time.to_value(u.s)
    header["SKYMETH"] = options.sky_subtraction.method
    header["NCUBES"] = options.n_cubes
    if options.sky_subtraction.sequence is not None:
        header["SKYSEQ"] = options.sky_subtraction.sequence
    if options.psf_pixel_scale is not None:
        header["PSFSCALE"] = options.psf_pixel_scale.to_value(u.mas)
    if options.psf_path is not None:
        header["PSFFILE"] = str(options.psf_path)
    return header


def _image_hdu(
    name: str,
    values: Any,
    *,
    header: fits.Header | None = None,
) -> fits.ImageHDU:
    image_header = fits.Header() if header is None else header.copy()
    if isinstance(values, u.Quantity):
        data = values.value
        image_header["BUNIT"] = _unit_string(values.unit)
    else:
        data = values
    return fits.ImageHDU(data=np.asarray(data), header=image_header, name=name)


def _cube_wcs(result: EtcResult) -> fits.Header:
    ny, nx, _ = result.snr.shape
    wavelength = result.wavelength.to_value(u.micron)
    pixel_scale = _result_pixel_scale(result).to_value(u.deg)
    angle = result.options.position_angle.to_value(u.rad)
    header = fits.Header()
    header["WCSAXES"] = 3
    header["CTYPE1"] = "WAVE"
    header["CUNIT1"] = "um"
    header["CRPIX1"] = 1.0
    header["CRVAL1"] = wavelength[0]
    header["CDELT1"] = wavelength[1] - wavelength[0]
    header["CRPIX2"] = (nx + 1) / 2
    header["CRPIX3"] = (ny + 1) / 2
    center = result.options.pointing_center
    if center is None:
        header["CTYPE2"] = "XOFFSET"
        header["CTYPE3"] = "YOFFSET"
        header["CUNIT2"] = "deg"
        header["CUNIT3"] = "deg"
        header["CRVAL2"] = 0.0
        header["CRVAL3"] = 0.0
    else:
        icrs = center.icrs
        header["CTYPE2"] = "RA---TAN"
        header["CTYPE3"] = "DEC--TAN"
        header["CUNIT2"] = "deg"
        header["CUNIT3"] = "deg"
        header["CRVAL2"] = icrs.ra.to_value(u.deg)
        header["CRVAL3"] = icrs.dec.to_value(u.deg)
    header["CD2_2"] = -pixel_scale * np.cos(angle)
    header["CD2_3"] = pixel_scale * np.sin(angle)
    header["CD3_2"] = pixel_scale * np.sin(angle)
    header["CD3_3"] = pixel_scale * np.cos(angle)
    return header


def _data_wcs(result: EtcResult) -> fits.Header:
    header = _cube_wcs(result)
    if result.data.ndim == 4:
        header["WCSAXES"] = 4
        header["CTYPE4"] = "CUBENUM"
        header["CRPIX4"] = 1.0
        header["CRVAL4"] = 1.0
        header["CDELT4"] = 1.0
    return header


def _result_pixel_scale(result: EtcResult) -> u.Quantity:
    return result.options.spaxel_scale


def _aperture_table(apertures: tuple[ApertureResult, ...]) -> fits.BinTableHDU:
    columns = [
        fits.Column(name="NAME", format="64A", array=[item.name for item in apertures]),
        fits.Column(name="SNR", format="D", array=[item.snr for item in apertures]),
    ]
    for group_name, prefix in (("signals", "SIG"), ("variances", "VAR")):
        if all(hasattr(item, group_name) for item in apertures):
            fields = type(getattr(apertures[0], group_name)).__dataclass_fields__
            for field in fields:
                values = [
                    getattr(getattr(item, group_name), field).value
                    for item in apertures
                ]
                unit = getattr(getattr(apertures[0], group_name), field).unit
                columns.append(
                    fits.Column(
                        name=f"{prefix}{field.upper()}",
                        format="D",
                        unit=_unit_string(unit),
                        array=values,
                    )
                )
    if all(hasattr(item, "data") for item in apertures):
        size = max(np.atleast_1d(item.data.value).size for item in apertures)
        values = np.vstack([np.atleast_1d(item.data.value) for item in apertures])
        columns.append(
            fits.Column(
                name="DATA",
                format=f"{size}D",
                unit=_unit_string(apertures[0].data.unit),
                array=values,
            )
        )
    return fits.BinTableHDU.from_columns(columns, name="APERTURE")


def _aperture_projection_hdus(
    result: EtcResult,
    aperture: ApertureResult,
    index: int,
) -> list[fits.ImageHDU]:
    hdus: list[fits.ImageHDU] = []
    for label, projection, wcs in (
        ("SPEC", aperture.spectra, _spectral_wcs(result)),
        ("MAP", aperture.maps, _spatial_wcs(result)),
    ):
        base_header = wcs.copy()
        base_header["APINDEX"] = index
        base_header["APNAME"] = aperture.name
        base_header["APVIEW"] = label
        hdus.append(
            _image_hdu(
                f"AP{index}{label}SNR",
                projection.snr * u.dimensionless_unscaled,
                header=base_header,
            )
        )
        for group_name, prefix in (("signals", "SIG"), ("variances", "VAR")):
            if not hasattr(projection, group_name):
                continue
            group = getattr(projection, group_name)
            for field in type(group).__dataclass_fields__:
                field_header = base_header.copy()
                field_header["APFIELD"] = field
                hdus.append(
                    _image_hdu(
                        f"AP{index}{label}{prefix}{field.upper()}",
                        getattr(group, field),
                        header=field_header,
                    )
                )
    return hdus


def _spectral_wcs(result: EtcResult) -> fits.Header:
    wavelength = result.wavelength.to_value(u.micron)
    header = fits.Header()
    header["WCSAXES"] = 1
    header["CTYPE1"] = "WAVE"
    header["CUNIT1"] = "um"
    header["CRPIX1"] = 1.0
    header["CRVAL1"] = wavelength[0]
    header["CDELT1"] = wavelength[1] - wavelength[0]
    return header


def _spatial_wcs(result: EtcResult) -> fits.Header:
    cube = _cube_wcs(result)
    header = fits.Header()
    header["WCSAXES"] = 2
    for target, source in (
        ("CTYPE1", "CTYPE2"),
        ("CTYPE2", "CTYPE3"),
        ("CUNIT1", "CUNIT2"),
        ("CUNIT2", "CUNIT3"),
        ("CRPIX1", "CRPIX2"),
        ("CRPIX2", "CRPIX3"),
        ("CRVAL1", "CRVAL2"),
        ("CRVAL2", "CRVAL3"),
        ("CD1_1", "CD2_2"),
        ("CD1_2", "CD2_3"),
        ("CD2_1", "CD3_2"),
        ("CD2_2", "CD3_3"),
    ):
        header[target] = cube[source]
    return header


def _unit_string(unit: u.UnitBase) -> str:
    if unit == u.dimensionless_unscaled:
        return "1"
    try:
        return unit.to_string("fits")
    except ValueError:
        return unit.to_string()
