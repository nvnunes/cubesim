"""Immutable result structures for forward ETC calculations."""

from __future__ import annotations

import copy
import pickle
import secrets
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from cubesim._variance import in_field_aperture_variance_spectrum


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
    total: u.Quantity


@dataclass(frozen=True, slots=True)
class Variances:
    target: u.Quantity
    sky: u.Quantity
    thermal: u.Quantity
    dark: u.Quantity
    read: u.Quantity
    total: u.Quantity


@dataclass(frozen=True, slots=True)
class _ApertureSamplingState:
    science_mean: np.ndarray
    sky_mean: np.ndarray
    sky_weight: np.ndarray
    science_read_variance: np.ndarray
    sky_read_variance: np.ndarray


@dataclass(frozen=True, slots=True)
class SampledCube:
    """Immutable noisy IFU cube realizations.

    Attributes:
        data: Electron data with shape ``(y, x, wavelength)`` for one
            realization or ``(n, y, x, wavelength)`` for multiple realizations.
        wavelength: Detector wavelength coordinate.
        seed: Random seed used to draw the realizations.
        options: Resolved calculation options retained for metadata and WCS.
    """

    data: u.Quantity
    wavelength: u.Quantity
    seed: int
    options: ResultOptions

    def __post_init__(self) -> None:
        if self.data.ndim not in {3, 4}:
            raise ValueError("Sampled cube data must be three- or four-dimensional.")
        if self.data.shape[-1] != len(self.wavelength):
            raise ValueError("Sampled cube data must match the wavelength coordinate.")
        if not self.data.unit.is_equivalent(u.electron):
            raise u.UnitConversionError("Sampled cube data must have electron units.")
        object.__setattr__(self, "data", _readonly_quantity(self.data))
        object.__setattr__(self, "wavelength", _readonly_quantity(self.wavelength))

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the sampled cube data, wavelength, metadata, and WCS as FITS."""

        _save_sample(path, _sampled_cube_hdus(self), overwrite=overwrite)


@dataclass(frozen=True, slots=True)
class SampledAperture:
    """Immutable noisy realizations integrated over one aperture.

    Attributes:
        data: Scalar electron data for one realization or one-dimensional
            electron data for multiple realizations.
        wavelength: Detector wavelength coordinate associated with the mask.
        seed: Random seed used to draw the realizations.
        name: Registered aperture name.
        mask: Three-dimensional Boolean aperture mask.
    """

    data: u.Quantity
    wavelength: u.Quantity
    seed: int
    name: str
    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim > 1:
            raise ValueError("Sampled aperture data must be scalar or one-dimensional.")
        if self.mask.ndim != 3 or self.mask.dtype.kind != "b":
            raise ValueError(
                "A sampled aperture mask must be a three-dimensional Boolean array."
            )
        if self.mask.shape[-1] != len(self.wavelength):
            raise ValueError("A sampled aperture mask must match the wavelength coordinate.")
        if not self.data.unit.is_equivalent(u.electron):
            raise u.UnitConversionError("Sampled aperture data must have electron units.")
        object.__setattr__(self, "data", _readonly_quantity(self.data))
        object.__setattr__(self, "wavelength", _readonly_quantity(self.wavelength))
        object.__setattr__(self, "mask", _readonly_array(self.mask))

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the aperture samples and aperture definition as FITS."""

        _save_sample(path, _sampled_aperture_hdus(self), overwrite=overwrite)


class ApertureProjection:
    """One immutable aperture reduction retaining wavelength or position."""

    __slots__ = ("__dict__", "_locked", "snr")

    def __init__(
        self,
        *,
        snr: np.ndarray,
        signals: Signals,
        variances: Variances,
    ) -> None:
        object.__setattr__(self, "_locked", False)
        self.snr = _readonly_array(snr)
        self.signals = readonly_signals(signals)
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
        object.__setattr__(
            self,
            "signals",
            readonly_signals(state.pop("signals")),
        )
        object.__setattr__(
            self,
            "variances",
            readonly_variances(state.pop("variances")),
        )
        object.__setattr__(self, "_locked", True)


class ApertureResult:
    """One immutable registered-aperture reduction."""

    __slots__ = (
        "__dict__",
        "_locked",
        "_wavelength",
        "maps",
        "mask",
        "name",
        "snr",
        "spectra",
    )

    def __init__(
        self,
        *,
        name: str,
        mask: np.ndarray,
        snr: float,
        spectra: ApertureProjection,
        maps: ApertureProjection,
        signals: Signals,
        variances: Variances,
        sampling: _ApertureSamplingState,
        wavelength: u.Quantity,
    ) -> None:
        object.__setattr__(self, "_locked", False)
        self.name = name
        self.mask = _readonly_array(mask)
        self.snr = float(snr)
        self.spectra = readonly_aperture_projection(spectra)
        self.maps = readonly_aperture_projection(maps)
        self.signals = readonly_signals(signals)
        self.variances = readonly_variances(variances)
        self._sampling = _readonly_aperture_sampling(sampling)
        self._wavelength = _readonly_quantity(wavelength)
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
            "_wavelength": self._wavelength,
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
        object.__setattr__(
            self,
            "signals",
            readonly_signals(state.pop("signals")),
        )
        object.__setattr__(
            self,
            "variances",
            readonly_variances(state.pop("variances")),
        )
        object.__setattr__(
            self,
            "_sampling",
            _readonly_aperture_sampling(state.pop("_sampling")),
        )
        object.__setattr__(
            self,
            "_wavelength",
            _readonly_quantity(state.pop("_wavelength")),
        )
        object.__setattr__(self, "_locked", True)

    def sample(
        self,
        n: int = 1,
        *,
        seed: int | None = None,
    ) -> SampledAperture:
        """Draw integrated noisy realizations for this aperture.

        The sample data is a scalar electron quantity when ``n=1`` and a
        one-dimensional quantity with one value per realization otherwise.
        When ``seed`` is omitted, a seed is generated and retained by the
        returned sample.
        """

        seed = _resolve_sample_request(n, seed)
        from cubesim._calculation import _sample_aperture_data

        data = _sample_aperture_data(
            self._sampling,
            n,
            np.random.default_rng(seed),
        )
        return SampledAperture(
            data=data,
            wavelength=self._wavelength,
            seed=seed,
            name=self.name,
            mask=self.mask,
        )


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
        signals: Signals,
        variances: Variances,
        read_noise: u.Quantity,
        models: Models | None = None,
        psf: np.ndarray | None = None,
        psf_pixel_scale: u.Quantity | None = None,
    ) -> None:
        object.__setattr__(self, "_locked", False)
        self.snr = _readonly_array(snr)
        self.wavelength = _readonly_quantity(wavelength)
        self.options = readonly_options(options)
        self.apertures = apertures
        self.signals = readonly_signals(signals)
        self.variances = readonly_variances(variances)
        self._read_noise = _readonly_quantity(read_noise)
        if models is not None:
            self.models = readonly_models(models)
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
        object.__setattr__(
            self,
            "signals",
            readonly_signals(state.pop("signals")),
        )
        object.__setattr__(
            self,
            "variances",
            readonly_variances(state.pop("variances")),
        )
        object.__setattr__(
            self,
            "_read_noise",
            _readonly_quantity(state.pop("_read_noise")),
        )
        if "models" in state:
            object.__setattr__(self, "models", readonly_models(state.pop("models")))
        if "psf" in state:
            object.__setattr__(self, "psf", _readonly_array(state.pop("psf")))
            object.__setattr__(
                self,
                "psf_pixel_scale",
                _readonly_quantity(state.pop("psf_pixel_scale")),
            )
        object.__setattr__(self, "_locked", True)

    def sample(
        self,
        n: int = 1,
        *,
        seed: int | None = None,
    ) -> SampledCube:
        """Draw noisy, sky-subtracted IFU data realizations.

        The sample data has shape ``(y, x, wavelength)`` when ``n=1``.
        Multiple realizations add a leading axis. When ``seed`` is omitted, a
        seed is generated and retained by the returned sample.
        """

        seed = _resolve_sample_request(n, seed)
        from cubesim._calculation import _sample_data

        data = _sample_data(
            self.signals,
            self.options.exposure,
            self._read_noise,
            self.options.sky_subtraction,
            n,
            np.random.default_rng(seed),
        )
        return SampledCube(
            data=data,
            wavelength=self.wavelength,
            seed=seed,
            options=self.options,
        )

    def _snr_spectrum_for_spatial_mask(
        self,
        spatial_mask: np.ndarray,
    ) -> np.ndarray:
        """Reduce the S/N spectrum over a spatial detector mask."""

        spatial_mask = np.asarray(spatial_mask)
        if (
            spatial_mask.shape != self.snr.shape[:2]
            or spatial_mask.dtype.kind != "b"
        ):
            raise ValueError("Spatial S/N reduction requires a 2D Boolean IFU mask.")
        if self.options.sky_subtraction.method == "in_field":
            sky_mask = self.options.sky_subtraction.mask
            raw_read_variance = np.full(
                self.snr.shape,
                self.options.exposure.n_target
                * self._read_noise.to_value(u.electron) ** 2,
            )
            raw_variance = self.signals.total.value + raw_read_variance
            aperture_mask = np.broadcast_to(
                spatial_mask[:, :, None],
                self.snr.shape,
            )
            variance = in_field_aperture_variance_spectrum(
                aperture_mask,
                sky_mask,
                raw_variance,
            )
        else:
            variance = self.variances.total[spatial_mask].sum(axis=0).value
        signal = self.signals.target[spatial_mask].sum(axis=0).value
        return _snr_values(signal, variance)

    def _snr_map_for_wavelength_indices(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Reduce the S/N map over detector wavelength indices."""

        signal = self.signals.target[:, :, indices].sum(axis=2).value
        variance = self.variances.total[:, :, indices].sum(axis=2).value
        return _snr_values(signal, variance)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the complete result as pickle or key science cubes as FITS."""

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
        signals=projection.signals,
        variances=projection.variances,
    )


def readonly_options(options: ResultOptions) -> ResultOptions:
    targets = copy.deepcopy(options.targets)
    _freeze_nested_data(targets)
    pointing_center = copy.deepcopy(options.pointing_center)
    _freeze_nested_data(pointing_center)
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
    )


def _readonly_aperture_sampling(
    sampling: _ApertureSamplingState,
) -> _ApertureSamplingState:
    return _ApertureSamplingState(
        science_mean=_readonly_array(sampling.science_mean),
        sky_mean=_readonly_array(sampling.sky_mean),
        sky_weight=_readonly_array(sampling.sky_weight),
        science_read_variance=_readonly_array(sampling.science_read_variance),
        sky_read_variance=_readonly_array(sampling.sky_read_variance),
    )


def _resolve_sample_request(
    n: int,
    seed: int | None,
) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if seed is None:
        return secrets.randbits(63)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be a non-negative integer.")
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.int64).max:
        raise ValueError("seed must be between 0 and 2**63 - 1.")
    return seed


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


def _freeze_nested_data(value: Any) -> None:
    if isinstance(value, u.Quantity):
        value.setflags(write=False)
        return
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return
    if isinstance(value, SkyCoord):
        for component in value.data.components:
            _freeze_nested_data(getattr(value.data, component))
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _freeze_nested_data(getattr(value, field.name))
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _freeze_nested_data(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _freeze_nested_data(item)


def _fits_hdus(result: EtcResult) -> list[Any]:
    header = _fits_metadata(result)
    cube_header = _cube_wcs(result)
    background = (
        result.signals.sky + result.signals.thermal + result.signals.dark
    )
    return [
        fits.PrimaryHDU(header=header),
        _image_hdu("WAVELEN", result.wavelength),
        _image_hdu(
            "SNR",
            result.snr * u.dimensionless_unscaled,
            header=cube_header,
        ),
        _image_hdu("SIGNAL", result.signals.target, header=cube_header),
        _image_hdu("BACKGROUND", background, header=cube_header),
        _image_hdu("VARIANCE", result.variances.total, header=cube_header),
    ]


def _sampled_cube_hdus(sample: SampledCube) -> list[Any]:
    header = _fits_metadata_from_options(sample.options)
    header["PRODUCT"] = "SAMPLED CUBE"
    header["RNGSEED"] = sample.seed
    header["NREAL"] = 1 if sample.data.ndim == 3 else sample.data.shape[0]
    cube_header = _cube_wcs_values(
        sample.wavelength,
        sample.data.shape[-3:],
        sample.options,
    )
    if sample.data.ndim == 4:
        cube_header["WCSAXES"] = 4
        cube_header["CTYPE4"] = "REALIZATION"
        cube_header["CRPIX4"] = 1.0
        cube_header["CRVAL4"] = 1.0
        cube_header["CD4_4"] = 1.0
    return [
        fits.PrimaryHDU(header=header),
        _image_hdu("WAVELEN", sample.wavelength),
        _image_hdu("DATA", sample.data, header=cube_header),
    ]


def _sampled_aperture_hdus(sample: SampledAperture) -> list[Any]:
    header = fits.Header()
    header["PRODUCT"] = "SAMPLED APERTURE"
    header["APERTURE"] = sample.name
    header["RNGSEED"] = sample.seed
    header["NREAL"] = 1 if sample.data.isscalar else len(sample.data)
    return [
        fits.PrimaryHDU(header=header),
        _image_hdu("WAVELEN", sample.wavelength),
        _image_hdu("DATA", np.atleast_1d(sample.data.value) * sample.data.unit),
        _image_hdu("MASK", sample.mask.astype(np.uint8)),
    ]


def _save_sample(
    path: str | Path,
    hdus: list[Any],
    *,
    overwrite: bool,
) -> None:
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Sample file already exists: {path}")
    if path.suffix.lower() not in {".fit", ".fits", ".fts"}:
        raise ValueError("Sample files must use FITS format.")
    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)


def _fits_metadata(result: EtcResult) -> fits.Header:
    return _fits_metadata_from_options(result.options)


def _fits_metadata_from_options(options: ResultOptions) -> fits.Header:
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
    if options.sky_subtraction.sequence is not None:
        header["SKYSEQ"] = options.sky_subtraction.sequence
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
    return _cube_wcs_values(result.wavelength, result.snr.shape, result.options)


def _cube_wcs_values(
    wavelength: u.Quantity,
    shape: tuple[int, int, int],
    options: ResultOptions,
) -> fits.Header:
    ny, nx, _ = shape
    wavelength = wavelength.to_value(u.micron)
    pixel_scale = options.spaxel_scale.to_value(u.deg)
    angle = options.position_angle.to_value(u.rad)
    header = fits.Header()
    header["WCSAXES"] = 3
    header["CTYPE1"] = "WAVE"
    header["CUNIT1"] = "um"
    header["CRPIX1"] = 1.0
    header["CRVAL1"] = wavelength[0]
    header["CD1_1"] = wavelength[1] - wavelength[0]
    header["CRPIX2"] = (nx + 1) / 2
    header["CRPIX3"] = (ny + 1) / 2
    center = options.pointing_center
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


def _snr_values(signal: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return np.divide(
        signal,
        np.sqrt(variance),
        out=np.zeros_like(signal, dtype=float),
        where=variance > 0,
    )


def _unit_string(unit: u.UnitBase) -> str:
    if unit == u.dimensionless_unscaled:
        return "1"
    try:
        return unit.to_string("fits")
    except ValueError:
        return unit.to_string()
