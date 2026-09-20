"""Stateful public entrypoint for cubesim calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from cubesim._calculation import calculate
from cubesim._instrument import InstrumentSelection, load_instrument
from cubesim._psf import Psf, load_psf
from cubesim._result import (
    EtcResult,
    ExposureOptions,
    ResultOptions,
    SkySubtractionOptions,
    readonly_models,
    readonly_signals,
    readonly_variances,
)
from cubesim.models import (
    ConstantVelocity,
    Gaussian,
    GaussianLines,
    Point,
    RotatingDisk,
    Sersic,
    SpatialImage,
    TabulatedSpectrum,
    Uniform,
    VelocityField,
)


@dataclass(frozen=True, slots=True)
class _Target:
    position: tuple[u.Quantity, u.Quantity]
    spatial: Point | Gaussian | Sersic | SpatialImage | Uniform
    spectrum: GaussianLines | TabulatedSpectrum
    velocity: ConstantVelocity | RotatingDisk | VelocityField | None


@dataclass(frozen=True, slots=True)
class _ExposureRequest:
    time: u.Quantity
    n: int | None
    n_target: int | None


@dataclass(frozen=True, slots=True)
class _SkySubtraction:
    method: str
    sequence: str | None
    mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _Aperture:
    name: str
    size: tuple[int, int, int] | None
    center: tuple[Any, Any, Any] | None
    start: tuple[Any, Any, Any] | None
    mask: np.ndarray | None


class Etc:
    """Configure and execute calculations for one instrument-data directory.

    Args:
        instrument_data: Directory containing ``etc.ini`` and the scientific
            files it references. Relative paths are resolved from this
            directory.

    Notes:
        The object is mutable during configuration. Each call to :meth:`run`
        returns an independent immutable result.
    """

    def __init__(self, instrument_data: str | Path) -> None:
        self._instrument = load_instrument(instrument_data)
        self._selection: InstrumentSelection | None = None
        self._psf: Psf | None = None
        self._targets: list[_Target] = []
        self._exposure: _ExposureRequest | None = None
        self._sky_subtraction = _SkySubtraction(
            method="nodding",
            sequence="AB",
            mask=None,
        )
        self._apertures: list[_Aperture] = []
        self._position_angle = 0 * u.deg
        self._pointing_center: Any | None = None

    def configure(
        self,
        *,
        scale: str,
        disperser: str,
        atmosphere: str,
    ) -> None:
        """Select an instrument configuration.

        Args:
            scale: Exact name of a configured detector scale.
            disperser: Exact name of a selectable disperser leaf.
            atmosphere: Exact name of a configured atmosphere.

        Calling this method again replaces the previous instrument selection.
        """

        self._selection = self._instrument.select(
            scale=scale,
            disperser=disperser,
            atmosphere=atmosphere,
        )

    def set_pointing(
        self,
        *,
        position_angle: u.Quantity = 0 * u.deg,
        center: Any | None = None,
    ) -> None:
        """Set IFU orientation and an optional absolute field center.

        Args:
            position_angle: Scalar angular position angle measured east of
                north.
            center: Optional scalar :class:`astropy.coordinates.SkyCoord` at
                the field center. FITS output uses celestial WCS when supplied
                and angular offsets otherwise.
        """

        self._position_angle = _angle(position_angle, "position_angle")
        if center is not None and (
            not isinstance(center, SkyCoord) or not center.isscalar
        ):
            raise TypeError("center must be a scalar astropy.coordinates.SkyCoord.")
        self._pointing_center = center

    def add_target(
        self,
        *,
        position: tuple[u.Quantity, u.Quantity],
        spatial: Point | Gaussian | Sersic | SpatialImage | Uniform,
        spectrum: GaussianLines | TabulatedSpectrum,
        velocity: ConstantVelocity | RotatingDisk | VelocityField | None = None,
    ) -> None:
        """Append one explicitly composed target to the calculation.

        Args:
            position: ``(east, north)`` angular offset from the pointing
                center.
            spatial: Spatial-profile model. ``Uniform`` requires a spectrum
                in surface-brightness units; all other profiles require
                integrated flux.
            spectrum: Gaussian-line or tabulated spectral model.
            velocity: Optional constant or spatially varying line-of-sight
                velocity model.
        """

        if not isinstance(spatial, (Point, Gaussian, Sersic, SpatialImage, Uniform)):
            raise TypeError("spatial must be a cubesim spatial model.")
        if not isinstance(spectrum, (GaussianLines, TabulatedSpectrum)):
            raise TypeError("spectrum must be a cubesim spectral model.")
        if velocity is not None and not isinstance(
            velocity,
            (ConstantVelocity, RotatingDisk, VelocityField),
        ):
            raise TypeError("velocity must be a cubesim velocity model or None.")
        if not isinstance(position, tuple) or len(position) != 2:
            raise TypeError("position must be an (east, north) tuple.")
        resolved_position = (
            _angle(position[0], "position east"),
            _angle(position[1], "position north"),
        )
        surface_brightness = spectrum.flux.unit.is_equivalent(
            u.erg / (u.s * u.cm**2 * u.arcsec**2)
        ) or spectrum.flux.unit.is_equivalent(
            u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.m)
        )
        if isinstance(spatial, Uniform) != surface_brightness:
            raise u.UnitConversionError(
                "Uniform targets require surface-brightness flux; other spatial "
                "models require integrated flux."
            )
        self._targets.append(
            _Target(
                position=resolved_position,
                spatial=spatial,
                spectrum=spectrum,
                velocity=velocity,
            )
        )

    def set_psf(
        self,
        psf: str | Path | Any,
        *,
        pixel_scale: Any | None = None,
    ) -> None:
        """Set the achromatic PSF used by subsequent calculations.

        Args:
            psf: FITS filename, NPY filename, or in-memory two-dimensional
                array. Relative paths are resolved against the instrument-data
                directory.
            pixel_scale: Positive angular pixel scale required for NPY and
                in-memory inputs. FITS inputs instead read ``PIXSCALE`` in
                milliarcseconds per pixel.

        Calling this method again replaces the previous PSF.
        """

        self._psf = load_psf(
            psf,
            pixel_scale=pixel_scale,
            instrument_root=self._instrument.root,
        )

    def set_exposure(
        self,
        *,
        time: u.Quantity,
        n: int | None = None,
        n_target: int | None = None,
    ) -> None:
        """Set the single-frame time and one exposure count.

        Args:
            time: Positive duration of one exposure.
            n: Total number of frames. For nodding, this must contain a whole
                number of configured sequences.
            n_target: Number of target frames. For nodding, this must imply a
                whole number of configured sequences.

        Exactly one of ``n`` and ``n_target`` is required. For in-field sky
        subtraction, all frames are target frames and the two count forms are
        equivalent.
        """

        if (n is None) == (n_target is None):
            raise ValueError("Provide exactly one of n and n_target.")
        resolved_time = _positive_time(time)
        self._exposure = _ExposureRequest(
            time=resolved_time,
            n=_positive_integer(n, "n") if n is not None else None,
            n_target=(
                _positive_integer(n_target, "n_target")
                if n_target is not None
                else None
            ),
        )

    def set_sky_subtraction(
        self,
        *,
        method: str,
        sequence: str | None = None,
        mask: Any | None = None,
    ) -> None:
        """Configure nodding or in-field sky subtraction.

        Args:
            method: ``"nodding"`` or ``"in_field"``.
            sequence: Nodding sequence made from ``A`` target and ``B`` sky
                frames. The default for nodding is ``"AB"``.
            mask: Nonempty two-dimensional Boolean detector mask for in-field
                subtraction. ``True`` spaxels define the sky sample.
        """

        if method == "nodding":
            sequence = "AB" if sequence is None else sequence.upper()
            if set(sequence) != {"A", "B"} or not sequence:
                raise ValueError("A nodding sequence must contain only A and B frames.")
            if mask is not None:
                raise ValueError("Nodding sky subtraction does not accept a mask.")
            self._sky_subtraction = _SkySubtraction(method, sequence, None)
            return
        if method == "in_field":
            if sequence is not None:
                raise ValueError("In-field sky subtraction does not accept a sequence.")
            if mask is None:
                raise ValueError("In-field sky subtraction requires a mask.")
            array = np.asarray(mask)
            if array.ndim != 2 or array.dtype != np.dtype(bool) or not array.any():
                raise ValueError(
                    "In-field sky mask must be a nonempty 2D Boolean array."
                )
            self._sky_subtraction = _SkySubtraction(method, None, array.copy())
            return
        raise ValueError("method must be 'nodding' or 'in_field'.")

    def add_aperture(
        self,
        *,
        name: str,
        size: tuple[int, int, int] | None = None,
        center: tuple[Any, Any, Any] | None = None,
        start: tuple[Any, Any, Any] | None = None,
        mask: Any | None = None,
    ) -> None:
        """Register a rectangular or custom three-dimensional aperture.

        Args:
            name: Nonempty name, unique within this ETC.
            size: Positive integer ``(y, x, wavelength)`` size for a
                rectangular aperture.
            center: Optional rectangle center. Spatial coordinates are detector
                pixels; the spectral coordinate may be a pixel or scalar
                wavelength.
            start: Optional rectangle start coordinate using the same coordinate
                forms as ``center``.
            mask: Boolean mask with the exact output shape
                ``(y, x, wavelength)``.

        A mask cannot be combined with rectangle arguments. ``center`` and
        ``start`` are mutually exclusive. Bounds are validated during
        :meth:`run`, after the wavelength grid is known. Without either
        placement, the aperture uses the central detector pixel spatially and
        the first wavelength of the first target's spectrum spectrally.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("Aperture name must be a nonempty string.")
        if any(aperture.name == name for aperture in self._apertures):
            raise ValueError(f"Duplicate aperture name: {name!r}.")
        if mask is not None:
            if any(value is not None for value in (size, center, start)):
                raise ValueError("mask cannot be combined with rectangle arguments.")
            array = np.asarray(mask)
            if array.ndim != 3 or array.dtype != np.dtype(bool):
                raise ValueError(
                    "Aperture mask must be a three-dimensional Boolean array."
                )
            self._apertures.append(_Aperture(name, None, None, None, array.copy()))
            return
        if size is None:
            raise ValueError("Rectangular apertures require size.")
        if center is not None and start is not None:
            raise ValueError("Provide at most one of center and start.")
        resolved_size = tuple(
            _positive_integer(value, "aperture size") for value in size
        )
        if len(resolved_size) != 3:
            raise ValueError("Aperture size must contain (y, x, wavelength).")
        for placement, label in ((center, "center"), (start, "start")):
            if placement is not None and len(placement) != 3:
                raise ValueError(f"Aperture {label} must contain three coordinates.")
        self._apertures.append(_Aperture(name, resolved_size, center, start, None))

    def run(
        self,
        *,
        include_models: bool = False,
        include_signals: bool = False,
        include_variances: bool = False,
        include_data: bool = False,
        n_cubes: int = 1,
        rng: Any | None = None,
    ) -> EtcResult:
        """Run the configured forward ETC calculation.

        Args:
            include_models: Include per-target high- and low-resolution model
                components, their combined detector-resolution cube, and the
                wavelength-grid transmission, sky, and thermal models.
            include_signals: Include target, background-component, and total
                detected electron cubes.
            include_variances: Include detector variance-component cubes.
            include_data: Include random noisy, sky-subtracted detector cubes.
            n_cubes: Number of noisy realizations. Values other than one
                require ``include_data=True``.
            rng: Optional NumPy random generator. Valid only when data are
                requested.

        Returns:
            An immutable result containing S/N, wavelength, resolved options,
            aperture reductions, and any requested optional groups.
        """

        selection, psf, exposure = self._validate_and_resolve(
            include_data,
            n_cubes,
            rng,
        )
        output = calculate(
            instrument=self._instrument,
            selection=selection,
            targets=tuple(self._targets),
            psf=psf,
            exposure=exposure,
            sky_subtraction=self._sky_subtraction,
            apertures=tuple(self._apertures),
            position_angle=self._position_angle,
            include_signals=include_signals,
            include_variances=include_variances,
            include_data=include_data,
            n_cubes=n_cubes,
            rng=rng,
        )
        sky_mask = self._sky_subtraction.mask
        if sky_mask is not None:
            sky_mask = sky_mask.copy()
            sky_mask.setflags(write=False)
        options = ResultOptions(
            instrument=self._instrument.name,
            scale=selection.scale.name,
            spaxel_scale=selection.scale.spaxel_scale,
            disperser=selection.disperser.name,
            atmosphere=selection.atmosphere.name,
            position_angle=self._position_angle,
            pointing_center=self._pointing_center,
            targets=tuple(self._targets),
            psf_pixel_scale=psf.pixel_scale if psf is not None else None,
            psf_path=psf.path if psf is not None else None,
            exposure=exposure,
            sky_subtraction=SkySubtractionOptions(
                method=self._sky_subtraction.method,
                sequence=self._sky_subtraction.sequence,
                mask=sky_mask,
            ),
            n_cubes=n_cubes,
        )
        return EtcResult(
            snr=output.snr,
            wavelength=output.grid.wavelength,
            options=options,
            apertures=output.apertures,
            models=readonly_models(output.models) if include_models else None,
            signals=readonly_signals(output.signals) if include_signals else None,
            variances=(
                readonly_variances(output.variances) if include_variances else None
            ),
            data=output.data if include_data else None,
        )

    def _validate_and_resolve(
        self,
        include_data: bool,
        n_cubes: int,
        rng: Any | None,
    ) -> tuple[InstrumentSelection, Psf | None, ExposureOptions]:
        if self._selection is None:
            raise ValueError("Call configure() before run().")
        if not self._targets:
            raise ValueError("Add at least one target before run().")
        if self._exposure is None:
            raise ValueError("Call set_exposure() before run().")
        if self._psf is None and any(
            not isinstance(target.spatial, Uniform) for target in self._targets
        ):
            raise ValueError("Call set_psf() before run() for non-uniform targets.")
        if not isinstance(n_cubes, int) or isinstance(n_cubes, bool) or n_cubes <= 0:
            raise ValueError("n_cubes must be a positive integer.")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator.")
        if not include_data and (n_cubes != 1 or rng is not None):
            raise ValueError("n_cubes and rng require include_data=True.")
        if self._sky_subtraction.method == "in_field":
            expected = (
                self._selection.scale.spaxels_y,
                self._selection.scale.spaxels_x,
            )
            if self._sky_subtraction.mask.shape != expected:
                raise ValueError(f"In-field sky mask shape must be {expected}.")
        return self._selection, self._psf, self._resolve_exposure()

    def _resolve_exposure(self) -> ExposureOptions:
        request = self._exposure
        sky = self._sky_subtraction
        if sky.method == "in_field":
            count = request.n if request.n is not None else request.n_target
            n_target = count
            n_sky = 0
            total = count
        else:
            sequence = sky.sequence
            target_per_sequence = sequence.count("A")
            sky_per_sequence = sequence.count("B")
            if request.n is not None:
                if request.n % len(sequence) != 0:
                    raise ValueError("n must contain a whole number of sky sequences.")
                repeats = request.n // len(sequence)
                total = request.n
            else:
                if request.n_target % target_per_sequence != 0:
                    raise ValueError(
                        "n_target must contain a whole number of sky sequences."
                    )
                repeats = request.n_target // target_per_sequence
                total = repeats * len(sequence)
            n_target = repeats * target_per_sequence
            n_sky = repeats * sky_per_sequence
        return ExposureOptions(
            time=request.time,
            n=total,
            n_target=n_target,
            n_sky=n_sky,
            target_time=n_target * request.time,
            sky_time=n_sky * request.time,
            total_time=total * request.time,
        )


def _angle(value: Any, name: str) -> u.Quantity:
    if not isinstance(value, u.Quantity) or not value.isscalar:
        raise TypeError(f"{name} must be a scalar angular quantity.")
    try:
        resolved = value.to(u.deg)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(f"{name} must have angular units.") from exc
    if not np.isfinite(resolved.value):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _positive_time(value: Any) -> u.Quantity:
    if not isinstance(value, u.Quantity) or not value.isscalar:
        raise TypeError("time must be a scalar time quantity.")
    try:
        resolved = value.to(u.s)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError("time must have time units.") from exc
    if not np.isfinite(resolved.value) or resolved.value <= 0:
        raise ValueError("time must be finite and positive.")
    return resolved


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value
