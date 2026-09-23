"""Stateful public entrypoint for cubesim calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from cubesim._calculation import calculate
from cubesim._hybrid import model_psf
from cubesim._instrument import InstrumentSelection, load_instrument
from cubesim._psf import Psf, load_psf
from cubesim._result import (
    EtcResult,
    ExposureOptions,
    HybridOptions,
    ResultOptions,
    SkySubtractionOptions,
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
class _TargetRequest:
    ifu_offset: tuple[u.Quantity, u.Quantity] | None
    pointing_offset: tuple[u.Quantity, u.Quantity] | None
    sky_position: SkyCoord | None
    spatial: Point | Gaussian | Sersic | SpatialImage | Uniform
    spectrum: GaussianLines | TabulatedSpectrum
    velocity: ConstantVelocity | RotatingDisk | VelocityField | None


@dataclass(frozen=True, slots=True)
class _IfuPosition:
    pointing_offset: tuple[u.Quantity, u.Quantity] | None
    sky_position: SkyCoord | None
    rotation: u.Quantity


@dataclass(frozen=True, slots=True)
class _HybridRequest:
    coordinate_form: str
    ngs_pointing_offsets: tuple[tuple[u.Quantity, u.Quantity], ...] | None
    ngs_sky_positions: SkyCoord | None
    ngs_magnitudes: u.Quantity
    wavelength: u.Quantity
    zenith_angle: u.Quantity


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
            files it references. A relative directory is resolved from the
            current working directory, then from the nearest project root
            containing ``pyproject.toml`` when the current path does not exist.

    Notes:
        The object is mutable during configuration. Each call to :meth:`run`
        returns an independent immutable result.
    """

    def __init__(self, instrument_data: str | Path) -> None:
        self._instrument = load_instrument(instrument_data)
        self._selection: InstrumentSelection | None = None
        self._psf: Psf | None = None
        self._hybrid_psf: _HybridRequest | None = None
        self._targets: list[_TargetRequest] = []
        self._exposure: _ExposureRequest | None = None
        self._sky_subtraction = _SkySubtraction(
            method="nodding",
            sequence="AB",
            mask=None,
        )
        self._apertures: list[_Aperture] = []
        self._position_angle = 0 * u.deg
        self._pointing_center: SkyCoord | None = None
        self._ifu_position = _IfuPosition((0 * u.deg, 0 * u.deg), None, 0 * u.deg)

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
        sky_position: SkyCoord | None = None,
    ) -> None:
        """Set telescope field orientation and optional absolute sky center.

        Args:
            position_angle: Sky position angle of telescope-frame ``+y``,
                measured east of north. At zero, ``+x`` points west and
                ``+y`` points north.
            sky_position: Optional scalar absolute telescope pointing center.
                Geometry is resolved in ICRS; result options retain the
                supplied coordinate frame.
        """

        angle = _angle(position_angle, "position_angle")
        center = _sky_position(sky_position, "sky_position")
        self._position_angle = angle
        self._pointing_center = center

    def set_ifu_position(
        self,
        *,
        pointing_offset: tuple[u.Quantity, u.Quantity] | None = None,
        sky_position: SkyCoord | None = None,
        rotation: u.Quantity = 0 * u.deg,
    ) -> None:
        """Place and rotate the IFU within the telescope pointing frame.

        Exactly one of ``pointing_offset`` and ``sky_position`` is required.
        The offset is ``(x, y)`` in telescope axes or ``(r, theta)`` when its
        units are ``(arcsec, deg)``. Polar ``theta`` runs from ``+x`` toward
        ``+y``. Rotation is relative to telescope axes. The default IFU
        placement is the pointing center.
        """

        if (pointing_offset is None) == (sky_position is None):
            raise ValueError(
                "Provide exactly one of pointing_offset and sky_position."
            )
        offset = (
            _pointing_offset(pointing_offset, "pointing_offset")
            if pointing_offset is not None
            else None
        )
        sky = _sky_position(sky_position, "sky_position")
        if sky is not None and self._pointing_center is None:
            raise ValueError("sky_position requires an absolute telescope pointing.")
        self._ifu_position = _IfuPosition(offset, sky, _angle(rotation, "rotation"))

    def add_target(
        self,
        *,
        ifu_offset: tuple[u.Quantity, u.Quantity] | None = None,
        pointing_offset: tuple[u.Quantity, u.Quantity] | None = None,
        sky_position: SkyCoord | None = None,
        spatial: Point | Gaussian | Sersic | SpatialImage | Uniform,
        spectrum: GaussianLines | TabulatedSpectrum,
        velocity: ConstantVelocity | RotatingDisk | VelocityField | None = None,
    ) -> None:
        """Append one explicitly composed target to the calculation.

        Args:
            ifu_offset: ``(x, y)`` offset in IFU detector axes.
            pointing_offset: ``(x, y)`` in telescope axes, or ``(r, theta)``
                when the units are ``(arcsec, deg)``. Polar ``theta`` runs from
                ``+x`` toward ``+y``.
            sky_position: Absolute scalar sky coordinate in any frame
                convertible to ICRS. Exactly one target position form is
                required.
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
        positions = (ifu_offset, pointing_offset, sky_position)
        if sum(value is not None for value in positions) != 1:
            raise ValueError(
                "Provide exactly one of ifu_offset, pointing_offset, and sky_position."
            )
        ifu_offset = (
            _xy_offset(ifu_offset, "ifu_offset") if ifu_offset is not None else None
        )
        pointing_offset = (
            _pointing_offset(pointing_offset, "pointing_offset")
            if pointing_offset is not None
            else None
        )
        sky_position = _sky_position(sky_position, "sky_position")
        if sky_position is not None and self._pointing_center is None:
            raise ValueError("sky_position requires an absolute telescope pointing.")
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
            _TargetRequest(
                ifu_offset=ifu_offset,
                pointing_offset=pointing_offset,
                sky_position=sky_position,
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
                array indexed ``[y, x]`` and oriented to detector axes.
                Relative paths are resolved against the instrument-data
                directory.
            pixel_scale: Positive angular pixel scale required for NPY and
                in-memory inputs. FITS inputs instead read ``PIXSCALE`` in
                milliarcseconds per pixel.

        Calling this method again replaces the previous PSF.
        Direct PSFs are already in IFU detector axes and are not rotated when
        the telescope or IFU orientation changes.
        """

        self._psf = load_psf(
            psf,
            pixel_scale=pixel_scale,
            instrument_root=self._instrument.root,
        )
        self._hybrid_psf = None

    def set_hybrid_psf(
        self,
        *,
        ngs_pointing_offsets: tuple[tuple[u.Quantity, u.Quantity], ...] | None = None,
        ngs_sky_positions: SkyCoord | None = None,
        ngs_magnitudes: u.Quantity,
        wavelength: u.Quantity,
        zenith_angle: u.Quantity,
    ) -> None:
        """Configure one Hybrid AO PSF to be modelled by the next ``run()``.

        Supply exactly one NGS coordinate form. Pointing offsets are ``(x, y)``
        angular pairs in telescope axes, or ``(r, theta)`` when their units
        are ``(arcsec, deg)``. Polar ``theta`` runs from ``+x`` toward ``+y``.
        Sky positions are a one-dimensional
        ``SkyCoord`` and require an absolute telescope pointing. Magnitudes
        must be a real, finite, matching one-dimensional quantity in ``mag``.
        The wavelength must be a finite, positive scalar length and lie within
        the selected disperser's range at ``run()``. The zenith angle must be
        a scalar angle in ``[0, 90)`` degrees.
        The instrument's ``[hybrid]`` section supplies the zeropoint and
        asset paths. The current IFU center is the single science position.
        Calling ``set_psf()`` or ``set_hybrid_psf()`` later replaces the PSF
        source; the last setter controls the next run.
        """

        if self._instrument.hybrid is None:
            raise ValueError("Instrument etc.ini has no [hybrid] configuration.")
        if (ngs_pointing_offsets is None) == (ngs_sky_positions is None):
            raise ValueError(
                "Provide exactly one of ngs_pointing_offsets and ngs_sky_positions."
            )
        offsets = None
        sky = None
        if ngs_pointing_offsets is not None:
            if not isinstance(ngs_pointing_offsets, (tuple, list)) or not ngs_pointing_offsets:
                raise ValueError(
                    "ngs_pointing_offsets must contain at least one (x, y) or (r, theta) pair."
                )
            offsets = tuple(
                _pointing_offset(point, f"ngs_pointing_offsets[{index}]")
                for index, point in enumerate(ngs_pointing_offsets)
            )
        else:
            if not isinstance(ngs_sky_positions, SkyCoord) or ngs_sky_positions.isscalar:
                raise TypeError("ngs_sky_positions must be a 1D SkyCoord.")
            if ngs_sky_positions.ndim != 1 or len(ngs_sky_positions) == 0:
                raise ValueError("ngs_sky_positions must be a nonempty 1D SkyCoord.")
            if self._pointing_center is None:
                raise ValueError(
                    "ngs_sky_positions requires an absolute telescope pointing."
                )
            sky = _icrs_position(ngs_sky_positions).copy()
            if not np.isfinite(sky.ra.to_value(u.deg)).all() or not np.isfinite(
                sky.dec.to_value(u.deg)
            ).all():
                raise ValueError("ngs_sky_positions must be finite.")
        if not isinstance(ngs_magnitudes, u.Quantity) or ngs_magnitudes.ndim != 1:
            raise TypeError("ngs_magnitudes must be a 1D magnitude quantity.")
        try:
            magnitudes = ngs_magnitudes.to(u.mag).copy()
        except u.UnitConversionError as exc:
            raise u.UnitConversionError("ngs_magnitudes must have mag units.") from exc
        if not np.isrealobj(magnitudes.value):
            raise ValueError("ngs_magnitudes must be real.")
        count = len(offsets) if offsets is not None else len(sky)
        if magnitudes.size != count or not np.isfinite(magnitudes.value).all():
            raise ValueError("ngs_magnitudes must be finite with one value per NGS.")
        if not isinstance(wavelength, u.Quantity) or not wavelength.isscalar:
            raise TypeError("wavelength must be a scalar spectral quantity.")
        try:
            wavelength = wavelength.to(u.um)
        except u.UnitConversionError as exc:
            raise u.UnitConversionError("wavelength must have length units.") from exc
        if not np.isfinite(wavelength.value) or wavelength.value <= 0:
            raise ValueError("wavelength must be finite and positive.")
        zenith_angle = _angle(zenith_angle, "zenith_angle")
        if not 0 <= zenith_angle.to_value(u.deg) < 90:
            raise ValueError("zenith_angle must be in [0, 90) degrees.")
        self._hybrid_psf = _HybridRequest(
            "pointing_offsets" if offsets is not None else "sky_positions",
            offsets,
            sky,
            magnitudes,
            wavelength.copy(),
            zenith_angle.copy(),
        )
        self._psf = None

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
                subtraction. ``True`` spaxels declare target-free sky samples.
        """

        if method == "nodding":
            if sequence is None:
                sequence = "AB"
            elif not isinstance(sequence, str):
                raise TypeError("Nodding sequence must be a string.")
            else:
                sequence = sequence.upper()
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
                pixel indices in ``(y, x)`` order; the spectral coordinate may
                be a pixel or scalar wavelength.
            start: Optional rectangle start coordinate using the same coordinate
                forms as ``center``.
            mask: Nonempty Boolean mask with the exact output shape
                ``(y, x, wavelength)``.

        A mask cannot be combined with rectangle arguments. ``center`` and
        ``start`` are mutually exclusive. Bounds are validated during
        :meth:`run`, after the wavelength grid is known. Without either
        placement, the aperture uses the central detector pixel spatially and
        the first wavelength of the first target's spectrum spectrally. An
        aperture cannot overlap an in-field sky mask.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("Aperture name must be a nonempty string.")
        if any(aperture.name == name for aperture in self._apertures):
            raise ValueError(f"Duplicate aperture name: {name!r}.")
        if mask is not None:
            if any(value is not None for value in (size, center, start)):
                raise ValueError("mask cannot be combined with rectangle arguments.")
            array = np.asarray(mask)
            if (
                array.ndim != 3
                or array.dtype != np.dtype(bool)
                or not array.any()
            ):
                raise ValueError(
                    "Aperture mask must be a nonempty three-dimensional Boolean array."
                )
            self._apertures.append(_Aperture(name, None, None, None, array.copy()))
            return
        if size is None:
            raise ValueError("Rectangular apertures require size.")
        if center is not None and start is not None:
            raise ValueError("Provide at most one of center and start.")
        if not isinstance(size, tuple):
            raise TypeError("Aperture size must be a (y, x, wavelength) tuple.")
        if len(size) != 3:
            raise ValueError("Aperture size must contain (y, x, wavelength).")
        resolved_size = tuple(
            _positive_integer(value, "aperture size") for value in size
        )
        for placement, label in ((center, "center"), (start, "start")):
            if placement is None:
                continue
            if not isinstance(placement, tuple):
                raise TypeError(f"Aperture {label} must be a three-coordinate tuple.")
            if len(placement) != 3:
                raise ValueError(f"Aperture {label} must contain three coordinates.")
        self._apertures.append(_Aperture(name, resolved_size, center, start, None))

    def run(
        self,
        *,
        include_models: bool = False,
    ) -> EtcResult:
        """Run the configured forward ETC calculation.

        Args:
            include_models: Include per-target high- and low-resolution model
                components, their combined detector-resolution cube, and the
                wavelength-grid transmission, sky, and thermal models.

        Returns:
            An immutable result containing S/N, wavelength, signal and variance
            components, resolved options, aperture reductions, and any requested
            model products.
        """

        if not isinstance(include_models, bool):
            raise TypeError("include_models must be a Boolean value.")
        selection, psf, exposure = self._validate_and_resolve()
        ifu_offset, ifu_center = self._resolve_ifu_position()
        ifu_angle = self._position_angle + self._ifu_position.rotation
        targets = tuple(
            self._resolve_target(target, ifu_offset, ifu_center, ifu_angle)
            for target in self._targets
        )
        hybrid_options: HybridOptions | None = None
        if self._hybrid_psf is not None:
            request = self._hybrid_psf
            if not (
                selection.disperser.wavelength_min
                <= request.wavelength
                <= selection.disperser.wavelength_max
            ):
                raise ValueError(
                    "Hybrid PSF wavelength must lie within the selected "
                    "disperser range."
                )
            if request.ngs_sky_positions is None:
                ngs_offsets = request.ngs_pointing_offsets
            else:
                if self._pointing_center is None:
                    raise ValueError(
                        "ngs_sky_positions requires an absolute telescope pointing."
                    )
                center = _icrs_position(self._pointing_center)
                ngs_offsets = tuple(
                    _sky_to_xy(*center.spherical_offsets_to(star), self._position_angle)
                    for star in request.ngs_sky_positions
                )
            psf, hybrid_options = model_psf(
                root=self._instrument.root,
                definition=self._instrument.hybrid,
                coordinate_form=request.coordinate_form,
                ngs_offsets=ngs_offsets,
                ngs_magnitudes=request.ngs_magnitudes,
                science_offset=ifu_offset,
                wavelength=request.wavelength,
                zenith_angle=request.zenith_angle,
                ifu_rotation=self._ifu_position.rotation,
            )
        output = calculate(
            instrument=self._instrument,
            selection=selection,
            targets=targets,
            psf=psf,
            exposure=exposure,
            sky_subtraction=self._sky_subtraction,
            apertures=tuple(self._apertures),
            position_angle=ifu_angle,
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
            ifu_position=self._ifu_position,
            ifu_center=ifu_center,
            ifu_position_angle=ifu_angle,
            targets=tuple(self._targets),
            psf_pixel_scale=psf.pixel_scale if psf is not None else None,
            psf_path=psf.path if psf is not None else None,
            exposure=exposure,
            sky_subtraction=SkySubtractionOptions(
                method=self._sky_subtraction.method,
                sequence=self._sky_subtraction.sequence,
                mask=sky_mask,
            ),
            hybrid=hybrid_options,
        )
        return EtcResult(
            snr=output.snr,
            wavelength=output.grid.wavelength,
            options=options,
            apertures=output.apertures,
            models=output.models if include_models else None,
            signals=output.signals,
            variances=output.variances,
            read_noise=self._instrument.detector.read_noise,
            psf=psf.data if psf is not None else None,
            psf_pixel_scale=psf.pixel_scale if psf is not None else None,
        )

    def _resolve_ifu_position(
        self,
    ) -> tuple[tuple[u.Quantity, u.Quantity], SkyCoord | None]:
        request = self._ifu_position
        if request.sky_position is not None:
            if self._pointing_center is None:
                raise ValueError(
                    "IFU sky_position requires an absolute telescope pointing."
                )
            ifu_center = _icrs_position(request.sky_position)
            east, north = _icrs_position(self._pointing_center).spherical_offsets_to(
                ifu_center
            )
            return _sky_to_xy(east, north, self._position_angle), ifu_center
        offset = request.pointing_offset
        if self._pointing_center is None:
            return offset, None
        east, north = _xy_to_sky(*offset, self._position_angle)
        return offset, _icrs_position(self._pointing_center).spherical_offsets_by(
            east, north
        )

    def _resolve_target(
        self,
        target: _TargetRequest,
        ifu_offset: tuple[u.Quantity, u.Quantity],
        ifu_center: SkyCoord | None,
        ifu_angle: u.Quantity,
    ) -> _Target:
        if target.ifu_offset is not None:
            east, north = _xy_to_sky(*target.ifu_offset, ifu_angle)
        elif target.pointing_offset is not None:
            x = target.pointing_offset[0] - ifu_offset[0]
            y = target.pointing_offset[1] - ifu_offset[1]
            east, north = _xy_to_sky(x, y, self._position_angle)
        else:
            if ifu_center is None:
                raise ValueError(
                    "Target sky_position requires an absolute telescope pointing."
                )
            east, north = ifu_center.spherical_offsets_to(
                _icrs_position(target.sky_position)
            )
        return _Target(
            position=(east, north),
            spatial=target.spatial,
            spectrum=target.spectrum,
            velocity=target.velocity,
        )

    def _validate_and_resolve(
        self,
    ) -> tuple[InstrumentSelection, Psf | None, ExposureOptions]:
        if self._selection is None:
            raise ValueError("Call configure() before run().")
        if not self._targets:
            raise ValueError("Add at least one target before run().")
        if self._exposure is None:
            raise ValueError("Call set_exposure() before run().")
        if self._psf is None and self._hybrid_psf is None and any(
            not isinstance(target.spatial, Uniform) for target in self._targets
        ):
            raise ValueError("Call set_psf() before run() for non-uniform targets.")
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


def _xy_offset(value: Any, name: str) -> tuple[u.Quantity, u.Quantity]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an (x, y) angular tuple.")
    return _angle(value[0], f"{name} x"), _angle(value[1], f"{name} y")


def _pointing_offset(value: Any, name: str) -> tuple[u.Quantity, u.Quantity]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an (x, y) or (r, theta) angular tuple.")
    radius, theta = value
    # Both coordinate forms are angular; exact units identify polar input.
    if (
        isinstance(radius, u.Quantity)
        and isinstance(theta, u.Quantity)
        and radius.unit == u.arcsec
        and theta.unit == u.deg
    ):
        radius = _angle(radius, f"{name} r")
        theta = _angle(theta, f"{name} theta")
        if radius.value < 0:
            raise ValueError(f"{name} r must be nonnegative.")
        angle = theta.to_value(u.rad)
        return radius * np.cos(angle), radius * np.sin(angle)
    return _xy_offset(value, name)


def _sky_position(value: Any, name: str) -> SkyCoord | None:
    if value is None:
        return None
    if not isinstance(value, SkyCoord) or not value.isscalar:
        raise TypeError(f"{name} must be a scalar astropy.coordinates.SkyCoord.")
    if not np.isfinite(value.icrs.ra.to_value(u.deg)) or not np.isfinite(
        value.icrs.dec.to_value(u.deg)
    ):
        raise ValueError(f"{name} must have finite coordinates.")
    return value.copy()


def _icrs_position(value: SkyCoord) -> SkyCoord:
    # SkyCoord retains source-frame attributes after .icrs; offsets require equal frames.
    return SkyCoord(value.icrs.frame)


def _xy_to_sky(
    x: u.Quantity, y: u.Quantity, position_angle: u.Quantity
) -> tuple[u.Quantity, u.Quantity]:
    angle = position_angle.to_value(u.rad)
    east = -x * np.cos(angle) + y * np.sin(angle)
    north = x * np.sin(angle) + y * np.cos(angle)
    return east, north


def _sky_to_xy(
    east: u.Quantity, north: u.Quantity, position_angle: u.Quantity
) -> tuple[u.Quantity, u.Quantity]:
    angle = position_angle.to_value(u.rad)
    x = -east * np.cos(angle) + north * np.sin(angle)
    y = east * np.sin(angle) + north * np.cos(angle)
    return x, y


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
