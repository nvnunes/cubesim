"""Forward ETC calculation preserving the retained numerical operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.modeling.models import Sersic2D
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.sparse import lil_matrix
from scipy.special import gammaincinv, gammaln

from cubesim._instrument import InstrumentDefinition, InstrumentSelection
from cubesim._psf import Psf, _center_psf
from cubesim._result import (
    ApertureProjection,
    ApertureResult,
    ModelGrid,
    Models,
    Signals,
    TargetModels,
    Variances,
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

_C = 2.99792458e8 * u.m / u.s
_H = 6.62607015e-34 * u.J * u.s
_K = 1.380649e-23 * u.J / u.K
_RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0 * u.arcsec
_ARCSEC2_TO_SR = (1 / _RAD_TO_ARCSEC.value) ** 2
_SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
_FWHM_TO_SIGMA = 1 / _SIGMA_TO_FWHM
_MODEL_UNIT = u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.m)
_LSF_MARGIN_SIGMA = 8
_VELOCITY_CHUNK_ELEMENTS = 4_000_000


@dataclass(frozen=True, slots=True)
class SpectralGrid:
    wavelength: u.Quantity
    edges: u.Quantity
    step: u.Quantity
    high_wavelength: u.Quantity
    high_step: u.Quantity
    resolution_fwhm: u.Quantity
    high_sigma_pixels: float
    photon_energy: u.Quantity


@dataclass(frozen=True, slots=True)
class CalculationOutput:
    grid: SpectralGrid
    models: Models
    signals: Signals
    variances: Variances
    snr: np.ndarray
    apertures: tuple[ApertureResult, ...]
    data: u.Quantity | None


def calculate(
    *,
    instrument: InstrumentDefinition,
    selection: InstrumentSelection,
    targets: tuple[Any, ...],
    psf: Psf | None,
    exposure: Any,
    sky_subtraction: Any,
    apertures: tuple[Any, ...],
    position_angle: u.Quantity,
    include_signals: bool,
    include_variances: bool,
    include_data: bool,
    n_cubes: int,
    rng: np.random.Generator | None,
) -> CalculationOutput:
    """Execute one resolved calculation without mutating caller-owned state."""

    grid = _build_spectral_grid(selection)
    target_models = tuple(
        _create_target_models(selection, grid, target, psf, position_angle)
        for target in targets
    )
    combined = sum(
        (target.combined for target in target_models),
        start=np.zeros(
            (
                selection.scale.spaxels_y,
                selection.scale.spaxels_x,
                len(grid.wavelength),
            )
        )
        * _MODEL_UNIT,
    )
    transmission = _smooth_table(
        selection.atmosphere.transmission,
        grid,
        sigma_divisor=1.0,
    )
    sky = _smooth_background(selection.atmosphere.background, grid)
    sky_radiance = (
        sky
        * grid.photon_energy.value
        / _ARCSEC2_TO_SR
        * 1.0e6
        * u.W
        / (u.m**2 * u.sr * u.m)
    )
    models = Models(
        targets=tuple(target.models for target in target_models),
        combined=combined,
        transmission=transmission,
        sky=sky_radiance,
        thermal=_thermal_background(selection, grid.wavelength),
    )
    signals, variances, snr = _detector_products(
        instrument,
        selection,
        grid,
        models,
        exposure,
        sky_subtraction,
    )
    data = None
    if include_data:
        data = _sample_data(
            signals,
            exposure,
            instrument.detector.read_noise,
            sky_subtraction,
            n_cubes,
            rng,
        )

    aperture_results = tuple(
        _reduce_aperture(
            aperture,
            grid,
            selection,
            targets,
            signals,
            variances,
            data,
            sky_subtraction,
            exposure,
            instrument.detector.read_noise,
            include_signals,
            include_variances,
        )
        for aperture in apertures
    )
    return CalculationOutput(
        grid=grid,
        models=models,
        signals=signals,
        variances=variances,
        snr=snr,
        apertures=aperture_results,
        data=data,
    )


@dataclass(frozen=True, slots=True)
class _TargetCalculation:
    models: TargetModels
    combined: u.Quantity


def _build_spectral_grid(selection: InstrumentSelection) -> SpectralGrid:
    disperser = selection.disperser
    reference = (disperser.wavelength_min + disperser.wavelength_max) / 2
    resolution_fwhm = (reference / disperser.resolving_power).to(u.nm)
    step = resolution_fwhm / disperser.pixels_per_resolution_element
    wavelength = (
        np.arange(
            disperser.wavelength_min.to_value(u.micron),
            disperser.wavelength_max.to_value(u.micron),
            step.to_value(u.micron),
        )
        * u.micron
    )
    if len(wavelength) < 2:
        raise ValueError(
            "Selected disperser produces fewer than two detector wavelength samples."
        )
    edges = wavelength - 0.5 * step
    edges = np.append(edges, wavelength[-1] + 0.5 * step)
    high_step = resolution_fwhm / 10
    high_wavelength = (
        np.arange(
            disperser.wavelength_min.to_value(u.micron),
            disperser.wavelength_max.to_value(u.micron),
            high_step.to_value(u.micron),
        )
        * u.micron
    )
    if len(high_wavelength) < 2:
        raise ValueError(
            "Selected disperser produces fewer than two high-resolution wavelength "
            "samples."
        )
    high_sigma_pixels = (
        resolution_fwhm.to_value(u.micron)
        * _FWHM_TO_SIGMA
        / high_step.to_value(u.micron)
    )
    return SpectralGrid(
        wavelength=wavelength,
        edges=edges,
        step=step,
        high_wavelength=high_wavelength,
        high_step=high_step,
        resolution_fwhm=resolution_fwhm,
        high_sigma_pixels=high_sigma_pixels,
        photon_energy=_H * _C / wavelength.to(u.m),
    )


def _create_target_models(
    selection: InstrumentSelection,
    grid: SpectralGrid,
    target: Any,
    psf: Psf | None,
    position_angle: u.Quantity,
) -> _TargetCalculation:
    if isinstance(target.velocity, (RotatingDisk, VelocityField)):
        return _create_varying_velocity_target_models(
            selection,
            grid,
            target,
            psf,
            position_angle,
        )

    spectrum = target.spectrum

    uniform = isinstance(target.spatial, Uniform)
    pixel_omega = _pixel_omega(selection.scale.spaxel_scale)
    velocity_offset = (
        target.velocity.offset
        if isinstance(target.velocity, ConstantVelocity)
        else None
    )
    if isinstance(spectrum, GaussianLines):
        spectrum_high = _create_spectrum_high(
            grid,
            spectrum,
            pixel_omega,
            surface_brightness=uniform,
            velocity_offset=velocity_offset,
        )
    else:
        spectrum_high = _create_tabulated_spectrum_high(
            grid,
            spectrum,
            pixel_omega,
            surface_brightness=uniform,
            velocity_offset=velocity_offset,
        )
    spectrum_convolved = _convolve_spectrum(grid, spectrum_high)
    spectrum_low = _resample_spectrum(grid, spectrum_convolved)

    if uniform:
        shape = (selection.scale.spaxels_y, selection.scale.spaxels_x)
        spatial_high = np.ones(shape)
        spatial_convolved = None
        spatial_low = np.ones(shape)
    else:
        spatial_high = _create_spatial_high(
            selection,
            target,
            psf.pixel_scale,
            position_angle,
        )
        spatial_convolved = _convolve_spatial(spatial_high, psf.data)
        spatial_low, _ = _resample_spatial(
            spatial_convolved,
            psf.pixel_scale,
            selection,
        )

    velocity_high = None
    velocity_low = None
    if velocity_offset is not None:
        velocity_high = (
            np.full(spatial_high.shape, velocity_offset.value) * velocity_offset.unit
        )
        velocity_low = (
            np.full(spatial_low.shape, velocity_offset.value) * velocity_offset.unit
        )

    combined = spatial_low[:, :, None] * spectrum_low[None, None, :]
    models = TargetModels(
        high=ModelGrid(
            wavelength=grid.high_wavelength,
            spatial=spatial_high,
            spatial_convolved=spatial_convolved,
            spectrum=spectrum_high,
            spectrum_convolved=spectrum_convolved,
            velocity=velocity_high,
        ),
        low=ModelGrid(
            wavelength=grid.wavelength,
            spatial=spatial_low,
            spectrum=spectrum_low,
            velocity=velocity_low,
        ),
    )
    return _TargetCalculation(models=models, combined=combined)


def _create_varying_velocity_target_models(
    selection: InstrumentSelection,
    grid: SpectralGrid,
    target: Any,
    psf: Psf | None,
    position_angle: u.Quantity,
) -> _TargetCalculation:
    uniform = isinstance(target.spatial, Uniform)
    pixel_omega = _pixel_omega(selection.scale.spaxel_scale)
    if uniform:
        shape = (selection.scale.spaxels_y, selection.scale.spaxels_x)
        high_scale = selection.scale.spaxel_scale
        spatial_high = np.ones(shape)
        spatial_convolved = None
        spatial_low = np.ones(shape)
        psf_data = None
    else:
        if psf is None:
            raise ValueError("A spatially varying target requires a PSF.")
        high_scale = psf.pixel_scale
        spatial_high = _create_spatial_high(
            selection,
            target,
            high_scale,
            position_angle,
        )
        spatial_convolved = _convolve_spatial(spatial_high, psf.data)
        spatial_low, _ = _resample_spatial(
            spatial_convolved,
            high_scale,
            selection,
        )
        psf_data = psf.data

    velocity_high = _create_velocity_high(
        target,
        spatial_high.shape,
        high_scale,
        position_angle,
    )
    if uniform:
        velocity_low = u.Quantity(velocity_high, copy=True)
    else:
        velocity_low = _resample_velocity(
            velocity_high,
            high_scale,
            selection,
        )
    velocity_grid = _extend_grid_for_velocity(grid, velocity_high)
    if isinstance(target.spectrum, GaussianLines):
        spectrum_high = _create_spectrum_high(
            velocity_grid,
            target.spectrum,
            pixel_omega,
            surface_brightness=uniform,
            velocity_offset=None,
        )
    else:
        spectrum_high = _create_tabulated_spectrum_high(
            velocity_grid,
            target.spectrum,
            pixel_omega,
            surface_brightness=uniform,
            velocity_offset=None,
        )
    spectrum_convolved = _convolve_spectrum(velocity_grid, spectrum_high)
    spectrum_low = _resample_spectrum(
        velocity_grid,
        spectrum_convolved,
        discard_outside=True,
    )
    combined = _assemble_varying_velocity_cube(
        selection,
        velocity_grid,
        spatial_high,
        high_scale,
        spectrum_high,
        velocity_high,
        psf_data,
    )
    models = TargetModels(
        high=ModelGrid(
            wavelength=velocity_grid.high_wavelength,
            spatial=spatial_high,
            spatial_convolved=spatial_convolved,
            spectrum=spectrum_high,
            spectrum_convolved=spectrum_convolved,
            velocity=velocity_high,
        ),
        low=ModelGrid(
            wavelength=grid.wavelength,
            spatial=spatial_low,
            spectrum=spectrum_low,
            velocity=velocity_low,
        ),
    )
    return _TargetCalculation(models=models, combined=combined)


def _extend_grid_for_velocity(
    grid: SpectralGrid,
    velocity: u.Quantity,
) -> SpectralGrid:
    maximum_beta = np.max(np.abs(velocity.to_value(u.km / u.s))) / _C.to_value(
        u.km / u.s
    )
    if maximum_beta >= 1:
        raise ValueError("Velocity magnitude must be less than the speed of light.")
    lsf_margin = (_LSF_MARGIN_SIGMA * grid.resolution_fwhm * _FWHM_TO_SIGMA).to(
        u.micron
    )
    output_min = grid.edges[0].to(u.micron) - lsf_margin
    output_max = grid.edges[-1].to(u.micron) + lsf_margin
    required_min = min(output_min, output_min / (1 + maximum_beta))
    required_max = max(output_max, output_max / (1 - maximum_beta))
    step = grid.high_step.to(u.micron)
    first = grid.high_wavelength[0].to(u.micron)
    last = grid.high_wavelength[-1].to(u.micron)
    before = max(0, int(np.ceil(((first - required_min) / step).value)))
    after = max(0, int(np.ceil(((required_max - last) / step).value)))
    offsets = np.arange(-before, len(grid.high_wavelength) + after)
    high_wavelength = first + offsets * step
    return SpectralGrid(
        wavelength=grid.wavelength,
        edges=grid.edges,
        step=grid.step,
        high_wavelength=high_wavelength,
        high_step=grid.high_step,
        resolution_fwhm=grid.resolution_fwhm,
        high_sigma_pixels=grid.high_sigma_pixels,
        photon_energy=grid.photon_energy,
    )


def _create_velocity_high(
    target: Any,
    shape: tuple[int, int],
    pixel_scale: u.Quantity,
    pointing_angle: u.Quantity,
) -> u.Quantity:
    ny, nx = shape
    east, north = target.position
    offset_x, offset_y = _sky_to_array(east, north, pointing_angle)
    center_x = nx / 2 + offset_x.to_value(u.arcsec) / pixel_scale.to_value(u.arcsec)
    center_y = ny / 2 + offset_y.to_value(u.arcsec) / pixel_scale.to_value(u.arcsec)
    velocity = target.velocity
    if isinstance(velocity, VelocityField):
        return _resample_velocity_field(
            velocity,
            shape,
            pixel_scale,
            center_y,
            center_x,
            pointing_angle,
        )

    x = (np.arange(nx) + 0.5 - center_x) * pixel_scale.to_value(u.arcsec)
    y = (np.arange(ny) + 0.5 - center_y) * pixel_scale.to_value(u.arcsec)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    angle = _sky_angle_to_array(velocity.position_angle, pointing_angle)
    theta = angle.to_value(u.rad)
    major = np.cos(theta) * xx + np.sin(theta) * yy
    minor = -np.sin(theta) * xx + np.cos(theta) * yy
    cosine_inclination = np.cos(velocity.inclination.to_value(u.rad))
    radius = np.sqrt(major**2 + (minor / cosine_inclination) ** 2)
    curve = (
        (2 / np.pi)
        * velocity.maximum_velocity.to_value(u.km / u.s)
        * np.arctan(radius / velocity.turnover_radius.to_value(u.arcsec))
    )
    projected = np.divide(
        curve * np.sin(velocity.inclination.to_value(u.rad)) * major,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0,
    )
    return (projected + velocity.systemic_velocity.to_value(u.km / u.s)) * (u.km / u.s)


def _resample_velocity_field(
    model: VelocityField,
    shape: tuple[int, int],
    output_scale: u.Quantity,
    center_y: float,
    center_x: float,
    pointing_angle: u.Quantity,
) -> u.Quantity:
    ny, nx = shape
    y = (np.arange(ny) + 0.5 - center_y) * output_scale.to_value(u.arcsec)
    x = (np.arange(nx) + 0.5 - center_x) * output_scale.to_value(u.arcsec)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    relative_angle = (model.position_angle - pointing_angle).to_value(u.rad)
    source_x = np.cos(relative_angle) * xx + np.sin(relative_angle) * yy
    source_y = -np.sin(relative_angle) * xx + np.cos(relative_angle) * yy
    source_scale = model.pixel_scale.to_value(u.arcsec)
    coordinates = np.array(
        [
            source_y / source_scale + (model.data.shape[0] - 1) / 2,
            source_x / source_scale + (model.data.shape[1] - 1) / 2,
        ]
    )
    tolerance = 1e-9
    if (
        np.any(coordinates[0] < -tolerance)
        or np.any(coordinates[0] > model.data.shape[0] - 1 + tolerance)
        or np.any(coordinates[1] < -tolerance)
        or np.any(coordinates[1] > model.data.shape[1] - 1 + tolerance)
    ):
        raise ValueError("Velocity field does not cover the selected IFU.")
    coordinates[0] = np.clip(coordinates[0], 0, model.data.shape[0] - 1)
    coordinates[1] = np.clip(coordinates[1], 0, model.data.shape[1] - 1)
    values = ndimage.map_coordinates(
        model.data.to_value(u.km / u.s),
        coordinates,
        order=1,
        mode="nearest",
    )
    return values * u.km / u.s


def _resample_velocity(
    velocity: u.Quantity,
    pixel_scale: u.Quantity,
    selection: InstrumentSelection,
) -> u.Quantity:
    ny_high, nx_high = velocity.shape
    scale = selection.scale
    weights = _flux_conserving_weights(
        ny_high,
        nx_high,
        scale.spaxels_y,
        scale.spaxels_x,
        pixel_scale,
        pixel_scale,
        scale.spaxel_scale,
        scale.spaxel_scale,
    ).tocsr()
    numerator = weights @ velocity.to_value(u.km / u.s).reshape(-1, 1)
    denominator = weights @ np.ones((ny_high * nx_high, 1))
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return values.reshape(scale.spaxels_y, scale.spaxels_x) * u.km / u.s


def _assemble_varying_velocity_cube(
    selection: InstrumentSelection,
    grid: SpectralGrid,
    spatial: np.ndarray,
    pixel_scale: u.Quantity,
    spectrum: u.Quantity,
    velocity: u.Quantity,
    psf: np.ndarray | None,
) -> u.Quantity:
    ny, nx = spatial.shape
    scale = selection.scale
    output = np.empty(
        (scale.spaxels_y, scale.spaxels_x, len(grid.high_wavelength)),
        dtype=float,
    )
    if psf is None:
        kernel = None
        weights = None
        area_factor = 1.0
    else:
        kernel = _prepare_psf(psf, ny, nx)[:, :, None]
        weights = _flux_conserving_weights(
            ny,
            nx,
            scale.spaxels_y,
            scale.spaxels_x,
            pixel_scale,
            pixel_scale,
            scale.spaxel_scale,
            scale.spaxel_scale,
        ).tocsr()
        area_factor = (
            scale.spaxel_scale.to_value(u.arcsec) ** 2
            / pixel_scale.to_value(u.arcsec) ** 2
        )
    chunk_size = max(1, _VELOCITY_CHUNK_ELEMENTS // (ny * nx))
    for start in range(0, len(grid.high_wavelength), chunk_size):
        stop = min(start + chunk_size, len(grid.high_wavelength))
        shifted = _shift_spectrum_by_velocity(
            grid.high_wavelength,
            spectrum,
            velocity,
            grid.high_wavelength[start:stop],
        )
        cube = spatial[:, :, None] * shifted.value
        if kernel is not None:
            cube = fftconvolve(cube, kernel, mode="same", axes=(0, 1))
            cube = weights @ cube.reshape(ny * nx, stop - start)
            cube = cube.reshape(scale.spaxels_y, scale.spaxels_x, stop - start)
            cube *= area_factor
        output[:, :, start:stop] = cube

    sum_before = output.sum()
    ndimage.gaussian_filter1d(
        output,
        sigma=grid.high_sigma_pixels,
        axis=2,
        mode="reflect",
        output=output,
    )
    sum_after = output.sum()
    if sum_before != sum_after:
        output *= sum_before / sum_after
    return _resample_spectrum(
        grid,
        output * spectrum.unit,
        discard_outside=True,
    )


def _shift_spectrum_by_velocity(
    source_wavelength: u.Quantity,
    source_spectrum: u.Quantity,
    velocity: u.Quantity,
    output_wavelength: u.Quantity,
) -> u.Quantity:
    factor = 1 + velocity.to_value(u.km / u.s)[:, :, None] / _C.to_value(u.km / u.s)
    if np.any(factor <= 0):
        raise ValueError("Velocity produces a non-positive Doppler factor.")
    source_coordinate = output_wavelength.to_value(u.micron)[None, None, :] / factor
    first = source_wavelength[0].to_value(u.micron)
    step = (source_wavelength[1] - source_wavelength[0]).to_value(u.micron)
    index = (source_coordinate - first) / step
    valid = (index >= 0) & (index <= len(source_wavelength) - 1)
    lower = np.floor(index).astype(np.int64)
    lower = np.clip(lower, 0, len(source_wavelength) - 2)
    fraction = index - lower
    values = source_spectrum.value
    shifted = (1 - fraction) * values[lower] + fraction * values[lower + 1]
    shifted = np.where(valid, shifted / factor, 0.0)
    return shifted * source_spectrum.unit


def _create_spectrum_high(
    grid: SpectralGrid,
    spectrum: GaussianLines,
    pixel_omega: u.Quantity,
    *,
    surface_brightness: bool,
    velocity_offset: u.Quantity | None,
) -> u.Quantity:
    wavelengths = np.atleast_1d(spectrum.wavelength)
    if velocity_offset is not None:
        wavelengths = wavelengths * (1 + velocity_offset / _C.to(u.km / u.s))
    fluxes = spectrum.flux
    if fluxes.isscalar:
        if spectrum.flux_ratios is None:
            fluxes = np.full(len(wavelengths), fluxes.value) * fluxes.unit
        else:
            fluxes = fluxes * np.concatenate(([1.0], spectrum.flux_ratios))

    output = np.zeros(grid.high_wavelength.shape) * _MODEL_UNIT
    for wavelength, flux in zip(wavelengths, fluxes, strict=True):
        if (
            wavelength < grid.high_wavelength[0]
            or wavelength > grid.high_wavelength[-1]
        ):
            continue
        fwhm = _line_fwhm(spectrum, wavelength)
        sigma = fwhm * _FWHM_TO_SIGMA
        profile = np.exp(
            -0.5
            * (
                (
                    grid.high_wavelength.to_value(u.micron)
                    - wavelength.to_value(u.micron)
                )
                / sigma.to_value(u.micron)
            )
            ** 2
        )
        profile_sum = profile.sum()
        if not np.isfinite(profile_sum):
            raise ValueError("Gaussian line profile contains non-finite values.")
        if profile_sum == 0:
            raise ValueError("Gaussian line is not sampled by the spectral grid.")
        profile /= profile_sum
        if surface_brightness:
            line = (
                profile
                * flux.to(u.erg / (u.s * u.cm**2 * u.arcsec**2))
                / grid.high_step.to(u.m)
            )
        else:
            line = (
                profile
                * flux.to(u.erg / (u.s * u.cm**2))
                / pixel_omega
                / grid.high_step.to(u.m)
            )
        output += line.to(_MODEL_UNIT)
    if spectrum.continuum is not None:
        continuum = spectrum.continuum
        if velocity_offset is not None:
            continuum = continuum / (1 + velocity_offset / _C.to(u.km / u.s))
        if not surface_brightness:
            continuum = continuum / pixel_omega
        output += continuum.to(_MODEL_UNIT)
    return output


def _line_fwhm(spectrum: GaussianLines, wavelength: u.Quantity) -> u.Quantity:
    width = spectrum.fwhm
    if width is None:
        width = spectrum.dispersion * _SIGMA_TO_FWHM
    elif width.unit.is_equivalent(u.km / u.s):
        width = width * _FWHM_TO_SIGMA * _SIGMA_TO_FWHM
    else:
        dispersion = width.to(u.micron) * _FWHM_TO_SIGMA / wavelength * _C
        width = dispersion * _SIGMA_TO_FWHM / _C * wavelength
    if width.unit.is_equivalent(u.km / u.s):
        return (width / _C * wavelength).to(u.nm)
    return width.to(u.nm)


def _create_tabulated_spectrum_high(
    grid: SpectralGrid,
    spectrum: TabulatedSpectrum,
    pixel_omega: u.Quantity,
    *,
    surface_brightness: bool,
    velocity_offset: u.Quantity | None,
) -> u.Quantity:
    source_wavelength = spectrum.wavelength
    doppler_factor = 1.0
    if velocity_offset is not None:
        doppler_factor = 1 + velocity_offset / _C.to(u.km / u.s)
        source_wavelength = source_wavelength * doppler_factor
    if (
        source_wavelength[0] > grid.high_wavelength[0]
        or source_wavelength[-1] < grid.high_wavelength[-1]
    ):
        raise ValueError("Tabulated spectrum does not cover the selected disperser.")
    values = (
        np.interp(
            grid.high_wavelength.to_value(u.micron),
            source_wavelength.to_value(u.micron),
            spectrum.flux.value,
        )
        * spectrum.flux.unit
        / doppler_factor
    )
    if not surface_brightness:
        values = values / pixel_omega
    return values.to(_MODEL_UNIT)


def _convolve_spectrum(
    grid: SpectralGrid,
    spectrum: u.Quantity,
) -> u.Quantity:
    sum_before = spectrum.value.sum()
    convolved = ndimage.gaussian_filter1d(
        spectrum.value,
        sigma=grid.high_sigma_pixels,
        mode="reflect",
    )
    sum_after = convolved.sum()
    if sum_before != sum_after:
        convolved *= sum_before / sum_after
    return convolved * spectrum.unit


def _resample_spectrum(
    grid: SpectralGrid,
    spectrum: u.Quantity,
    *,
    discard_outside: bool = False,
) -> u.Quantity:
    indices = (
        np.digitize(
            grid.high_wavelength.to_value(u.micron),
            grid.edges.to_value(u.micron),
            right=True,
        )
        - 1
    )
    if discard_outside:
        valid = (indices >= 0) & (indices < len(grid.wavelength))
    else:
        indices = np.clip(indices, 0, len(grid.wavelength) - 1)
        valid = np.ones(indices.shape, dtype=bool)
    values = np.zeros((*spectrum.shape[:-1], len(grid.wavelength)))
    output = np.moveaxis(values, -1, 0)
    samples = np.moveaxis(spectrum.value[..., valid], -1, 0)
    np.add.at(output, indices[valid], samples)
    values *= grid.high_step.to_value(u.micron) / grid.step.to_value(u.micron)
    return values * spectrum.unit


def _create_spatial_high(
    selection: InstrumentSelection,
    target: Any,
    pixel_scale: u.Quantity,
    pointing_angle: u.Quantity,
) -> np.ndarray:
    scale = selection.scale
    fov_x = scale.spaxels_x * scale.spaxel_scale
    fov_y = scale.spaxels_y * scale.spaxel_scale
    nx = int(np.ceil(fov_x.to_value(u.arcsec) / pixel_scale.to_value(u.arcsec)))
    ny = int(np.ceil(fov_y.to_value(u.arcsec) / pixel_scale.to_value(u.arcsec)))
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1

    east, north = target.position
    offset_x, offset_y = _sky_to_array(east, north, pointing_angle)
    center_x = nx / 2 + offset_x.to_value(u.arcsec) / pixel_scale.to_value(u.arcsec)
    center_y = ny / 2 + offset_y.to_value(u.arcsec) / pixel_scale.to_value(u.arcsec)

    if isinstance(target.spatial, Point):
        profile = np.zeros((ny, nx))
        center_x -= 0.5
        center_y -= 0.5
        x0 = int(np.floor(center_x))
        y0 = int(np.floor(center_y))
        dx = center_x - x0
        dy = center_y - y0
        _deposit(profile, y0, x0, (1 - dx) * (1 - dy))
        _deposit(profile, y0, x0 + 1, dx * (1 - dy))
        _deposit(profile, y0 + 1, x0, (1 - dx) * dy)
        _deposit(profile, y0 + 1, x0 + 1, dx * dy)
        return profile

    if isinstance(target.spatial, SpatialImage):
        return _resample_spatial_image(
            target.spatial,
            ny,
            nx,
            pixel_scale,
            center_y,
            center_x,
            pointing_angle,
        )

    if not isinstance(target.spatial, (Gaussian, Sersic)):
        raise NotImplementedError("Unsupported spatial model.")
    subsample = 10
    x = (
        (np.arange(nx * subsample) + 0.5) / subsample - center_x
    ) * pixel_scale.to_value(u.arcsec)
    y = (
        (np.arange(ny * subsample) + 0.5) / subsample - center_y
    ) * pixel_scale.to_value(u.arcsec)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    angle = _sky_angle_to_array(target.spatial.position_angle, pointing_angle)
    theta = angle.to_value(u.rad)
    if isinstance(target.spatial, Gaussian):
        sigma_major = target.spatial.fwhm.to_value(u.arcsec) * _FWHM_TO_SIGMA
        sigma_minor = sigma_major * target.spatial.axis_ratio
        xp = np.cos(theta) * xx + np.sin(theta) * yy
        yp = -np.sin(theta) * xx + np.cos(theta) * yy
        profile = np.exp(-0.5 * ((xp / sigma_major) ** 2 + (yp / sigma_minor) ** 2))
        total = 2 * np.pi * sigma_major * sigma_minor
    else:
        sersic = Sersic2D(
            amplitude=1.0,
            r_eff=target.spatial.effective_radius.to_value(u.arcsec),
            n=target.spatial.index,
            ellip=1 - target.spatial.axis_ratio,
            x_0=0.0,
            y_0=0.0,
            theta=-theta,
        )
        profile = sersic(xx, yy)
        index = target.spatial.index
        b_n = gammaincinv(2 * index, 0.5)
        radius = target.spatial.effective_radius.to_value(u.arcsec)
        log_total = (
            np.log(2 * np.pi * index * target.spatial.axis_ratio * radius**2)
            + b_n
            + gammaln(2 * index)
            - 2 * index * np.log(b_n)
        )
        total = np.exp(log_total)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Spatial profile normalization is not finite and positive.")
    profile = profile.reshape(ny, subsample, nx, subsample).sum(axis=(1, 3))
    subpixel_area = (pixel_scale.to_value(u.arcsec) / subsample) ** 2
    profile *= subpixel_area / total
    if not np.isfinite(profile).all():
        raise ValueError("Spatial profile contains non-finite values.")
    return profile


def _resample_spatial_image(
    model: SpatialImage,
    ny: int,
    nx: int,
    output_scale: u.Quantity,
    center_y: float,
    center_x: float,
    pointing_angle: u.Quantity,
) -> np.ndarray:
    y = (np.arange(ny) + 0.5 - center_y) * output_scale.to_value(u.arcsec)
    x = (np.arange(nx) + 0.5 - center_x) * output_scale.to_value(u.arcsec)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    relative_angle = (model.position_angle - pointing_angle).to_value(u.rad)
    source_x = np.cos(relative_angle) * xx + np.sin(relative_angle) * yy
    source_y = -np.sin(relative_angle) * xx + np.cos(relative_angle) * yy
    source_scale = model.pixel_scale.to_value(u.arcsec)
    coordinates = np.array(
        [
            source_y / source_scale + (model.data.shape[0] - 1) / 2,
            source_x / source_scale + (model.data.shape[1] - 1) / 2,
        ]
    )
    profile = ndimage.map_coordinates(
        model.data,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    profile *= (output_scale.to_value(u.arcsec) / source_scale) ** 2
    return profile


def _deposit(profile: np.ndarray, y: int, x: int, weight: float) -> None:
    if weight != 0 and 0 <= y < profile.shape[0] and 0 <= x < profile.shape[1]:
        profile[y, x] += weight


def _sky_to_array(
    east: u.Quantity,
    north: u.Quantity,
    position_angle: u.Quantity,
) -> tuple[u.Quantity, u.Quantity]:
    angle = position_angle.to_value(u.rad)
    x = -east * np.cos(angle) + north * np.sin(angle)
    y = east * np.sin(angle) + north * np.cos(angle)
    return x, y


def _sky_angle_to_array(
    model_angle: u.Quantity,
    pointing_angle: u.Quantity,
) -> u.Quantity:
    relative = (model_angle - pointing_angle).to_value(u.rad)
    x = -np.sin(relative)
    y = np.cos(relative)
    return np.arctan2(y, x) * u.rad


def _convolve_spatial(profile: np.ndarray, psf: np.ndarray) -> np.ndarray:
    ny, nx = profile.shape
    psf = _prepare_psf(psf, ny, nx)
    return fftconvolve(profile, psf, mode="same")


def _prepare_psf(psf: np.ndarray, ny: int, nx: int) -> np.ndarray:
    psf = _pad_match_parity(psf, ny, nx)
    py, px = psf.shape
    if py > ny:
        start = (py - ny) // 2
        psf = psf[start : start + ny, :]
        py = ny
    if px > nx:
        start = (px - nx) // 2
        psf = psf[:, start : start + nx]
        px = nx
    if py < ny or px < nx:
        padded = np.zeros((ny, nx))
        offset_y = max(0, (ny - py) // 2)
        offset_x = max(0, (nx - px) // 2)
        padded[offset_y : offset_y + py, offset_x : offset_x + px] = psf
        psf = padded
    psf = np.array(_center_psf(psf), copy=True)
    psf /= psf.sum()
    return psf


def _pad_match_parity(psf: np.ndarray, ny: int, nx: int) -> np.ndarray:
    pad_y = 1 if psf.shape[0] % 2 == 0 and ny % 2 == 1 else 0
    pad_x = 1 if psf.shape[1] % 2 == 0 and nx % 2 == 1 else 0
    if pad_y or pad_x:
        return np.pad(psf, ((pad_y, 0), (pad_x, 0)), mode="constant")
    return psf


def _resample_spatial(
    profile: np.ndarray,
    pixel_scale: u.Quantity,
    selection: InstrumentSelection,
) -> tuple[np.ndarray, float]:
    ny_high, nx_high = profile.shape
    scale = selection.scale
    weights = _flux_conserving_weights(
        ny_high,
        nx_high,
        scale.spaxels_y,
        scale.spaxels_x,
        pixel_scale,
        pixel_scale,
        scale.spaxel_scale,
        scale.spaxel_scale,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        low = np.matmul(weights.toarray(), profile.reshape(-1, 1))
    low = low.reshape(scale.spaxels_y, scale.spaxels_x)
    area_factor = (
        scale.spaxel_scale.to_value(u.arcsec) ** 2 / pixel_scale.to_value(u.arcsec) ** 2
    )
    low *= area_factor
    return low, area_factor


def _flux_conserving_weights(
    ny_high: int,
    nx_high: int,
    ny_low: int,
    nx_low: int,
    high_scale_x: u.Quantity,
    high_scale_y: u.Quantity,
    low_scale_x: u.Quantity,
    low_scale_y: u.Quantity,
) -> lil_matrix:
    weights = lil_matrix((ny_low * nx_low, ny_high * nx_high), dtype=float)
    dy_high = high_scale_y.to_value(u.arcsec)
    dx_high = high_scale_x.to_value(u.arcsec)
    dy_low = low_scale_y.to_value(u.arcsec)
    dx_low = low_scale_x.to_value(u.arcsec)
    y_high = (np.arange(ny_high) - (ny_high - 1) / 2) * dy_high
    x_high = (np.arange(nx_high) - (nx_high - 1) / 2) * dx_high

    for low_index in range(ny_low * nx_low):
        y_index = low_index // nx_low
        x_index = low_index % nx_low
        y_min = (y_index - (ny_low - 1) / 2 - 0.5) * dy_low
        y_max = y_min + dy_low
        x_min = (x_index - (nx_low - 1) / 2 - 0.5) * dx_low
        x_max = x_min + dx_low
        y_mask = (y_high + dy_high / 2 > y_min) & (y_high - dy_high / 2 < y_max)
        x_mask = (x_high + dx_high / 2 > x_min) & (x_high - dx_high / 2 < x_max)
        for high_y in np.where(y_mask)[0]:
            for high_x in np.where(x_mask)[0]:
                overlap_y = max(
                    0,
                    min(y_max, y_high[high_y] + dy_high / 2)
                    - max(y_min, y_high[high_y] - dy_high / 2),
                )
                overlap_x = max(
                    0,
                    min(x_max, x_high[high_x] + dx_high / 2)
                    - max(x_min, x_high[high_x] - dx_high / 2),
                )
                area = overlap_y * overlap_x
                if area > 0:
                    weights[low_index, high_y * nx_high + high_x] = area / (
                        dy_low * dx_low
                    )
    return weights


def _detector_products(
    instrument: InstrumentDefinition,
    selection: InstrumentSelection,
    grid: SpectralGrid,
    models: Models,
    exposure: Any,
    sky_subtraction: Any,
) -> tuple[Signals, Variances, np.ndarray]:
    model = models.combined
    shape = model.shape
    if not np.isfinite(model.value).all():
        raise ValueError("Target model contains non-finite values.")
    qe = _quantum_efficiency(instrument, grid)
    throughput = 1.0
    for component in selection.optical_components:
        throughput *= component.throughput
    etendue = _etendue(instrument, selection.scale.spaxel_scale)

    target_radiance = model.to(u.W / (u.m**2 * u.sr * u.m))
    sky_radiance = _cube(models.sky, shape)
    thermal_radiance = _cube(models.thermal, shape)
    transmission_cube = _cube(models.transmission, shape)
    qe_cube = _cube(qe, shape)
    photon_energy = _cube(grid.photon_energy, shape)

    target_rate = (
        target_radiance
        * throughput
        * transmission_cube
        * etendue
        * grid.step.to(u.m)
        * qe_cube
        / photon_energy
    ).to(1 / u.s)
    sky_rate = (
        sky_radiance
        * throughput
        * etendue
        * grid.step.to(u.m)
        * qe_cube
        / photon_energy
    ).to(1 / u.s)
    thermal_rate = (
        thermal_radiance * etendue * grid.step.to(u.m) * qe_cube / photon_energy
    ).to(1 / u.s)
    thermal_rate += instrument.detector.light_leak.to_value(u.electron / u.s) / u.s

    target = target_rate * exposure.n_target * exposure.time * u.electron
    sky_signal = sky_rate * exposure.n_target * exposure.time * u.electron
    thermal_signal = thermal_rate * exposure.n_target * exposure.time * u.electron
    dark = (
        np.full(shape, instrument.detector.dark_current.to_value(u.electron / u.s))
        / u.s
        * exposure.n_target
        * exposure.time
        * u.electron
    )
    background = sky_signal + thermal_signal + dark
    total = target + background
    signals = Signals(
        target=target,
        sky=sky_signal,
        thermal=thermal_signal,
        dark=dark,
        background=background,
        total=total,
    )

    raw_read_variance = np.full(
        shape,
        exposure.n_target * instrument.detector.read_noise.to_value(u.electron) ** 2,
    )
    if sky_subtraction.method == "nodding":
        snr_target = target
        sky_weight = 1 + exposure.n_target / exposure.n_sky
        variance_target = target.value * u.electron**2
        variance_sky = sky_weight * sky_signal.value * u.electron**2
        variance_thermal = sky_weight * thermal_signal.value * u.electron**2
        variance_dark = sky_weight * dark.value * u.electron**2
        variance_read = sky_weight * raw_read_variance * u.electron**2
    else:
        sky_mask = sky_subtraction.mask
        snr_target = _subtract_in_field_estimate(target, sky_mask)
        variance_target = (
            _in_field_marginal_variance(target.value, sky_mask) * u.electron**2
        )
        variance_sky = (
            _in_field_marginal_variance(sky_signal.value, sky_mask) * u.electron**2
        )
        variance_thermal = (
            _in_field_marginal_variance(thermal_signal.value, sky_mask) * u.electron**2
        )
        variance_dark = (
            _in_field_marginal_variance(dark.value, sky_mask) * u.electron**2
        )
        variance_read = (
            _in_field_marginal_variance(raw_read_variance, sky_mask) * u.electron**2
        )
    variance_total = (
        variance_target
        + variance_sky
        + variance_thermal
        + variance_dark
        + variance_read
    )
    variances = Variances(
        target=variance_target,
        sky=variance_sky,
        thermal=variance_thermal,
        dark=variance_dark,
        read=variance_read,
        total=variance_total,
    )
    if not np.isfinite(variance_total.value).all():
        raise ValueError("Detector variance contains non-finite values.")
    snr = np.divide(
        snr_target.value,
        np.sqrt(variance_total.value),
        out=np.zeros(shape),
        where=variance_total.value > 0,
    )
    return signals, variances, snr


def _smooth_background(table: Any, grid: SpectralGrid) -> np.ndarray:
    return _smooth_table(table, grid, sigma_divisor=1.0)


def _smooth_table(
    table: Any,
    grid: SpectralGrid,
    *,
    sigma_divisor: float,
) -> np.ndarray:
    spacing = np.mean(np.diff(table.wavelength.to_value(u.micron)))
    sigma_pixels = (
        grid.resolution_fwhm.to_value(u.micron)
        * _FWHM_TO_SIGMA
        / spacing
        / sigma_divisor
    )
    smoothed = ndimage.gaussian_filter1d(
        table.values.value,
        sigma=sigma_pixels,
        mode="reflect",
    )
    interpolate = interp1d(
        table.wavelength.to_value(u.micron),
        smoothed,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolate(grid.wavelength.to_value(u.micron))


def _quantum_efficiency(
    instrument: InstrumentDefinition,
    grid: SpectralGrid,
) -> np.ndarray:
    detector = instrument.detector
    if detector.quantum_efficiency is not None:
        return np.full(len(grid.wavelength), detector.quantum_efficiency)
    table = detector.quantum_efficiency_table
    interpolate = interp1d(
        table.wavelength.to_value(u.micron),
        table.values.value,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolate(grid.wavelength.to_value(u.micron))


def _thermal_background(
    selection: InstrumentSelection,
    wavelength: u.Quantity,
) -> u.Quantity:
    total = np.zeros(len(wavelength)) * u.W / (u.m**2 * u.sr * u.m)
    components = selection.optical_components
    for index, component in enumerate(components):
        if component.emissivity == 0:
            continue
        downstream = 1.0
        for following in components[index + 1 :]:
            downstream *= following.throughput
        total += (
            component.emissivity
            * _blackbody(wavelength.to_value(u.m), component.temperature)
            * downstream
        )
    return total


def _blackbody(wavelength_m: np.ndarray, temperature: u.Quantity) -> u.Quantity:
    radiance = (2 * _H.value * _C.value**2 / wavelength_m**5) / (
        np.exp(_H.value * _C.value / (wavelength_m * _K.value * temperature.value)) - 1
    )
    return radiance * u.W / (u.m**2 * u.sr * u.m)


def _etendue(
    instrument: InstrumentDefinition,
    pixel_scale: u.Quantity,
) -> u.Quantity:
    diameter = 2 * np.sqrt(
        (instrument.telescope.primary_diameter / 2) ** 2
        - (instrument.telescope.central_obscuration / 2) ** 2
    )
    numerical_aperture = 1 / (2 * instrument.telescope.f_number)
    solid_angle = 2 * np.pi * (1 - np.sqrt(1 - numerical_aperture**2)) * u.steradian
    side = (
        instrument.telescope.f_number
        * diameter
        * pixel_scale.to(u.arcsec)
        / _RAD_TO_ARCSEC
    )
    return side * side * solid_angle


def _pixel_omega(pixel_scale: u.Quantity) -> u.Quantity:
    return (pixel_scale.to(u.rad) ** 2).to(u.sr)


def _cube(values: Any, shape: tuple[int, int, int]) -> Any:
    if np.isscalar(values):
        return np.full(shape, values)
    return values[None, None, :] * np.ones((shape[0], shape[1], 1))


def _sample_data(
    signals: Signals,
    exposure: Any,
    read_noise: u.Quantity,
    sky_subtraction: Any,
    n_cubes: int,
    rng: np.random.Generator | None,
) -> u.Quantity:
    rng = np.random.default_rng() if rng is None else rng
    shape = signals.target.shape
    output_shape = shape if n_cubes == 1 else (n_cubes, *shape)
    target_frame_mean = signals.total.to_value(u.electron)
    target_frames = rng.poisson(target_frame_mean, size=output_shape)
    read_sigma = read_noise.to_value(u.electron)
    target_read = rng.normal(
        0.0,
        read_sigma * np.sqrt(exposure.n_target),
        size=output_shape,
    )
    target_data = target_frames + target_read
    if sky_subtraction.method == "in_field":
        sky_mask = sky_subtraction.mask
        if n_cubes == 1:
            sky_estimate = target_data[sky_mask].mean(axis=0)
            return (target_data - sky_estimate[None, None, :]) * u.electron
        sky_estimate = target_data[:, sky_mask, :].mean(axis=1)
        return (target_data - sky_estimate[:, None, None, :]) * u.electron

    sky_scale = exposure.n_target / exposure.n_sky
    sky_frame_mean = signals.background.to_value(u.electron) / sky_scale
    sky_frames = rng.poisson(sky_frame_mean, size=output_shape)
    sky_read = rng.normal(
        0.0,
        read_sigma * np.sqrt(exposure.n_sky),
        size=output_shape,
    )
    return (target_data - sky_scale * (sky_frames + sky_read)) * u.electron


def _reduce_aperture(
    aperture: Any,
    grid: SpectralGrid,
    selection: InstrumentSelection,
    targets: tuple[Any, ...],
    signals: Signals,
    variances: Variances,
    data: u.Quantity | None,
    sky_subtraction: Any,
    exposure: Any,
    read_noise: u.Quantity,
    include_signals: bool,
    include_variances: bool,
) -> ApertureResult:
    mask = _aperture_mask(aperture, grid, selection, targets)
    signal_values = {
        field: getattr(signals, field) for field in Signals.__dataclass_fields__
    }
    if sky_subtraction.method == "in_field":
        signal_values = {
            field: _subtract_in_field_estimate(values, sky_subtraction.mask)
            for field, values in signal_values.items()
        }
        integrated_variances, spectral_variances, map_variances = (
            _in_field_aperture_variance_products(
                mask,
                sky_subtraction.mask,
                signals,
                exposure,
                read_noise,
            )
        )
    else:
        integrated_variances, spectral_variances, map_variances = (
            _independent_aperture_variance_products(mask, variances)
        )

    spectral_signals = _project_signals(signal_values, mask, axis=(0, 1))
    map_signals = _project_signals(signal_values, mask, axis=2)
    target = spectral_signals.target.sum()
    total_variance = integrated_variances.total
    spectral_support = mask.any(axis=(0, 1))
    map_support = mask.any(axis=2)
    spectra = ApertureProjection(
        snr=_projected_snr(
            spectral_signals.target,
            spectral_variances.total,
            spectral_support,
        ),
        signals=spectral_signals if include_signals else None,
        variances=spectral_variances if include_variances else None,
    )
    maps = ApertureProjection(
        snr=_projected_snr(map_signals.target, map_variances.total, map_support),
        signals=map_signals if include_signals else None,
        variances=map_variances if include_variances else None,
    )
    reduced_signals = None
    if include_signals:
        reduced_signals = Signals(
            **{
                field: getattr(spectral_signals, field).sum()
                for field in Signals.__dataclass_fields__
            }
        )
    reduced_variances = integrated_variances if include_variances else None
    reduced_data = None
    if data is not None:
        reduced_data = data[mask].sum() if data.ndim == 3 else data[:, mask].sum(axis=1)
    return ApertureResult(
        name=aperture.name,
        mask=mask,
        snr=target.value / np.sqrt(total_variance.value),
        spectra=spectra,
        maps=maps,
        signals=reduced_signals,
        variances=reduced_variances,
        data=reduced_data,
    )


def _project_signals(
    signal_values: dict[str, u.Quantity],
    mask: np.ndarray,
    *,
    axis: int | tuple[int, ...],
) -> Signals:
    return Signals(
        **{
            field: _masked_sum(values, mask, axis=axis)
            for field, values in signal_values.items()
        }
    )


def _independent_aperture_variance_products(
    mask: np.ndarray,
    variances: Variances,
) -> tuple[Variances, Variances, Variances]:
    spectral = Variances(
        **{
            field: _masked_sum(getattr(variances, field), mask, axis=(0, 1))
            for field in Variances.__dataclass_fields__
        }
    )
    maps = Variances(
        **{
            field: _masked_sum(getattr(variances, field), mask, axis=2)
            for field in Variances.__dataclass_fields__
        }
    )
    integrated = Variances(
        **{
            field: getattr(spectral, field).sum()
            for field in Variances.__dataclass_fields__
        }
    )
    return integrated, spectral, maps


def _masked_sum(
    values: u.Quantity,
    mask: np.ndarray,
    *,
    axis: int | tuple[int, ...],
) -> u.Quantity:
    return np.where(mask, values.value, 0.0).sum(axis=axis) * values.unit


def _projected_snr(
    target: u.Quantity,
    total_variance: u.Quantity,
    support: np.ndarray,
) -> np.ndarray:
    snr = np.zeros(target.shape, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr[support] = target.value[support] / np.sqrt(total_variance.value[support])
    return snr


def _in_field_aperture_variance_products(
    aperture_mask: np.ndarray,
    sky_mask: np.ndarray,
    signals: Signals,
    exposure: Any,
    read_noise: u.Quantity,
) -> tuple[Variances, Variances, Variances]:
    shape = signals.target.shape
    raw_read_variance = np.full(
        shape,
        exposure.n_target * read_noise.to_value(u.electron) ** 2,
    )
    raw = {
        "target": signals.target.value,
        "sky": signals.sky.value,
        "thermal": signals.thermal.value,
        "dark": signals.dark.value,
        "read": raw_read_variance,
    }
    spectral_values = {
        name: _in_field_aperture_variance_spectrum(
            aperture_mask,
            sky_mask,
            values,
        )
        * u.electron**2
        for name, values in raw.items()
    }
    map_values = {
        name: _masked_sum(
            _in_field_marginal_variance(values, sky_mask) * u.electron**2,
            aperture_mask,
            axis=2,
        )
        for name, values in raw.items()
    }
    spectral_values["total"] = sum(
        spectral_values.values(), start=np.zeros(shape[2]) * u.electron**2
    )
    map_values["total"] = sum(
        map_values.values(), start=np.zeros(shape[:2]) * u.electron**2
    )
    spectral = Variances(**spectral_values)
    maps = Variances(**map_values)
    integrated = Variances(
        **{
            field: getattr(spectral, field).sum()
            for field in Variances.__dataclass_fields__
        }
    )
    return integrated, spectral, maps


def _aperture_mask(
    aperture: Any,
    grid: SpectralGrid,
    selection: InstrumentSelection,
    targets: tuple[Any, ...],
) -> np.ndarray:
    shape = (
        selection.scale.spaxels_y,
        selection.scale.spaxels_x,
        len(grid.wavelength),
    )
    if aperture.mask is not None:
        mask = np.asarray(aperture.mask)
        if mask.shape != shape:
            raise ValueError(f"Aperture {aperture.name!r} mask shape must be {shape}.")
        return mask.copy()

    size = aperture.size
    placement = aperture.center if aperture.center is not None else aperture.start
    if placement is None:
        first = targets[0]
        placement = (
            (selection.scale.spaxels_y - 1) // 2,
            (selection.scale.spaxels_x - 1) // 2,
            first.spectrum.wavelength[0],
        )
    coordinates = list(placement)
    if isinstance(coordinates[2], u.Quantity):
        coordinates[2] = _wavelength_pixel(grid.wavelength, coordinates[2])
    bounds: list[tuple[int, int]] = []
    for axis, (coordinate, pixels, length) in enumerate(zip(coordinates, size, shape)):
        if aperture.start is not None:
            start = int(np.rint(coordinate))
            end = start + pixels
        else:
            start, end = _centered_bounds(coordinate, pixels)
        if start < 0 or end > length:
            raise ValueError(f"Aperture {aperture.name!r} extends beyond axis {axis}.")
        bounds.append((start, end))
    mask = np.zeros(shape, dtype=bool)
    mask[
        bounds[0][0] : bounds[0][1],
        bounds[1][0] : bounds[1][1],
        bounds[2][0] : bounds[2][1],
    ] = True
    return mask


def _wavelength_pixel(wavelength: u.Quantity, value: u.Quantity) -> float:
    if not value.isscalar:
        raise TypeError("Aperture wavelength must be a scalar quantity.")
    coordinate = value.to_value(wavelength.unit)
    if not np.isfinite(coordinate):
        raise ValueError("Aperture wavelength must be finite.")
    samples = wavelength.value
    if coordinate < samples[0] or coordinate > samples[-1]:
        raise ValueError("Aperture wavelength is outside the selected disperser.")
    if coordinate == samples[0]:
        return 0.0
    if coordinate == samples[-1]:
        return float(len(samples) - 1)
    right = np.searchsorted(samples, coordinate, side="left")
    return (right - 1) + (coordinate - samples[right - 1]) / (
        samples[right] - samples[right - 1]
    )


def _centered_bounds(center: float, pixels: int) -> tuple[int, int]:
    if pixels % 2 == 1:
        radius = (pixels - 1) // 2
        start = int(np.rint(center)) - radius
        return start, start + pixels
    half = pixels / 2
    start = int(np.rint(center - half)) + 1
    end = int(np.rint(center + half)) + 1
    return start, end


def _in_field_marginal_variance(
    raw_variance: np.ndarray,
    sky_mask: np.ndarray,
) -> np.ndarray:
    sky_count = int(sky_mask.sum())
    estimator_variance = raw_variance[sky_mask].sum(axis=0) / sky_count**2
    covariance = np.zeros_like(raw_variance)
    covariance[sky_mask] = 2 * raw_variance[sky_mask] / sky_count
    return raw_variance + estimator_variance[None, None, :] - covariance


def _subtract_in_field_estimate(
    signal: u.Quantity,
    sky_mask: np.ndarray,
) -> u.Quantity:
    estimate = signal[sky_mask].mean(axis=0)
    return signal - estimate[None, None, :]


def _in_field_aperture_variance_spectrum(
    aperture_mask: np.ndarray,
    sky_mask: np.ndarray,
    raw_variance: np.ndarray,
) -> np.ndarray:
    sky_count = int(sky_mask.sum())
    aperture_count = aperture_mask.sum(axis=(0, 1))
    aperture_variance = np.where(aperture_mask, raw_variance, 0.0).sum(
        axis=(0, 1)
    )
    estimator_variance = raw_variance[sky_mask].sum(axis=0) / sky_count**2
    overlap = aperture_mask & sky_mask[:, :, None]
    covariance = np.where(overlap, raw_variance, 0.0).sum(axis=(0, 1)) / sky_count
    return (
        aperture_variance
        + aperture_count**2 * estimator_variance
        - 2 * aperture_count * covariance
    )
