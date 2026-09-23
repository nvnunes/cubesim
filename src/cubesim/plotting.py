"""Plot immutable CubeSim results without changing calculation state."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cubesim._result import ApertureResult, EtcResult, ModelGrid
from cubesim.diagnostics import PsfStats

__all__ = [
    "plot_aperture_signal_maps",
    "plot_aperture_signal_spectra",
    "plot_aperture_snr",
    "plot_aperture_snr_map",
    "plot_aperture_snr_spectrum",
    "plot_background_models",
    "plot_background_sky",
    "plot_background_thermal",
    "plot_background_transmission",
    "plot_psf",
    "plot_signal_components",
    "plot_signal_dark",
    "plot_signal_sky",
    "plot_signal_target",
    "plot_signal_thermal",
    "plot_signal_total",
    "plot_snr",
    "plot_target_models",
    "plot_target_spatial",
    "plot_target_spectrum",
    "plot_target_velocity",
]

_SIGNAL_FIELDS = ("target", "sky", "thermal", "dark", "total")
_SIGNAL_LABELS = {
    "target": "Target",
    "sky": "Sky",
    "thermal": "Thermal",
    "dark": "Dark",
    "total": "Total",
}
_SIGNAL_COLORS = {
    "target": "tab:green",
    "sky": "tab:blue",
    "thermal": "tab:red",
    "dark": "black",
    "total": "tab:orange",
}
_TARGET_SPECTRUM_UNIT = u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.AA)
_APERTURE_COLORS = tuple(f"C{index}" for index in range(10))
_ApertureSelector = int | str | ApertureResult
_ApertureSelection = _ApertureSelector | Sequence[_ApertureSelector]
_ApertureMapOverlay = tuple[str, str, np.ndarray]
_ColorbarRange = tuple[float, float]
_YRange = tuple[float, float]


def plot_psf(
    result: EtcResult,
    *,
    title: str | None = None,
    stats: PsfStats | None = None,
    radius: u.Quantity | None = None,
    cbar_range: _ColorbarRange | None = None,
) -> Figure:
    """Plot the normalized PSF snapshot used by a calculation.

    Parameters
    ----------
    result
        Completed CubeSim result containing a configured PSF.
    title
        Figure title. Defaults to ``"PSF"``.
    stats
        Optional PSF measurements to include in the title.
    radius
        Optional positive angular display radius. The stored PSF is not cropped.
    cbar_range
        Optional lower and upper colorbar limits in log relative intensity.
    """

    result = _require_result(result)
    if not hasattr(result, "psf"):
        raise ValueError("plot_psf requires a result produced with a configured PSF.")
    if stats is not None and not isinstance(stats, PsfStats):
        raise TypeError("stats must be a PsfStats result from diagnostics.psf_stats().")
    radius_mas = None
    if radius is not None:
        if not isinstance(radius, u.Quantity) or not radius.isscalar:
            raise TypeError("radius must be a scalar angular Quantity.")
        try:
            radius_mas = radius.to_value(u.mas)
        except u.UnitConversionError as exc:
            raise ValueError("radius must have angular units.") from exc
        if not np.isfinite(radius_mas) or radius_mas <= 0:
            raise ValueError("radius must be finite and positive.")

    figure, axis = plt.subplots(figsize=(6, 6))
    psf = result.psf.data
    peak = psf.max()
    with np.errstate(divide="ignore"):
        relative = np.log10(np.clip(psf / peak, 1e-10, None))
    extent = _angular_extent(psf.shape, result.psf.pixel_scale)
    norm = _colorbar_norm(cbar_range)
    if norm is None:
        norm = Normalize(vmin=max(-10.0, float(relative.min())), vmax=0.0)
    image = axis.imshow(
        relative,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=norm,
    )
    levels = np.asarray([-3.0, -2.5, -2.0, -1.5, -1.0])
    levels = levels[(levels > relative.min()) & (levels < relative.max())]
    if levels.size:
        ny, nx = psf.shape
        x = np.linspace(extent[0], extent[1], nx)
        y = np.linspace(extent[2], extent[3], ny)
        contours = axis.contour(
            x,
            y,
            relative,
            levels=levels,
            colors="white",
            linewidths=0.5,
        )
        axis.clabel(contours, contours.levels, inline=True, fmt="%.1f", fontsize=8)
    _add_colorbar(figure, axis, image, "Log relative intensity")
    axis.set(xlabel="X offset [mas]", ylabel="Y offset [mas]")
    axis.grid(color="white", alpha=0.2)
    if radius_mas is not None:
        axis.set_xlim(-radius_mas, radius_mas)
        axis.set_ylim(-radius_mas, radius_mas)
    heading = "PSF" if title is None else title
    if stats is not None:
        heading = _psf_stats_title(heading, stats)
    axis.set_title(heading)
    figure.tight_layout()
    return figure


def _psf_stats_title(title: str, stats: PsfStats) -> str:
    parts = []
    if stats.sr is not None:
        parts.append(f"SR: {stats.sr.to_value(u.one):.2f}")
    fwhm = stats.fwhm.to_value(u.mas)
    parts.append(f"FWHM: {fwhm:.0f} mas" if np.isfinite(fwhm) else "FWHM: unavailable")
    widths = np.atleast_1d(stats.ee_apertures.to_value(u.mas))
    energies = np.atleast_1d(stats.ee.to_value(u.one))
    if widths.shape != energies.shape:
        raise ValueError("stats EE values must match their aperture widths.")
    if widths.size == 0:
        raise ValueError("stats must contain at least one EE aperture.")
    ee_label = f"EE({widths[0]:g} mas): {energies[0]:.2f}"
    return ", ".join((title, *parts, ee_label))


def plot_target_models(
    result: EtcResult,
    target: int = 0,
    *,
    high_res: bool = False,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot spatial, spectral, and optional velocity models for one target."""

    result, model = _target_model(
        result, target, high_res, function_name="plot_target_models"
    )
    norm = _colorbar_norm(cbar_range)
    count = 3 if model.velocity is not None else 2
    figure, axes = plt.subplots(count, 1, figsize=(6, 5 * count), squeeze=False)
    _draw_target_spatial(
        figure, axes[0, 0], result, model, high_res, norm=norm
    )
    _draw_target_spectrum(axes[1, 0], model, high_res, y_range=y_range)
    if model.velocity is not None:
        _draw_target_velocity(
            figure, axes[2, 0], result, model, high_res, norm=norm
        )
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_target_spatial(
    result: EtcResult,
    target: int = 0,
    *,
    high_res: bool = False,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
) -> Figure:
    """Plot the spatial model for one target with optional colorbar limits."""

    result, model = _target_model(
        result, target, high_res, function_name="plot_target_spatial"
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    _draw_target_spatial(
        figure,
        axis,
        result,
        model,
        high_res,
        norm=_colorbar_norm(cbar_range),
    )
    _set_title(axis, title, "Target spatial model")
    figure.tight_layout()
    return figure


def plot_target_spectrum(
    result: EtcResult,
    target: int = 0,
    *,
    high_res: bool = False,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot the spectrum for one target."""

    _, model = _target_model(
        result, target, high_res, function_name="plot_target_spectrum"
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    _draw_target_spectrum(axis, model, high_res, y_range=y_range)
    _set_title(axis, title, "Target spectrum")
    figure.tight_layout()
    return figure


def plot_target_velocity(
    result: EtcResult,
    target: int = 0,
    *,
    high_res: bool = False,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
) -> Figure:
    """Plot the velocity map for one target with optional colorbar limits."""

    result, model = _target_model(
        result, target, high_res, function_name="plot_target_velocity"
    )
    if model.velocity is None:
        raise ValueError("The selected target has no velocity model.")
    figure, axis = plt.subplots(figsize=(6, 5))
    _draw_target_velocity(
        figure,
        axis,
        result,
        model,
        high_res,
        norm=_colorbar_norm(cbar_range),
    )
    _set_title(axis, title, "Target velocity")
    figure.tight_layout()
    return figure


def plot_background_models(
    result: EtcResult,
    *,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot atmospheric transmission, sky, and thermal model spectra."""

    result = _require_group(result, "models", "plot_background_models", "models")
    figure, axes = plt.subplots(3, 1, figsize=(6, 15))
    _draw_background_transmission(axes[0], result, y_range=y_range)
    _draw_background_radiance(axes[1], result, "sky", y_range=y_range)
    _draw_background_radiance(axes[2], result, "thermal", y_range=y_range)
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_background_transmission(
    result: EtcResult,
    *,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot atmospheric transmission on the detector wavelength grid."""

    result = _require_group(
        result, "models", "plot_background_transmission", "models"
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    _draw_background_transmission(axis, result, y_range=y_range)
    _set_title(axis, title, "Atmospheric transmission")
    figure.tight_layout()
    return figure


def plot_background_sky(
    result: EtcResult,
    *,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot the atmospheric sky-radiance model."""

    return _plot_background_radiance(
        result, "sky", title, "plot_background_sky", y_range
    )


def plot_background_thermal(
    result: EtcResult,
    *,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot the instrument thermal-radiance model."""

    return _plot_background_radiance(
        result, "thermal", title, "plot_background_thermal", y_range
    )


def plot_signal_components(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot detector signal components at a position or wavelength selection.

    A scalar ``position=(y, x)`` plots spectra at one spaxel. Inclusive
    ``position=((y_min, y_max), (x_min, x_max))`` bounds instead sum that
    detector region into spectra. A scalar wavelength index or Quantity plots
    one detector channel, while a two-element spectral Quantity sums the
    inclusive wavelength interval into maps. When ``position`` and a
    wavelength range are supplied together, the position still selects the
    spectrum and the wavelength range limits its displayed interval.

    Map panels share one logarithmic normalization spanning at most six
    decades so their absolute signal levels can be compared directly.
    ``apertures`` may select registered apertures by zero-based index, name,
    or retained object. Map output outlines their spatial support; spectrum
    output marks the start and end of their wavelength support. When neither
    ``position`` nor ``wavelength`` is supplied, selected apertures with the
    same contiguous spectral support select that wavelength range and produce
    a map. Otherwise, a wavelength range must be supplied explicitly.
    """

    result = _require_result(result)
    selected_apertures = _resolve_apertures(result, apertures)
    selection = _plot_selection(
        result,
        position,
        wavelength,
        selected_apertures,
    )
    if selection[0].startswith("position"):
        _reject_spectrum_colorbar_range(cbar_range)
        figure, axis = plt.subplots(figsize=(8, 5))
        _draw_selected_signal_spectra(
            axis,
            result,
            selection,
            _SIGNAL_FIELDS,
            y_range=y_range,
        )
        _overlay_aperture_ranges(axis, result, selected_apertures)
        if title is not None:
            figure.suptitle(title)
        figure.tight_layout()
        return figure
    else:
        _reject_map_y_range(y_range)
        signals = {
            field: _signal_map_values(result, field, selection)
            for field in _SIGNAL_FIELDS
        }
        panel_titles = {
            field: (
                f"{_SIGNAL_LABELS[field]} at "
                f"{_wavelength_selection_label(result, selection)}"
            )
            for field in _SIGNAL_FIELDS
        }
        return _plot_signal_maps(
            signals,
            panel_titles,
            title=title,
            cbar_range=cbar_range,
            aperture_overlays=_aperture_map_overlays(
                result,
                selected_apertures,
                selection,
            ),
        )


def plot_signal_target(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot target signal, with optional registered-aperture overlays."""

    return _plot_signal_component(
        result,
        "target",
        position,
        wavelength,
        apertures,
        title,
        cbar_range,
        y_range,
    )


def plot_signal_sky(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot sky signal, with optional registered-aperture overlays."""

    return _plot_signal_component(
        result, "sky", position, wavelength, apertures, title, cbar_range, y_range
    )


def plot_signal_thermal(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot thermal signal, with optional registered-aperture overlays."""

    return _plot_signal_component(
        result,
        "thermal",
        position,
        wavelength,
        apertures,
        title,
        cbar_range,
        y_range,
    )


def plot_signal_dark(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot dark signal, with optional registered-aperture overlays."""

    return _plot_signal_component(
        result, "dark", position, wavelength, apertures, title, cbar_range, y_range
    )


def plot_signal_total(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot total signal, with optional registered-aperture overlays."""

    return _plot_signal_component(
        result, "total", position, wavelength, apertures, title, cbar_range, y_range
    )


def plot_snr(
    result: EtcResult,
    *,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    wavelength: int | u.Quantity | None = None,
    apertures: Sequence[int | str | ApertureResult] | None = None,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot S/N for a detector position or wavelength selection.

    Scalar selectors use the always-present S/N cube. A position may be paired
    with a wavelength range to limit the displayed S/N spectrum. Spatial or
    wavelength reductions recompute S/N after summing the selected target
    signal and propagating its variance.

    ``apertures`` may select registered apertures by zero-based index, name,
    or retained object. Map output outlines their spatial support; spectrum
    output marks the start and end of their wavelength support. When neither
    ``position`` nor ``wavelength`` is supplied, selected apertures with the
    same contiguous spectral support select that wavelength range and produce
    a map. Otherwise, a wavelength range must be supplied explicitly.
    """

    result = _require_result(result)
    selected_apertures = _resolve_apertures(result, apertures)
    selection = _plot_selection(
        result,
        position,
        wavelength,
        selected_apertures,
    )
    if selection[0].startswith("position"):
        _reject_spectrum_colorbar_range(cbar_range)
        norm = None
    else:
        _reject_map_y_range(y_range)
        norm = _colorbar_norm(cbar_range)
    figure, axis = plt.subplots(figsize=(8, 5))
    if selection[0] == "position":
        y, x = selection[1]
        axis.plot(_wavelength(result), result.snr[y, x, :])
        axis.set(
            xlabel="Wavelength [micron]",
            ylabel="S/N",
            title=f"S/N at position (y={y}, x={x})",
        )
        axis.grid(True, alpha=0.3)
    elif selection[0] == "wavelength":
        index = selection[1]
        _draw_detector_image(
            figure,
            axis,
            result.snr[:, :, index],
            "S/N",
            f"S/N at {_wavelength_label(result, index)}",
            norm=norm,
        )
    elif selection[0].startswith("position"):
        position_selection, wavelength_indices = _spectrum_selection_parts(selection)
        if position_selection[0] == "position":
            y, x = position_selection[1]
            values = result.snr[y, x, :]
        else:
            values = _snr_spectrum_for_position_range(
                result,
                position_selection[1],
            )
        wavelength_values = _wavelength(result)
        if wavelength_indices is not None:
            wavelength_values = wavelength_values[wavelength_indices]
            values = values[wavelength_indices]
        axis.plot(wavelength_values, values)
        axis.set_xlim(wavelength_values[0], wavelength_values[-1])
        axis.set(
            xlabel="Wavelength [micron]",
            ylabel="S/N",
            title=f"S/N for {_position_label(position_selection)}",
        )
        axis.grid(True, alpha=0.3)
    else:
        values = _snr_map_for_wavelength_range(result, selection[1])
        _draw_detector_image(
            figure,
            axis,
            values,
            "S/N",
            f"S/N for {_wavelength_selection_label(result, selection)}",
            norm=norm,
        )
    if selection[0].startswith("position"):
        _set_y_range(axis, y_range)
    if selection[0].startswith("position"):
        _overlay_aperture_ranges(axis, result, selected_apertures)
    else:
        _overlay_aperture_outlines(
            axis,
            _aperture_map_overlays(result, selected_apertures, selection),
        )
    if title is not None:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_aperture_signal_spectra(
    result: EtcResult,
    aperture: int | str | ApertureResult = 0,
    *,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot every available signal-component spectrum for one aperture."""

    result, selected = _aperture(result, aperture)
    figure, axis = plt.subplots(figsize=(8, 5))
    _draw_aperture_signal_spectra(axis, result, selected, y_range=y_range)
    _set_title(axis, title, f"{selected.name} signal spectra")
    figure.tight_layout()
    return figure


def plot_aperture_signal_maps(
    result: EtcResult,
    aperture: int | str | ApertureResult = 0,
    *,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
) -> Figure:
    """Plot every available signal-component map for one aperture.

    Panels share one logarithmic normalization spanning at most six decades
    so their absolute signal levels can be compared directly.
    """

    _, selected = _aperture(result, aperture)
    return _plot_aperture_signal_maps(selected, title, cbar_range)


def plot_aperture_snr(
    result: EtcResult,
    aperture: _ApertureSelection = 0,
    *,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot S/N spectra and maps for one or more apertures.

    Multiple apertures form a vertical stack and share one map normalization.
    """

    result, selected = _aperture_selection(result, aperture)
    norm = _aperture_snr_norm(selected, cbar_range)
    figure, axes = plt.subplots(
        len(selected),
        2,
        figsize=(14, 5 * len(selected)),
        squeeze=False,
    )
    for row, item in zip(axes, selected, strict=True):
        _draw_aperture_snr_spectrum(
            row[0], result, item, y_range=y_range
        )
        _draw_aperture_snr_map(figure, row[1], item, norm=norm)
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_aperture_snr_spectrum(
    result: EtcResult,
    aperture: _ApertureSelection = 0,
    *,
    title: str | None = None,
    y_range: _YRange | None = None,
) -> Figure:
    """Plot spatially collapsed S/N spectra for one or more apertures."""

    result, selected = _aperture_selection(result, aperture)
    figure, axes = plt.subplots(
        len(selected),
        1,
        figsize=(8, 5 * len(selected)),
        squeeze=False,
    )
    for axis, item in zip(axes.flat, selected, strict=True):
        _draw_aperture_snr_spectrum(axis, result, item, y_range=y_range)
    if title is not None:
        if len(selected) == 1:
            axes[0, 0].set_title(title)
        else:
            figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_aperture_snr_map(
    result: EtcResult,
    aperture: _ApertureSelection = 0,
    *,
    title: str | None = None,
    cbar_range: _ColorbarRange | None = None,
) -> Figure:
    """Plot spectrally collapsed S/N maps for one or more apertures.

    Multiple aperture maps form a vertical stack and share one normalization.
    """

    _, selected = _aperture_selection(result, aperture)
    norm = _aperture_snr_norm(selected, cbar_range)
    figure, axes = plt.subplots(
        len(selected),
        1,
        figsize=(6, 5 * len(selected)),
        squeeze=False,
    )
    for axis, item in zip(axes.flat, selected, strict=True):
        _draw_aperture_snr_map(figure, axis, item, norm=norm)
    if title is not None:
        if len(selected) == 1:
            axes[0, 0].set_title(title)
        else:
            figure.suptitle(title)
    figure.tight_layout()
    return figure


def _plot_background_radiance(
    result: EtcResult,
    field: str,
    title: str | None,
    function_name: str,
    y_range: _YRange | None,
) -> Figure:
    result = _require_group(result, "models", function_name, "models")
    figure, axis = plt.subplots(figsize=(8, 5))
    _draw_background_radiance(axis, result, field, y_range=y_range)
    _set_title(axis, title, f"{field.title()} radiance")
    figure.tight_layout()
    return figure


def _plot_signal_component(
    result: EtcResult,
    field: str,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None,
    wavelength: int | u.Quantity | None,
    apertures: Sequence[_ApertureSelector] | None,
    title: str | None,
    cbar_range: _ColorbarRange | None,
    y_range: _YRange | None,
) -> Figure:
    result = _require_result(result)
    selected_apertures = _resolve_apertures(result, apertures)
    selection = _plot_selection(
        result,
        position,
        wavelength,
        selected_apertures,
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    if selection[0].startswith("position"):
        _reject_spectrum_colorbar_range(cbar_range)
        _draw_selected_signal_spectra(
            axis,
            result,
            selection,
            (field,),
            y_range=y_range,
        )
    else:
        _reject_map_y_range(y_range)
        values = _signal_map_values(result, field, selection)
        _draw_signal_map(
            figure,
            axis,
            values,
            f"{_SIGNAL_LABELS[field]} at "
            f"{_wavelength_selection_label(result, selection)}",
            norm=_colorbar_norm(cbar_range, logarithmic=True),
        )
    if selection[0].startswith("position"):
        _overlay_aperture_ranges(axis, result, selected_apertures)
    else:
        _overlay_aperture_outlines(
            axis,
            _aperture_map_overlays(result, selected_apertures, selection),
        )
    if title is not None:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def _draw_target_spatial(
    figure: Figure,
    axis: Axes,
    result: EtcResult,
    model: ModelGrid,
    high_res: bool,
    *,
    norm: Normalize | None = None,
) -> None:
    if high_res:
        pixel_scale = (
            result.psf.pixel_scale
            if hasattr(result, "psf")
            else result.options.spaxel_scale
        )
        image = axis.imshow(
            model.spatial,
            origin="lower",
            extent=_angular_extent(model.spatial.shape, pixel_scale),
            cmap="viridis",
            norm=norm,
        )
        axis.set(xlabel="X offset [mas]", ylabel="Y offset [mas]")
    else:
        image = axis.imshow(
            model.spatial,
            origin="lower",
            extent=_detector_extent(model.spatial.shape),
            cmap="viridis",
            norm=norm,
        )
        _decorate_detector_axes(axis, model.spatial.shape)
    _add_colorbar(figure, axis, image, "Relative flux")
    axis.set_title("Target spatial model")


def _draw_target_spectrum(
    axis: Axes,
    model: ModelGrid,
    high_res: bool,
    *,
    y_range: _YRange | None = None,
) -> None:
    wavelength = model.wavelength.to_value(u.micron)
    spectrum = model.spectrum.to(_TARGET_SPECTRUM_UNIT)
    plotted = [spectrum.value]
    axis.plot(wavelength, spectrum.value, label="Intrinsic")
    if high_res and model.spectrum_convolved is not None:
        spectrum_convolved = model.spectrum_convolved.to(_TARGET_SPECTRUM_UNIT)
        plotted.append(spectrum_convolved.value)
        axis.plot(
            wavelength,
            spectrum_convolved.value,
            label="LSF convolved",
        )
        axis.legend()
    axis.set_yscale("log", nonpositive="mask")
    if y_range is None:
        positive = np.concatenate([values[values > 0] for values in plotted])
        peak = positive.max()
        lower = max(positive.min(), peak * 1e-6)
        if lower == peak:
            lower = peak / 10
        axis.set_ylim(lower, peak * 2)
    else:
        _set_y_range(axis, y_range, logarithmic=True)
    axis.set(
        xlabel="Wavelength [micron]",
        ylabel=_quantity_label("Spectral surface brightness", spectrum),
        title="Target spectrum",
    )
    axis.grid(True, alpha=0.3)


def _draw_target_velocity(
    figure: Figure,
    axis: Axes,
    result: EtcResult,
    model: ModelGrid,
    high_res: bool,
    *,
    norm: Normalize | None = None,
) -> None:
    velocity = model.velocity.to(u.km / u.s)
    if high_res:
        pixel_scale = (
            result.psf.pixel_scale
            if hasattr(result, "psf")
            else result.options.spaxel_scale
        )
        extent = _angular_extent(velocity.shape, pixel_scale)
        image = axis.imshow(
            velocity.value,
            origin="lower",
            extent=extent,
            cmap="viridis",
            norm=norm,
        )
        axis.set(xlabel="X offset [mas]", ylabel="Y offset [mas]")
    else:
        image = axis.imshow(
            velocity.value,
            origin="lower",
            extent=_detector_extent(velocity.shape),
            cmap="viridis",
            norm=norm,
        )
        _decorate_detector_axes(axis, velocity.shape)
    _add_colorbar(figure, axis, image, "Velocity [km/s]")
    axis.set_title("Target velocity")


def _draw_background_transmission(
    axis: Axes,
    result: EtcResult,
    *,
    y_range: _YRange | None = None,
) -> None:
    axis.plot(_wavelength(result), result.models.transmission)
    axis.set(
        xlabel="Wavelength [micron]",
        ylabel="Transmission",
        title="Atmospheric transmission",
    )
    _set_y_range(axis, y_range)
    axis.grid(True, alpha=0.3)


def _draw_background_radiance(
    axis: Axes,
    result: EtcResult,
    field: str,
    *,
    y_range: _YRange | None = None,
) -> None:
    values = getattr(result.models, field)
    axis.plot(_wavelength(result), values.value)
    logarithmic = bool(np.all(values.value > 0))
    if logarithmic:
        axis.set_yscale("log")
    axis.set(
        xlabel="Wavelength [micron]",
        ylabel=_quantity_label("Radiance", values),
        title=f"{field.title()} radiance",
    )
    _set_y_range(axis, y_range, logarithmic=logarithmic)
    axis.grid(True, alpha=0.3)


def _draw_selected_signal_spectra(
    axis: Axes,
    result: EtcResult,
    selection: tuple[str, Any],
    fields: tuple[str, ...],
    *,
    y_range: _YRange | None = None,
) -> None:
    wavelength = _wavelength(result)
    position_selection, wavelength_indices = _spectrum_selection_parts(selection)
    if wavelength_indices is not None:
        wavelength = wavelength[wavelength_indices]
    signals = {
        field: _signal_spectrum_values(result, field, selection) for field in fields
    }
    _draw_signal_spectra(
        axis,
        wavelength,
        signals,
        f"Signals at {_position_label(position_selection)}",
        y_range=y_range,
    )


def _draw_signal_spectra(
    axis: Axes,
    wavelength: np.ndarray,
    signals: dict[str, u.Quantity],
    title: str,
    *,
    support: np.ndarray | None = None,
    y_range: _YRange | None = None,
) -> None:
    plotted = {}
    for field, values in signals.items():
        plotted[field] = (
            values.value
            if support is None
            else np.ma.masked_where(~support, values.value)
        )
        axis.plot(
            wavelength,
            plotted[field],
            label=_SIGNAL_LABELS[field],
            color=_SIGNAL_COLORS[field],
        )
    axis.set(
        xlabel="Wavelength [micron]",
        ylabel=_quantity_label("Signal", next(iter(signals.values()))),
        title=title,
    )
    axis.set_xlim(wavelength[0], wavelength[-1])
    axis.set_yscale("log", nonpositive="mask")
    if y_range is None:
        positive = np.concatenate(
            [_positive_values(values) for values in plotted.values()]
        )
        if positive.size:
            peak = positive.max()
            background = [
                _positive_values(plotted[field])
                for field in ("sky", "thermal", "dark")
                if field in plotted
            ]
            if background:
                lower = np.concatenate(background).min() / 2
            else:
                lower = max(positive.min(), peak * 1e-6)
            if lower == peak:
                lower = peak / 10
            axis.set_ylim(lower, peak * 2)
    else:
        _set_y_range(axis, y_range, logarithmic=True)
    axis.grid(True, alpha=0.3)
    if len(signals) > 1:
        axis.legend()


def _draw_signal_map(
    figure: Figure,
    axis: Axes,
    values: u.Quantity,
    title: str,
    *,
    norm: Normalize | None = None,
    support: np.ndarray | None = None,
) -> None:
    plotted = (
        values.value
        if support is None
        else np.ma.masked_where(~support, values.value)
    )
    _draw_detector_image(
        figure,
        axis,
        plotted,
        _quantity_label("Signal", values),
        title,
        norm=norm,
    )
    if support is not None:
        _set_aperture_spatial_limits(axis, support)


def _draw_aperture_signal_spectra(
    axis: Axes,
    result: EtcResult,
    aperture: ApertureResult,
    *,
    y_range: _YRange | None = None,
) -> None:
    support = aperture.mask.any(axis=(0, 1))
    signals = {
        field: getattr(aperture.spectra.signals, field) for field in _SIGNAL_FIELDS
    }
    _draw_signal_spectra(
        axis,
        _wavelength(result),
        signals,
        f"{aperture.name} signal spectra",
        support=support,
        y_range=y_range,
    )


def _draw_aperture_snr_spectrum(
    axis: Axes,
    result: EtcResult,
    aperture: ApertureResult,
    *,
    y_range: _YRange | None = None,
) -> None:
    support = aperture.mask.any(axis=(0, 1))
    axis.plot(
        _wavelength(result),
        np.ma.masked_where(~support, aperture.spectra.snr),
    )
    axis.set(
        xlabel="Wavelength [micron]",
        ylabel="S/N",
        title=f"{aperture.name} S/N spectrum",
    )
    _set_y_range(axis, y_range)
    axis.grid(True, alpha=0.3)


def _draw_aperture_snr_map(
    figure: Figure,
    axis: Axes,
    aperture: ApertureResult,
    *,
    norm: Normalize | None = None,
) -> None:
    support = aperture.mask.any(axis=2)
    _draw_detector_image(
        figure,
        axis,
        np.ma.masked_where(~support, aperture.maps.snr),
        "S/N",
        f"{aperture.name} S/N map",
        norm=norm,
    )
    _set_aperture_spatial_limits(axis, support)


def _plot_aperture_signal_maps(
    aperture: ApertureResult,
    title: str | None,
    cbar_range: _ColorbarRange | None,
) -> Figure:
    signals = {
        field: getattr(aperture.maps.signals, field) for field in _SIGNAL_FIELDS
    }
    panel_titles = {field: _SIGNAL_LABELS[field] for field in _SIGNAL_FIELDS}
    return _plot_signal_maps(
        signals,
        panel_titles,
        title=title,
        support=aperture.mask.any(axis=2),
        cbar_range=cbar_range,
    )


def _plot_signal_maps(
    signals: dict[str, u.Quantity],
    panel_titles: dict[str, str],
    *,
    title: str | None,
    cbar_range: _ColorbarRange | None = None,
    support: np.ndarray | None = None,
    aperture_overlays: tuple[_ApertureMapOverlay, ...] = (),
) -> Figure:
    count = len(signals)
    plotted = [
        values.value
        if support is None
        else np.ma.masked_where(~support, values.value)
        for values in signals.values()
    ]
    norm = _shared_signal_norm(plotted, cbar_range)
    figure, axes = plt.subplots(
        count,
        1,
        figsize=(6, 5 * count),
        squeeze=False,
    )
    for index, (axis, field) in enumerate(
        zip(axes.flat, signals, strict=True)
    ):
        _draw_signal_map(
            figure,
            axis,
            signals[field],
            panel_titles[field],
            norm=norm,
            support=support,
        )
        _overlay_aperture_outlines(
            axis,
            aperture_overlays,
            show_legend=index == 0,
        )
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def _resolve_apertures(
    result: EtcResult,
    selectors: Sequence[_ApertureSelector] | None,
) -> tuple[ApertureResult, ...]:
    if selectors is None:
        return ()
    if isinstance(selectors, (str, bytes)) or not isinstance(selectors, Sequence):
        raise TypeError(
            "apertures must be a sequence of zero-based integer indices, "
            "names, or ApertureResult objects."
        )
    selected_ids = {
        id(_aperture(result, selector)[1])
        for selector in selectors
    }
    return tuple(
        aperture for aperture in result.apertures if id(aperture) in selected_ids
    )


def _aperture_map_overlays(
    result: EtcResult,
    apertures: tuple[ApertureResult, ...],
    selection: tuple[str, Any],
) -> tuple[_ApertureMapOverlay, ...]:
    if selection[0] == "wavelength":
        indices = np.asarray([selection[1]])
    else:
        indices = selection[1]
    overlays = []
    for aperture in apertures:
        support = aperture.mask[:, :, indices].any(axis=2)
        if support.any():
            overlays.append(
                (
                    aperture.name,
                    _aperture_color(result, aperture),
                    support,
                )
            )
    return tuple(overlays)


def _overlay_aperture_ranges(
    axis: Axes,
    result: EtcResult,
    apertures: tuple[ApertureResult, ...],
) -> None:
    handles = []
    lower_limit, upper_limit = axis.get_xlim()
    wavelength = _wavelength(result)
    for aperture in apertures:
        indices = np.flatnonzero(aperture.mask.any(axis=(0, 1)))
        lower = wavelength[indices[0]]
        upper = wavelength[indices[-1]]
        if upper < lower_limit or lower > upper_limit:
            continue
        color = _aperture_color(result, aperture)
        drawn = False
        if lower_limit <= lower <= upper_limit:
            axis.axvline(
                lower,
                color=color,
                linestyle="-",
                linewidth=1.75,
            )
            drawn = True
        if lower_limit <= upper <= upper_limit:
            axis.axvline(
                upper,
                color=color,
                linestyle="-",
                linewidth=1.75,
            )
            drawn = True
        if drawn:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    linestyle="-",
                    linewidth=1.75,
                    label=aperture.name,
                )
            )
    _add_aperture_legend(axis, handles)


def _overlay_aperture_outlines(
    axis: Axes,
    overlays: tuple[_ApertureMapOverlay, ...],
    *,
    show_legend: bool = True,
) -> None:
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    handles = []
    for name, color, support in overlays:
        segments = _mask_boundary_segments(support)
        axis.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=3.5,
                linestyles="-",
                capstyle="butt",
                joinstyle="miter",
                zorder=3,
            )
        )
        axis.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=1.75,
                linestyles="-",
                capstyle="butt",
                joinstyle="miter",
                zorder=4,
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle="-",
                linewidth=1.75,
                label=name,
            )
        )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    if show_legend:
        _add_aperture_legend(axis, handles)


def _add_aperture_legend(axis: Axes, aperture_handles: list[Line2D]) -> None:
    if not aperture_handles:
        return
    if axis.get_legend() is None:
        handles, labels = [], []
    else:
        handles, labels = axis.get_legend_handles_labels()
    aperture_labels = [handle.get_label() for handle in aperture_handles]
    axis.legend(
        [*handles, *aperture_handles],
        [*labels, *aperture_labels],
    )


def _aperture_color(result: EtcResult, aperture: ApertureResult) -> str:
    index = _aperture_index(result, aperture)
    return _APERTURE_COLORS[index % len(_APERTURE_COLORS)]


def _aperture_index(result: EtcResult, aperture: ApertureResult) -> int:
    index = next(
        index
        for index, candidate in enumerate(result.apertures)
        if aperture is candidate
    )
    return index


def _mask_boundary_segments(support: np.ndarray) -> list[list[tuple[int, int]]]:
    ny, nx = support.shape
    segments = []
    for y, x in np.argwhere(support):
        if y == 0 or not support[y - 1, x]:
            segments.append([(x, y), (x + 1, y)])
        if y == ny - 1 or not support[y + 1, x]:
            segments.append([(x, y + 1), (x + 1, y + 1)])
        if x == 0 or not support[y, x - 1]:
            segments.append([(x, y), (x, y + 1)])
        if x == nx - 1 or not support[y, x + 1]:
            segments.append([(x + 1, y), (x + 1, y + 1)])
    return segments


def _set_aperture_spatial_limits(axis: Axes, support: np.ndarray) -> None:
    y, x = np.nonzero(support)
    axis.set_xlim(x.min(), x.max() + 1)
    axis.set_ylim(y.min(), y.max() + 1)


def _draw_detector_image(
    figure: Figure,
    axis: Axes,
    values: np.ndarray,
    colorbar_label: str,
    title: str,
    *,
    norm: Normalize | None = None,
) -> None:
    image = axis.imshow(
        values,
        origin="lower",
        extent=_detector_extent(values.shape),
        cmap="viridis",
        norm=norm,
    )
    _add_colorbar(figure, axis, image, colorbar_label)
    _decorate_detector_axes(axis, values.shape)
    axis.set_title(title)


def _shared_signal_norm(
    values: list[np.ndarray],
    cbar_range: _ColorbarRange | None,
) -> Normalize:
    explicit = _colorbar_norm(cbar_range, logarithmic=True)
    if explicit is not None:
        return explicit
    positive = np.concatenate([_positive_values(value) for value in values])
    if positive.size == 0:
        return Normalize(vmin=0, vmax=1, clip=True)
    maximum = float(positive.max())
    minimum = max(float(positive.min()), maximum * 1e-6)
    if minimum == maximum:
        minimum = maximum / 10
    return LogNorm(vmin=minimum, vmax=maximum, clip=True)


def _aperture_snr_norm(
    apertures: tuple[ApertureResult, ...],
    cbar_range: _ColorbarRange | None,
) -> Normalize | None:
    explicit = _colorbar_norm(cbar_range)
    if explicit is not None:
        return explicit
    if len(apertures) == 1:
        return None
    values = [
        np.ma.masked_where(
            ~aperture.mask.any(axis=2),
            aperture.maps.snr,
        )
        for aperture in apertures
    ]
    return _shared_linear_norm(values)


def _shared_linear_norm(values: list[np.ndarray]) -> Normalize:
    finite_values = []
    for value in values:
        unmasked = np.ma.asarray(value).compressed()
        finite = unmasked[np.isfinite(unmasked)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return Normalize(vmin=0, vmax=1, clip=True)
    finite = np.concatenate(finite_values)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum == maximum:
        delta = abs(minimum) * 0.05 or 1.0
        minimum -= delta
        maximum += delta
    return Normalize(vmin=minimum, vmax=maximum, clip=True)


def _colorbar_norm(
    cbar_range: _ColorbarRange | None,
    *,
    logarithmic: bool = False,
) -> Normalize | None:
    limits = _plot_range(cbar_range, "cbar_range", logarithmic=logarithmic)
    if limits is None:
        return None
    norm_type = LogNorm if logarithmic else Normalize
    return norm_type(vmin=limits[0], vmax=limits[1], clip=True)


def _set_y_range(
    axis: Axes,
    y_range: _YRange | None,
    *,
    logarithmic: bool = False,
) -> None:
    limits = _plot_range(y_range, "y_range", logarithmic=logarithmic)
    if limits is not None:
        axis.set_ylim(*limits)


def _plot_range(
    value_range: tuple[float, float] | None,
    name: str,
    *,
    logarithmic: bool,
) -> tuple[float, float] | None:
    if value_range is None:
        return None
    if (
        not isinstance(value_range, tuple)
        or len(value_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in value_range
        )
    ):
        raise TypeError(f"{name} must be a (minimum, maximum) numeric tuple.")
    minimum, maximum = (float(value) for value in value_range)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"{name} limits must be finite.")
    if minimum >= maximum:
        raise ValueError(f"{name} minimum must be less than its maximum.")
    if logarithmic and minimum <= 0:
        raise ValueError(f"A logarithmic {name} minimum must be positive.")
    return minimum, maximum


def _reject_spectrum_colorbar_range(
    cbar_range: _ColorbarRange | None,
) -> None:
    if cbar_range is not None:
        raise ValueError("cbar_range is only valid for map output, not spectra.")


def _reject_map_y_range(y_range: _YRange | None) -> None:
    if y_range is not None:
        raise ValueError("y_range is only valid for spectrum output, not maps.")


def _positive_values(values: np.ndarray) -> np.ndarray:
    unmasked = np.ma.asarray(values).compressed()
    return unmasked[(unmasked > 0) & np.isfinite(unmasked)]


def _decorate_detector_axes(axis: Axes, shape: tuple[int, ...]) -> None:
    ny, nx = shape[:2]
    axis.set(xlabel="X [spaxel]", ylabel="Y [spaxel]")
    if nx <= 50 and ny <= 50:
        axis.set_xticks(np.arange(0, nx, 4))
        axis.set_yticks(np.arange(0, ny, 4))
        axis.set_xticks(np.arange(nx), minor=True)
        axis.set_yticks(np.arange(ny), minor=True)
        axis.grid(
            which="minor",
            color="white",
            linestyle="--",
            linewidth=0.3,
            alpha=0.5,
        )
        axis.grid(
            which="major",
            color="white",
            linestyle="-",
            linewidth=0.5,
            alpha=0.8,
        )


def _target_model(
    result: EtcResult,
    target: int,
    high_res: bool,
    *,
    function_name: str,
) -> tuple[EtcResult, ModelGrid]:
    result = _require_group(result, "models", function_name, "models")
    if not isinstance(high_res, bool):
        raise TypeError("high_res must be a Boolean.")
    if isinstance(target, bool) or not isinstance(target, Integral):
        raise TypeError("target must be a zero-based integer index.")
    target = int(target)
    if not 0 <= target < len(result.models.targets):
        raise ValueError(f"target index {target} is out of range.")
    selected = result.models.targets[target]
    return result, selected.high if high_res else selected.low


def _slice_selection(
    result: EtcResult,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None,
    wavelength: int | u.Quantity | None,
) -> tuple[str, Any]:
    if position is not None and wavelength is not None:
        position_selection = _position_selection(result, position)
        wavelength_selection = _wavelength_selection(result, wavelength)
        if wavelength_selection[0] != "wavelength_range":
            raise ValueError(
                "When position and wavelength are both provided, wavelength "
                "must be a two-element spectral Quantity range."
            )
        if wavelength_selection[1].size < 2:
            raise ValueError(
                "A wavelength range used with position must contain at least "
                "two detector wavelength samples."
            )
        return "position_wavelength_range", (
            position_selection,
            wavelength_selection[1],
        )
    if wavelength is not None:
        return _wavelength_selection(result, wavelength)
    if position is None:
        position = (result.snr.shape[0] // 2, result.snr.shape[1] // 2)
    return _position_selection(result, position)


def _plot_selection(
    result: EtcResult,
    position: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] | None,
    wavelength: int | u.Quantity | None,
    apertures: tuple[ApertureResult, ...],
) -> tuple[str, Any]:
    if position is not None or wavelength is not None or not apertures:
        return _slice_selection(result, position, wavelength)
    return "wavelength_range", _shared_aperture_wavelength_indices(apertures)


def _shared_aperture_wavelength_indices(
    apertures: tuple[ApertureResult, ...],
) -> np.ndarray:
    support = apertures[0].mask.any(axis=(0, 1))
    if any(
        not np.array_equal(aperture.mask.any(axis=(0, 1)), support)
        for aperture in apertures[1:]
    ):
        raise ValueError(
            "Cannot infer wavelength because the selected apertures have "
            "different spectral support; provide a wavelength range explicitly."
        )
    indices = np.flatnonzero(support)
    if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError(
            "Cannot infer wavelength because the selected aperture spectral "
            "support is not contiguous; provide a wavelength range explicitly."
        )
    return indices


def _position_selection(
    result: EtcResult,
    selector: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
) -> tuple[str, Any]:
    if not isinstance(selector, tuple) or len(selector) != 2:
        raise TypeError(
            "position must be (y, x) or inclusive "
            "((y_min, y_max), (x_min, x_max)) bounds."
        )
    if all(_is_integer(value) for value in selector):
        y, x = (int(value) for value in selector)
        if not (0 <= y < result.snr.shape[0] and 0 <= x < result.snr.shape[1]):
            raise ValueError(f"position (y={y}, x={x}) is out of range.")
        return "position", (y, x)
    if not all(
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and all(_is_integer(value) for value in bounds)
        for bounds in selector
    ):
        raise TypeError(
            "position must be (y, x) or inclusive "
            "((y_min, y_max), (x_min, x_max)) bounds."
        )
    (y_min, y_max), (x_min, x_max) = (
        tuple(int(value) for value in bounds) for bounds in selector
    )
    if y_min > y_max or x_min > x_max:
        raise ValueError("position range lower bounds must not exceed upper bounds.")
    ny, nx = result.snr.shape[:2]
    if not (0 <= y_min <= y_max < ny and 0 <= x_min <= x_max < nx):
        raise ValueError(
            "position range "
            f"((y={y_min}, {y_max}), (x={x_min}, {x_max})) is out of range."
        )
    return "position_range", (y_min, y_max, x_min, x_max)


def _is_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral)


def _wavelength_selection(
    result: EtcResult,
    selector: int | u.Quantity,
) -> tuple[str, Any]:
    if isinstance(selector, u.Quantity) and not selector.isscalar:
        return "wavelength_range", _wavelength_range_indices(result, selector)
    return "wavelength", _wavelength_index(result, selector)


def _wavelength_index(result: EtcResult, selector: int | u.Quantity) -> int:
    if isinstance(selector, bool):
        raise TypeError("wavelength must be an integer index or scalar Quantity.")
    if isinstance(selector, Integral):
        selector = int(selector)
        if not 0 <= selector < result.wavelength.size:
            raise ValueError(f"wavelength index {selector} is out of range.")
        return selector
    if not isinstance(selector, u.Quantity) or not selector.isscalar:
        raise TypeError("wavelength must be an integer index or scalar Quantity.")
    try:
        value = selector.to_value(result.wavelength.unit)
    except u.UnitConversionError as exc:
        raise ValueError("wavelength must have spectral units.") from exc
    if not np.isfinite(value):
        raise ValueError("wavelength must be finite.")
    lower = result.wavelength[0].value
    upper = result.wavelength[-1].value
    if not lower <= value <= upper:
        raise ValueError("wavelength is outside the detector wavelength range.")
    return int(np.argmin(np.abs(result.wavelength.value - value)))


def _wavelength_range_indices(
    result: EtcResult,
    selector: u.Quantity,
) -> np.ndarray:
    if selector.shape != (2,):
        raise TypeError(
            "wavelength must be an integer index, scalar Quantity, or "
            "two-element spectral Quantity range."
        )
    try:
        lower, upper = selector.to_value(result.wavelength.unit)
    except u.UnitConversionError as exc:
        raise ValueError("wavelength must have spectral units.") from exc
    if not np.isfinite([lower, upper]).all():
        raise ValueError("wavelength range bounds must be finite.")
    if lower > upper:
        raise ValueError(
            "wavelength range lower bound must not exceed its upper bound."
        )
    detector_lower = result.wavelength[0].value
    detector_upper = result.wavelength[-1].value
    if lower < detector_lower or upper > detector_upper:
        raise ValueError("wavelength range is outside the detector wavelength range.")
    indices = np.flatnonzero(
        (result.wavelength.value >= lower) & (result.wavelength.value <= upper)
    )
    if indices.size == 0:
        raise ValueError("wavelength range contains no detector wavelength samples.")
    return indices


def _signal_spectrum_values(
    result: EtcResult,
    field: str,
    selection: tuple[str, Any],
) -> u.Quantity:
    position_selection, wavelength_indices = _spectrum_selection_parts(selection)
    values = getattr(result.signals, field)
    if position_selection[0] == "position":
        y, x = position_selection[1]
        spectrum = values[y, x, :]
    else:
        y_min, y_max, x_min, x_max = position_selection[1]
        spectrum = values[y_min : y_max + 1, x_min : x_max + 1, :].sum(
            axis=(0, 1)
        )
    if wavelength_indices is not None:
        spectrum = spectrum[wavelength_indices]
    return spectrum


def _signal_map_values(
    result: EtcResult,
    field: str,
    selection: tuple[str, Any],
) -> u.Quantity:
    values = getattr(result.signals, field)
    if selection[0] == "wavelength":
        return values[:, :, selection[1]]
    return values[:, :, selection[1]].sum(axis=2)


def _position_label(selection: tuple[str, Any]) -> str:
    if selection[0] == "position":
        y, x = selection[1]
        return f"position (y={y}, x={x})"
    y_min, y_max, x_min, x_max = selection[1]
    return f"position range (y={y_min}:{y_max}, x={x_min}:{x_max})"


def _spectrum_selection_parts(
    selection: tuple[str, Any],
) -> tuple[tuple[str, Any], np.ndarray | None]:
    if selection[0] == "position_wavelength_range":
        return selection[1]
    return selection, None


def _wavelength_selection_label(
    result: EtcResult,
    selection: tuple[str, Any],
) -> str:
    if selection[0] == "wavelength":
        return _wavelength_label(result, selection[1])
    indices = selection[1]
    lower = result.wavelength[indices[0]].to_value(u.micron)
    upper = result.wavelength[indices[-1]].to_value(u.micron)
    return f"{lower:.6f}-{upper:.6f} micron"


def _snr_spectrum_for_position_range(
    result: EtcResult,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    y_min, y_max, x_min, x_max = bounds
    mask = np.zeros(result.snr.shape[:2], dtype=bool)
    mask[y_min : y_max + 1, x_min : x_max + 1] = True
    return result._snr_spectrum_for_spatial_mask(mask)


def _snr_map_for_wavelength_range(
    result: EtcResult,
    indices: np.ndarray,
) -> np.ndarray:
    return result._snr_map_for_wavelength_indices(indices)


def _aperture(
    result: EtcResult,
    selector: int | str | ApertureResult,
) -> tuple[EtcResult, ApertureResult]:
    result = _require_result(result)
    if isinstance(selector, bool):
        raise TypeError(
            "aperture must be a zero-based integer index, name, or an "
            "ApertureResult from result.apertures."
        )
    if isinstance(selector, Integral):
        selector = int(selector)
        if not 0 <= selector < len(result.apertures):
            raise ValueError(f"aperture index {selector} is out of range.")
        aperture = result.apertures[selector]
    elif isinstance(selector, str):
        matches = [item for item in result.apertures if item.name == selector]
        if not matches:
            raise ValueError(f"No aperture is named {selector!r}.")
        aperture = matches[0]
    elif isinstance(selector, ApertureResult):
        if not any(selector is item for item in result.apertures):
            raise ValueError("aperture does not belong to this result.")
        aperture = selector
    else:
        raise TypeError(
            "aperture must be a zero-based integer index, name, or an "
            "ApertureResult from result.apertures."
        )
    return result, aperture


def _aperture_selection(
    result: EtcResult,
    selection: _ApertureSelection,
) -> tuple[EtcResult, tuple[ApertureResult, ...]]:
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        if not selection:
            raise ValueError("aperture selection must not be empty.")
        selected = tuple(_aperture(result, selector)[1] for selector in selection)
    else:
        result, aperture = _aperture(result, selection)
        selected = (aperture,)
    if len({id(aperture) for aperture in selected}) != len(selected):
        raise ValueError("aperture selection must not contain duplicates.")
    return _require_result(result), selected


def _require_result(result: EtcResult) -> EtcResult:
    if not isinstance(result, EtcResult):
        raise TypeError("result must be an EtcResult returned by Etc.run().")
    return result


def _require_group(
    result: EtcResult,
    group: str,
    function_name: str,
    run_option: str,
) -> EtcResult:
    result = _require_result(result)
    if not hasattr(result, group):
        raise ValueError(
            f"{function_name} requires result.{group}; "
            f"call Etc.run(include_{run_option}=True)."
        )
    return result


def _angular_extent(
    shape: tuple[int, ...],
    pixel_scale: u.Quantity,
) -> tuple[float, float, float, float]:
    ny, nx = shape[:2]
    scale = pixel_scale.to_value(u.mas)
    return (-nx * scale / 2, nx * scale / 2, -ny * scale / 2, ny * scale / 2)


def _detector_extent(shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    ny, nx = shape[:2]
    return (0, nx, 0, ny)


def _add_colorbar(
    figure: Figure,
    axis: Axes,
    image: Any,
    label: str,
) -> None:
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.08)
    figure.colorbar(image, cax=colorbar_axis, label=label)


def _wavelength(result: EtcResult) -> np.ndarray:
    return result.wavelength.to_value(u.micron)


def _wavelength_label(result: EtcResult, index: int) -> str:
    return f"{result.wavelength[index].to_value(u.micron):.4f} micron"


def _quantity_label(name: str, values: u.Quantity) -> str:
    return f"{name} [{values.unit.to_string('latex_inline')}]"


def _set_title(axis: Axes, title: str | None, default: str) -> None:
    axis.set_title(default if title is None else title)
