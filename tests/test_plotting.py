"""Public plotting API tests."""

from __future__ import annotations

import inspect
import sys

import astropy.units as u
import matplotlib
import numpy as np
import pytest
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

import cubesim

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import cubesim.plotting as plotting  # noqa: PLR0402
from cubesim.diagnostics import PsfStats


def test_plotting_range_parameters_are_consistent():
    map_functions = (
        plotting.plot_psf,
        plotting.plot_target_models,
        plotting.plot_target_spatial,
        plotting.plot_target_velocity,
        plotting.plot_signal_components,
        plotting.plot_signal_target,
        plotting.plot_signal_sky,
        plotting.plot_signal_thermal,
        plotting.plot_signal_dark,
        plotting.plot_signal_total,
        plotting.plot_snr,
        plotting.plot_aperture_signal_maps,
        plotting.plot_aperture_snr,
        plotting.plot_aperture_snr_map,
    )
    spectrum_functions = (
        plotting.plot_target_models,
        plotting.plot_target_spectrum,
        plotting.plot_background_models,
        plotting.plot_background_transmission,
        plotting.plot_background_sky,
        plotting.plot_background_thermal,
        plotting.plot_signal_components,
        plotting.plot_signal_target,
        plotting.plot_signal_sky,
        plotting.plot_signal_thermal,
        plotting.plot_signal_dark,
        plotting.plot_signal_total,
        plotting.plot_snr,
        plotting.plot_aperture_signal_spectra,
        plotting.plot_aperture_snr,
        plotting.plot_aperture_snr_spectrum,
    )

    assert all(
        "cbar_range" in inspect.signature(function).parameters
        for function in map_functions
    )
    assert all(
        "y_range" in inspect.signature(function).parameters
        for function in spectrum_functions
    )


@pytest.fixture
def result(instrument_data):
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
        velocity=cubesim.ConstantVelocity(75 * u.km / u.s),
    )
    psf = np.arange(1, 26, dtype=float).reshape(5, 5)
    etc.set_psf(psf, pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    etc.add_aperture(
        name="line",
        size=(2, 2, 3),
        center=(1.5, 1.5, 1.1 * u.micron),
    )
    return etc.run(include_models=True)


def _minimal_etc(instrument_data):
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Uniform(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2 * u.arcsec**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_exposure(time=100 * u.s, n_target=2)
    return etc


def test_public_plotting_functions_return_figures_without_showing(result, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: pytest.fail("show called"))
    calls = [
        lambda: plotting.plot_psf(result),
        lambda: plotting.plot_target_models(result),
        lambda: plotting.plot_target_spatial(result),
        lambda: plotting.plot_target_spectrum(result),
        lambda: plotting.plot_target_velocity(result),
        lambda: plotting.plot_background_models(result),
        lambda: plotting.plot_background_transmission(result),
        lambda: plotting.plot_background_sky(result),
        lambda: plotting.plot_background_thermal(result),
        lambda: plotting.plot_signal_components(result),
        lambda: plotting.plot_signal_target(result),
        lambda: plotting.plot_signal_sky(result),
        lambda: plotting.plot_signal_thermal(result),
        lambda: plotting.plot_signal_dark(result),
        lambda: plotting.plot_signal_total(result),
        lambda: plotting.plot_snr(result),
        lambda: plotting.plot_aperture_signal_spectra(result),
        lambda: plotting.plot_aperture_signal_maps(result),
        lambda: plotting.plot_aperture_snr(result),
        lambda: plotting.plot_aperture_snr_spectrum(result),
        lambda: plotting.plot_aperture_snr_map(result),
    ]

    for call in calls:
        figure = call()
        assert isinstance(figure, Figure)
        plt.close(figure)


def test_psf_plot_uses_result_snapshot_and_angular_radius(result):
    figure = plotting.plot_psf(result, radius=15 * u.mas)
    axis = figure.axes[0]
    expected = np.log10(np.clip(result.psf.data / result.psf.data.max(), 1e-10, None))

    assert np.allclose(axis.images[0].get_array(), expected)
    assert axis.images[0].get_extent() == [-25.0, 25.0, -25.0, 25.0]
    assert axis.get_xlim() == pytest.approx((-15.0, 15.0))
    assert axis.get_ylim() == pytest.approx((-15.0, 15.0))
    assert axis.get_title() == "PSF"
    assert not axis.lines
    plt.close(figure)


def test_psf_plot_accepts_explicit_title(result):
    figure = plotting.plot_psf(result, title="Configured PSF")

    assert figure.axes[0].get_title() == "Configured PSF"
    plt.close(figure)


def test_psf_plot_does_not_require_ao_stats(result, monkeypatch):
    monkeypatch.setitem(sys.modules, "ao_stats", None)

    figure = plotting.plot_psf(result)

    assert figure.axes[0].get_title() == "PSF"
    plt.close(figure)


def test_psf_plot_includes_optional_stats_in_title(result):
    stats = PsfStats(
        sr=None,
        fwhm=78 * u.mas,
        ee_apertures=np.array([50, 100]) * u.mas,
        ee=np.array([0.4, 0.7]) * u.one,
    )
    figure = plotting.plot_psf(result, stats=stats, title="Measured PSF")

    assert figure.axes[0].get_title() == (
        "Measured PSF, FWHM: 78 mas, EE(50 mas): 0.40"
    )
    plt.close(figure)

    hybrid_stats = PsfStats(
        sr=0.35 * u.one,
        fwhm=78 * u.mas,
        ee_apertures=np.array([100]) * u.mas,
        ee=np.array([0.7]) * u.one,
    )
    figure = plotting.plot_psf(result, stats=hybrid_stats)

    assert figure.axes[0].get_title() == (
        "PSF, SR: 0.35, FWHM: 78 mas, EE(100 mas): 0.70"
    )
    plt.close(figure)

    with pytest.raises(TypeError, match="PsfStats"):
        plotting.plot_psf(result, stats=object())


def test_image_colorbars_match_plot_height(result):
    figures = (
        plotting.plot_psf(result),
        plotting.plot_target_models(result),
        plotting.plot_signal_components(result, wavelength=0),
        plotting.plot_snr(result, wavelength=0),
        plotting.plot_aperture_signal_maps(result),
        plotting.plot_aperture_snr(result),
    )

    for figure in figures:
        figure.canvas.draw()
        image_axes = [axis for axis in figure.axes if axis.images]
        colorbar_axes = [axis for axis in figure.axes if hasattr(axis, "_colorbar")]
        assert len(colorbar_axes) == len(image_axes)
        for image_axis, colorbar_axis in zip(
            image_axes,
            colorbar_axes,
            strict=True,
        ):
            image_bounds = image_axis.get_position()
            colorbar_bounds = colorbar_axis.get_position()
            assert colorbar_bounds.y0 == pytest.approx(image_bounds.y0)
            assert colorbar_bounds.y1 == pytest.approx(image_bounds.y1)

    for figure in figures:
        plt.close(figure)


def test_map_plotters_accept_explicit_colorbar_ranges(result):
    cases = (
        (plotting.plot_psf(result, cbar_range=(-4, 0)), (-4, 0)),
        (plotting.plot_target_models(result, cbar_range=(0, 1)), (0, 1)),
        (plotting.plot_target_spatial(result, cbar_range=(0, 1)), (0, 1)),
        (
            plotting.plot_target_velocity(result, cbar_range=(-100, 100)),
            (-100, 100),
        ),
        (
            plotting.plot_signal_components(
                result,
                wavelength=0,
                cbar_range=(1e-6, 1e6),
            ),
            (1e-6, 1e6),
        ),
        (
            plotting.plot_signal_target(
                result,
                wavelength=0,
                cbar_range=(1e-6, 1e6),
            ),
            (1e-6, 1e6),
        ),
        (plotting.plot_snr(result, wavelength=0, cbar_range=(0, 50)), (0, 50)),
        (
            plotting.plot_aperture_signal_maps(
                result,
                cbar_range=(1e-6, 1e6),
            ),
            (1e-6, 1e6),
        ),
        (plotting.plot_aperture_snr(result, cbar_range=(0, 50)), (0, 50)),
        (plotting.plot_aperture_snr_map(result, cbar_range=(0, 50)), (0, 50)),
    )

    for figure, expected in cases:
        for axis in (axis for axis in figure.axes if axis.images):
            assert axis.images[0].norm.vmin == expected[0]
            assert axis.images[0].norm.vmax == expected[1]
        plt.close(figure)


def test_spectrum_plotters_accept_explicit_y_ranges(result):
    cases = (
        (plotting.plot_target_models(result, y_range=(1e-30, 1e-10)), (1e-30, 1e-10)),
        (
            plotting.plot_target_spectrum(result, y_range=(1e-30, 1e-10)),
            (1e-30, 1e-10),
        ),
        (plotting.plot_background_models(result, y_range=(1, 2)), (1, 2)),
        (
            plotting.plot_background_transmission(result, y_range=(0, 1)),
            (0, 1),
        ),
        (plotting.plot_background_sky(result, y_range=(1, 2)), (1, 2)),
        (
            plotting.plot_signal_components(
                result,
                position=(1, 2),
                y_range=(1e-6, 1e6),
            ),
            (1e-6, 1e6),
        ),
        (
            plotting.plot_signal_target(
                result,
                position=(1, 2),
                y_range=(1e-6, 1e6),
            ),
            (1e-6, 1e6),
        ),
        (plotting.plot_snr(result, position=(1, 2), y_range=(0, 100)), (0, 100)),
        (
            plotting.plot_aperture_signal_spectra(
                result,
                y_range=(1e-6, 1e6),
            ),
            (1e-6, 1e6),
        ),
        (plotting.plot_aperture_snr(result, y_range=(0, 100)), (0, 100)),
        (
            plotting.plot_aperture_snr_spectrum(result, y_range=(0, 100)),
            (0, 100),
        ),
    )

    for figure, expected in cases:
        spectrum_axes = [
            axis
            for axis in figure.axes
            if axis.lines and not hasattr(axis, "_colorbar")
        ]
        assert spectrum_axes
        for axis in spectrum_axes:
            assert axis.get_ylim() == pytest.approx(expected)
        plt.close(figure)


def test_target_plots_use_requested_resolution(result):
    spectrum_unit = u.erg / (u.s * u.cm**2 * u.arcsec**2 * u.AA)
    low = plotting.plot_target_spatial(result)
    high = plotting.plot_target_spatial(result, high_res=True)
    spectrum = plotting.plot_target_spectrum(result, high_res=True)
    models = plotting.plot_target_models(result)

    assert np.array_equal(low.axes[0].images[0].get_array(), result.models.targets[0].low.spatial)
    assert np.array_equal(
        high.axes[0].images[0].get_array(), result.models.targets[0].high.spatial
    )
    assert np.array_equal(
        spectrum.axes[0].lines[0].get_ydata(),
        result.models.targets[0].high.spectrum.to_value(spectrum_unit),
    )
    assert np.array_equal(
        spectrum.axes[0].lines[1].get_ydata(),
        result.models.targets[0].high.spectrum_convolved.to_value(spectrum_unit),
    )
    assert spectrum.axes[0].get_yscale() == "log"
    assert models.axes[1].get_yscale() == "log"
    assert spectrum.axes[0].get_ylabel() == (
        "Spectral surface brightness "
        f"[{spectrum_unit.to_string('latex_inline')}]"
    )
    spectrum_peak = result.models.targets[0].high.spectrum.to_value(
        spectrum_unit
    ).max()
    assert spectrum.axes[0].get_ylim()[0] >= spectrum_peak * 1e-6
    for figure in (low, high, spectrum, models):
        plt.close(figure)


def test_target_models_use_vertical_six_inch_layout(result):
    figure = plotting.plot_target_models(result)
    panel_axes = [axis for axis in figure.axes if axis.get_title()]

    assert figure.get_size_inches() == pytest.approx((6, 15))
    assert [axis.get_title() for axis in panel_axes] == [
        "Target spatial model",
        "Target spectrum",
        "Target velocity",
    ]
    panel_y = [axis.get_position().y0 for axis in panel_axes]
    assert panel_y == sorted(panel_y, reverse=True)
    plt.close(figure)


def test_background_plots_use_retained_models(result):
    transmission = plotting.plot_background_transmission(result)
    sky = plotting.plot_background_sky(result)
    thermal = plotting.plot_background_thermal(result)

    assert np.array_equal(
        transmission.axes[0].lines[0].get_ydata(), result.models.transmission
    )
    assert np.array_equal(sky.axes[0].lines[0].get_ydata(), result.models.sky.value)
    assert np.array_equal(
        thermal.axes[0].lines[0].get_ydata(), result.models.thermal.value
    )
    for figure in (transmission, sky, thermal):
        plt.close(figure)


def test_background_models_use_vertical_six_inch_layout(result):
    figure = plotting.plot_background_models(result)
    panel_axes = [axis for axis in figure.axes if axis.get_title()]

    assert figure.get_size_inches() == pytest.approx((6, 15))
    assert [axis.get_title() for axis in panel_axes] == [
        "Atmospheric transmission",
        "Sky radiance",
        "Thermal radiance",
    ]
    panel_y = [axis.get_position().y0 for axis in panel_axes]
    assert panel_y == sorted(panel_y, reverse=True)
    plt.close(figure)


def test_signal_and_snr_select_orthogonal_cube_slices(result):
    position = plotting.plot_signal_target(result, position=(1, 2))
    wavelength = result.wavelength[7] + 0.1 * np.diff(result.wavelength[:2])[0]
    signal_map = plotting.plot_signal_target(result, wavelength=wavelength)
    snr_spectrum = plotting.plot_snr(result, position=(1, 2))
    snr_map = plotting.plot_snr(result, wavelength=7)

    assert np.array_equal(
        position.axes[0].lines[0].get_ydata(), result.signals.target[1, 2, :].value
    )
    assert position.axes[0].get_yscale() == "log"
    assert np.array_equal(
        signal_map.axes[0].images[0].get_array(), result.signals.target[:, :, 7].value
    )
    assert np.array_equal(
        snr_spectrum.axes[0].lines[0].get_ydata(), result.snr[1, 2, :]
    )
    assert np.array_equal(snr_map.axes[0].images[0].get_array(), result.snr[:, :, 7])
    for figure in (position, signal_map, snr_spectrum, snr_map):
        plt.close(figure)


def test_signal_ranges_sum_selected_detector_samples(result):
    position = ((0, 1), (1, 3))
    wavelength = result.wavelength[[5, 7]]
    spectrum = plotting.plot_signal_components(result, position=position)
    constrained_spectrum = plotting.plot_signal_components(
        result,
        position=position,
        wavelength=wavelength,
    )
    signal_map = plotting.plot_signal_target(result, wavelength=wavelength)

    assert np.array_equal(
        spectrum.axes[0].lines[0].get_ydata(),
        result.signals.target[0:2, 1:4, :].sum(axis=(0, 1)).value,
    )
    assert spectrum.axes[0].get_xlim() == pytest.approx(
        result.wavelength[[0, -1]].to_value(u.micron)
    )
    assert spectrum.axes[0].get_yscale() == "log"
    peak = max(np.asarray(line.get_ydata()).max() for line in spectrum.axes[0].lines)
    background = np.concatenate(
        [
            np.asarray(line.get_ydata())
            for line in spectrum.axes[0].lines
            if line.get_label() in {"Sky", "Thermal", "Dark"}
        ]
    )
    assert spectrum.axes[0].get_ylim() == pytest.approx(
        (background[background > 0].min() / 2, peak * 2)
    )
    assert np.array_equal(
        constrained_spectrum.axes[0].lines[0].get_xdata(),
        result.wavelength[5:8].to_value(u.micron),
    )
    assert np.array_equal(
        constrained_spectrum.axes[0].lines[0].get_ydata(),
        result.signals.target[0:2, 1:4, 5:8].sum(axis=(0, 1)).value,
    )
    assert constrained_spectrum.axes[0].get_xlim() == pytest.approx(
        result.wavelength[[5, 7]].to_value(u.micron)
    )
    assert np.array_equal(
        signal_map.axes[0].images[0].get_array(),
        result.signals.target[:, :, 5:8].sum(axis=2).value,
    )
    plt.close(spectrum)
    plt.close(constrained_spectrum)
    plt.close(signal_map)


def test_snr_ranges_recompute_after_summing_signal_and_variance(result):
    position = ((0, 1), (1, 3))
    wavelength = result.wavelength[[5, 7]]
    spectrum = plotting.plot_snr(result, position=position)
    constrained_spectrum = plotting.plot_snr(
        result,
        position=position,
        wavelength=wavelength,
    )
    snr_map = plotting.plot_snr(result, wavelength=wavelength)

    position_signal = result.signals.target[0:2, 1:4, :].sum(axis=(0, 1))
    position_variance = result.variances.total[0:2, 1:4, :].sum(axis=(0, 1))
    wavelength_signal = result.signals.target[:, :, 5:8].sum(axis=2)
    wavelength_variance = result.variances.total[:, :, 5:8].sum(axis=2)
    assert np.allclose(
        spectrum.axes[0].lines[0].get_ydata(),
        position_signal.value / np.sqrt(position_variance.value),
    )
    assert np.allclose(
        constrained_spectrum.axes[0].lines[0].get_ydata(),
        (position_signal.value / np.sqrt(position_variance.value))[5:8],
    )
    assert np.allclose(
        snr_map.axes[0].images[0].get_array(),
        wavelength_signal.value / np.sqrt(wavelength_variance.value),
    )
    plt.close(spectrum)
    plt.close(constrained_spectrum)
    plt.close(snr_map)


def test_signal_and_snr_aperture_overlays_use_stable_ranges_and_colors(result):
    aperture = result.apertures[0]
    spectral_support = aperture.mask.any(axis=(0, 1))
    indices = np.flatnonzero(spectral_support)
    spectrum = plotting.plot_snr(
        result,
        position=(1, 2),
        apertures=(0,),
    )
    signal_spectrum = plotting.plot_signal_target(
        result,
        position=(1, 2),
        apertures=(aperture,),
    )

    expected_boundaries = result.wavelength[indices[[0, -1]]].to_value(u.micron)
    for figure in (spectrum, signal_spectrum):
        boundary_lines = figure.axes[0].lines[-2:]
        assert [line.get_xdata()[0] for line in boundary_lines] == pytest.approx(
            expected_boundaries
        )
        assert all(line.get_color() == "C0" for line in boundary_lines)
        assert all(line.get_linestyle() == "-" for line in boundary_lines)
        assert [text.get_text() for text in figure.axes[0].get_legend().texts] == [
            "line"
        ]
        plt.close(figure)


@pytest.mark.parametrize("use_range", [False, True])
def test_snr_aperture_overlays_follow_spaxel_boundaries(result, use_range):
    aperture = result.apertures[0]
    spectral_support = aperture.mask.any(axis=(0, 1))
    index = np.flatnonzero(spectral_support)[0]
    wavelength = (
        result.wavelength[[index, index + 1]]
        if use_range
        else int(index)
    )
    figure = plotting.plot_snr(
        result,
        wavelength=wavelength,
        apertures=("line",),
    )
    axis = figure.axes[0]
    colored_outline = [
        collection
        for collection in axis.collections
        if isinstance(collection, LineCollection)
    ][-1]
    vertices = np.concatenate(colored_outline.get_segments())
    selected_indices = [index, index + 1] if use_range else [index]
    support = aperture.mask[:, :, selected_indices].any(axis=2)
    y, x = np.nonzero(support)

    assert vertices[:, 0].min() == x.min()
    assert vertices[:, 0].max() == x.max() + 1
    assert vertices[:, 1].min() == y.min()
    assert vertices[:, 1].max() == y.max() + 1
    assert np.array_equal(vertices, vertices.astype(int))
    assert np.allclose(
        colored_outline.get_colors()[0],
        matplotlib.colors.to_rgba("C0"),
    )
    assert colored_outline.get_linestyle() == [(0.0, None)]
    assert [text.get_text() for text in axis.get_legend().texts] == ["line"]
    plt.close(figure)


def test_aperture_overlay_omits_nonoverlapping_wavelength(result):
    aperture = result.apertures[0]
    index = int(np.flatnonzero(~aperture.mask.any(axis=(0, 1)))[0])
    figure = plotting.plot_snr(result, wavelength=index, apertures=(0,))
    axis = figure.axes[0]

    assert axis.get_legend() is None
    assert not [
        collection
        for collection in axis.collections
        if isinstance(collection, LineCollection)
    ]
    plt.close(figure)


def test_position_range_snr_preserves_in_field_covariance(instrument_data):
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_psf(np.ones((3, 3)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    sky_mask = np.zeros((3, 4), dtype=bool)
    sky_mask[:, 0] = True
    etc.set_sky_subtraction(method="in_field", mask=sky_mask)
    etc.add_aperture(name="range", size=(2, 2, 3), start=(0, 1, 0))
    result = etc.run()

    spectrum = plotting.plot_snr(result, position=((0, 1), (1, 2)))
    snr_map = plotting.plot_snr(result, wavelength=result.wavelength[[0, 2]])
    aperture = result.apertures[0]
    spectral_support = aperture.mask.any(axis=(0, 1))
    map_support = aperture.mask.any(axis=2)

    assert np.allclose(
        spectrum.axes[0].lines[0].get_ydata()[spectral_support],
        aperture.spectra.snr[spectral_support],
    )
    assert np.allclose(
        snr_map.axes[0].images[0].get_array()[map_support],
        aperture.maps.snr[map_support],
    )
    plt.close(spectrum)
    plt.close(snr_map)


def test_snr_ranges_use_default_signal_and_variance_groups(instrument_data):
    etc = _minimal_etc(instrument_data)
    result = etc.run()

    scalar = plotting.plot_snr(result, wavelength=0)
    constrained_scalar = plotting.plot_snr(
        result,
        position=(0, 0),
        wavelength=result.wavelength[[0, 2]],
    )
    position_range = plotting.plot_snr(result, position=((0, 1), (0, 1)))
    wavelength_range = plotting.plot_snr(
        result,
        wavelength=result.wavelength[[0, 1]],
    )
    plt.close(scalar)
    assert np.array_equal(
        constrained_scalar.axes[0].lines[0].get_ydata(),
        result.snr[0, 0, 0:3],
    )
    plt.close(constrained_scalar)
    plt.close(position_range)
    plt.close(wavelength_range)


def test_signal_component_maps_use_vertical_six_inch_layout(result):
    figure = plotting.plot_signal_components(result, wavelength=0)
    panel_axes = [axis for axis in figure.axes if axis.images]

    assert figure.get_size_inches() == pytest.approx((6, 25))
    assert [axis.get_title().split(" at ")[0] for axis in panel_axes] == [
        "Target",
        "Sky",
        "Thermal",
        "Dark",
        "Total",
    ]
    norms = [axis.images[0].norm for axis in panel_axes]
    assert isinstance(norms[0], LogNorm)
    assert all(norm is norms[0] for norm in norms)
    assert norms[0].vmax == max(
        getattr(result.signals, field)[:, :, 0].value.max()
        for field in result.signals.__dataclass_fields__
    )
    assert norms[0].vmin >= norms[0].vmax * 1e-6
    panel_y = [axis.get_position().y0 for axis in panel_axes]
    assert panel_y == sorted(panel_y, reverse=True)
    plt.close(figure)


def test_detector_maps_use_one_boundary_lattice(result):
    figure = plotting.plot_snr(result, wavelength=0)
    axis = figure.axes[0]

    assert axis.images[0].get_extent() == [0, 4, 0, 3]
    assert np.array_equal(axis.get_xticks(), np.arange(0, result.snr.shape[1], 4))
    assert np.array_equal(axis.get_yticks(), np.arange(0, result.snr.shape[0], 4))
    assert np.array_equal(axis.get_xticks(minor=True), np.arange(1, 4))
    assert np.array_equal(axis.get_yticks(minor=True), np.arange(1, 3))
    assert axis.get_xlabel() == "X [spaxel]"
    assert axis.get_ylabel() == "Y [spaxel]"
    plt.close(figure)


def test_aperture_plots_use_precomputed_reductions_and_mask_support(result):
    aperture = result.apertures[0]
    spectra = plotting.plot_aperture_signal_spectra(result, aperture=aperture)
    maps = plotting.plot_aperture_signal_maps(result, aperture="line")
    snr_spectrum = plotting.plot_aperture_snr_spectrum(result, aperture="line")
    snr_map = plotting.plot_aperture_snr_map(result, aperture="line")
    overview = plotting.plot_snr(
        result,
        apertures=(aperture,),
    )

    expected_spectrum = np.ma.masked_where(
        ~aperture.mask.any(axis=(0, 1)), aperture.spectra.signals.target.value
    )
    expected_map = np.ma.masked_where(
        ~aperture.mask.any(axis=2), aperture.maps.signals.target.value
    )
    assert spectra.axes[0].get_title() == "line signal spectra"
    assert snr_spectrum.axes[0].get_title() == "line S/N spectrum"
    assert snr_map.axes[0].get_title() == "line S/N map"
    assert np.ma.allequal(spectra.axes[0].lines[0].get_ydata(), expected_spectrum)
    assert spectra.axes[0].get_yscale() == "log"
    assert np.ma.allequal(maps.axes[0].images[0].get_array(), expected_map)
    assert np.ma.allequal(
        snr_spectrum.axes[0].lines[0].get_ydata(),
        np.ma.masked_where(~aperture.mask.any(axis=(0, 1)), aperture.spectra.snr),
    )
    assert np.ma.allequal(
        snr_map.axes[0].images[0].get_array(),
        np.ma.masked_where(~aperture.mask.any(axis=2), aperture.maps.snr),
    )
    spatial_support = aperture.mask.any(axis=2)
    assert np.allclose(
        overview.axes[0].images[0].get_array()[spatial_support],
        aperture.maps.snr[spatial_support],
    )
    overview_outline = [
        collection
        for collection in overview.axes[0].collections
        if isinstance(collection, LineCollection)
    ][-1]
    overview_vertices = np.concatenate(overview_outline.get_segments())
    assert (
        overview_vertices[:, 0].min(),
        overview_vertices[:, 0].max(),
    ) == pytest.approx(snr_map.axes[0].get_xlim())
    assert (
        overview_vertices[:, 1].min(),
        overview_vertices[:, 1].max(),
    ) == pytest.approx(snr_map.axes[0].get_ylim())
    y, x = np.nonzero(aperture.mask.any(axis=2))
    expected_xlim = (x.min(), x.max() + 1)
    expected_ylim = (y.min(), y.max() + 1)
    map_axes = [axis for axis in maps.axes if axis.images]
    for axis in (*map_axes, snr_map.axes[0]):
        assert axis.get_xlim() == pytest.approx(expected_xlim)
        assert axis.get_ylim() == pytest.approx(expected_ylim)
    for figure in (spectra, maps, snr_spectrum, snr_map, overview):
        plt.close(figure)


def test_automatic_aperture_wavelength_requires_shared_support(instrument_data):
    etc = _minimal_etc(instrument_data)
    etc.add_aperture(name="first", size=(1, 1, 2), start=(0, 0, 0))
    etc.add_aperture(name="second", size=(1, 1, 2), start=(1, 1, 1))
    result = etc.run()

    with pytest.raises(
        ValueError,
        match="different spectral support; provide a wavelength range explicitly",
    ):
        plotting.plot_snr(result, apertures=result.apertures)

    spectrum = plotting.plot_snr(
        result,
        position=(0, 0),
        apertures=result.apertures,
    )
    assert spectrum.axes[0].lines
    plt.close(spectrum)


def test_multiple_aperture_snr_plots_stack_with_shared_colorbar(instrument_data):
    etc = _minimal_etc(instrument_data)
    for name, start in (
        ("core", (0, 0, 0)),
        ("redshifted", (1, 1, 0)),
        ("blueshifted", (2, 2, 0)),
    ):
        etc.add_aperture(name=name, size=(1, 1, 3), start=start)
    result = etc.run()

    combined = plotting.plot_aperture_snr(
        result,
        aperture=result.apertures,
        cbar_range=(0, 50),
        y_range=(0, 100),
    )
    map_axes = [axis for axis in combined.axes if axis.images]
    spectrum_axes = [
        axis
        for axis in combined.axes
        if axis.lines and not hasattr(axis, "_colorbar")
    ]
    norms = [axis.images[0].norm for axis in map_axes]

    assert combined.get_size_inches() == pytest.approx((14, 15))
    assert len(map_axes) == 3
    assert len(spectrum_axes) == 3
    assert all(axis.get_ylim() == pytest.approx((0, 100)) for axis in spectrum_axes)
    assert all(norm is norms[0] for norm in norms)
    assert (norms[0].vmin, norms[0].vmax) == (0, 50)
    assert [axis.get_title() for axis in map_axes] == [
        "core S/N map",
        "redshifted S/N map",
        "blueshifted S/N map",
    ]
    assert [axis.get_position().y0 for axis in map_axes] == sorted(
        (axis.get_position().y0 for axis in map_axes),
        reverse=True,
    )
    plt.close(combined)

    maps = plotting.plot_aperture_snr_map(
        result,
        aperture=("core", "redshifted", "blueshifted"),
    )
    map_axes = [axis for axis in maps.axes if axis.images]
    norms = [axis.images[0].norm for axis in map_axes]
    assert maps.get_size_inches() == pytest.approx((6, 15))
    assert all(norm is norms[0] for norm in norms)
    plt.close(maps)


def test_aperture_signal_maps_use_vertical_six_inch_layout(result):
    maps = plotting.plot_aperture_signal_maps(result)

    assert maps.get_size_inches() == pytest.approx((6, 25))
    map_axes = [axis for axis in maps.axes if axis.images]
    assert len(map_axes) == 5
    norms = [axis.images[0].norm for axis in map_axes]
    assert isinstance(norms[0], LogNorm)
    assert all(norm is norms[0] for norm in norms)
    plt.close(maps)


def test_plotting_rejects_invalid_selectors(result):
    figure = plotting.plot_snr(
        result,
        position=(np.int64(1), np.int64(2)),
    )
    plt.close(figure)

    with pytest.raises(ValueError, match="two-element spectral Quantity range"):
        plotting.plot_snr(result, position=(0, 0), wavelength=0)
    with pytest.raises(ValueError, match="position.*out of range"):
        plotting.plot_snr(result, position=(20, 0))
    with pytest.raises(ValueError, match="wavelength.*outside"):
        plotting.plot_snr(result, wavelength=2 * u.micron)
    with pytest.raises(ValueError, match="position range lower bounds"):
        plotting.plot_snr(result, position=((2, 1), (0, 1)))
    with pytest.raises(ValueError, match="position range.*out of range"):
        plotting.plot_snr(result, position=((0, 3), (0, 1)))
    with pytest.raises(TypeError, match="inclusive"):
        plotting.plot_snr(result, position=((0, 1), (0, "1")))
    with pytest.raises(ValueError, match="wavelength range lower bound"):
        plotting.plot_snr(result, wavelength=result.wavelength[[2, 1]])
    with pytest.raises(ValueError, match="wavelength range is outside"):
        plotting.plot_snr(
            result,
            wavelength=u.Quantity([0.8, 1.0], u.micron),
        )
    midpoint = (result.wavelength[0] + result.wavelength[1]) / 2
    with pytest.raises(ValueError, match="contains no detector wavelength samples"):
        plotting.plot_snr(
            result,
            wavelength=u.Quantity(
                [midpoint - 1e-12 * u.micron, midpoint + 1e-12 * u.micron]
            ),
        )
    with pytest.raises(ValueError, match="target index"):
        plotting.plot_target_models(result, target=4)
    with pytest.raises(TypeError, match="high_res"):
        plotting.plot_target_models(result, high_res="yes")
    with pytest.raises(ValueError, match="No aperture"):
        plotting.plot_aperture_snr(result, aperture="missing")
    with pytest.raises(TypeError, match="ApertureResult"):
        plotting.plot_aperture_snr(result, aperture=object())
    with pytest.raises(TypeError, match="apertures must be a sequence"):
        plotting.plot_snr(result, wavelength=0, apertures=0)
    with pytest.raises(ValueError, match="No aperture"):
        plotting.plot_snr(result, wavelength=0, apertures=("missing",))
    with pytest.raises(ValueError, match="finite and positive"):
        plotting.plot_psf(result, radius=0 * u.mas)
    with pytest.raises(ValueError, match="only valid for map output"):
        plotting.plot_snr(result, position=(0, 0), cbar_range=(0, 50))
    with pytest.raises(ValueError, match="only valid for spectrum output"):
        plotting.plot_snr(result, wavelength=0, y_range=(0, 50))
    with pytest.raises(ValueError, match="minimum must be less"):
        plotting.plot_snr(result, wavelength=0, cbar_range=(50, 0))
    with pytest.raises(ValueError, match="logarithmic.*positive"):
        plotting.plot_signal_target(
            result,
            wavelength=0,
            cbar_range=(0, 1),
        )
    with pytest.raises(ValueError, match="logarithmic.*positive"):
        plotting.plot_signal_target(
            result,
            position=(0, 0),
            y_range=(0, 1),
        )
    with pytest.raises(TypeError, match="numeric tuple"):
        plotting.plot_snr(result, wavelength=0, cbar_range=[0, 50])
    with pytest.raises(TypeError, match="numeric tuple"):
        plotting.plot_snr(result, position=(0, 0), y_range=[0, 50])
    with pytest.raises(ValueError, match="must not be empty"):
        plotting.plot_aperture_snr(result, aperture=())
    with pytest.raises(ValueError, match="must not contain duplicates"):
        plotting.plot_aperture_snr(result, aperture=(0, "line"))
    with pytest.raises(TypeError, match="EtcResult"):
        plotting.plot_snr(object())


def test_plotting_reports_missing_models_and_psf(instrument_data):
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Uniform(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2 * u.arcsec**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_exposure(time=100 * u.s, n_target=2)
    etc.add_aperture(name="voxel", size=(1, 1, 1), start=(0, 0, 0))
    result = etc.run()

    with pytest.raises(ValueError, match=r"include_models=True"):
        plotting.plot_target_models(result)
    signal_figure = plotting.plot_signal_components(result)
    aperture_figure = plotting.plot_aperture_signal_maps(result)
    plt.close(signal_figure)
    plt.close(aperture_figure)
    with pytest.raises(ValueError, match="configured PSF"):
        plotting.plot_psf(result)


def test_plotting_does_not_mutate_result(result):
    psf = result.psf.data.copy()
    snr = result.snr.copy()
    target = result.signals.target.copy()
    aperture = result.apertures[0].spectra.signals.target.copy()

    figures = (
        plotting.plot_psf(result),
        plotting.plot_signal_components(result, wavelength=2),
        plotting.plot_aperture_signal_maps(result),
    )

    assert np.array_equal(result.psf.data, psf)
    assert np.array_equal(result.snr, snr)
    assert np.array_equal(result.signals.target.value, target.value)
    assert np.array_equal(
        result.apertures[0].spectra.signals.target.value, aperture.value
    )
    for figure in figures:
        plt.close(figure)
