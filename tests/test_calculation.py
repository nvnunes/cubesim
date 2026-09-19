"""Focused tests for forward-calculation numerical contracts."""

from __future__ import annotations

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import fftconvolve

from cubesim._calculation import (
    SpectralGrid,
    _assemble_varying_velocity_cube,
    _centered_bounds,
    _convolve_spatial,
    _prepare_psf,
    _resample_spatial,
    _resample_spectrum,
    _sample_data,
    _shift_spectrum_by_velocity,
)
from cubesim._result import Signals


def test_sample_data_uses_independent_poisson_and_read_draws() -> None:
    target = np.array([[[12.0, 20.0]]]) * u.electron
    background = np.array([[[4.0, 8.0]]]) * u.electron
    signals = Signals(
        target=target,
        sky=background,
        thermal=np.zeros_like(background.value) * u.electron,
        dark=np.zeros_like(background.value) * u.electron,
        background=background,
        total=target + background,
    )
    exposure = SimpleNamespace(n_target=2, n_sky=1)

    actual = _sample_data(
        signals,
        exposure,
        3 * u.electron,
        SimpleNamespace(method="nodding"),
        3,
        np.random.default_rng(1234),
    )

    rng = np.random.default_rng(1234)
    shape = (3, 1, 1, 2)
    target_frames = rng.poisson(signals.total.value, size=shape)
    target_read = rng.normal(0, 3 * np.sqrt(2), size=shape)
    sky_frames = rng.poisson(background.value / 2, size=shape)
    sky_read = rng.normal(0, 3, size=shape)
    expected = target_frames + target_read - 2 * (sky_frames + sky_read)

    assert actual.unit == u.electron
    assert actual.shape == shape
    assert np.array_equal(actual.value, expected)


def test_in_field_data_preserves_shared_sky_estimate_covariance() -> None:
    background = np.full((2, 2, 1), 20.0) * u.electron
    signals = Signals(
        target=np.zeros((2, 2, 1)) * u.electron,
        sky=background,
        thermal=np.zeros((2, 2, 1)) * u.electron,
        dark=np.zeros((2, 2, 1)) * u.electron,
        background=background,
        total=background,
    )
    sky_mask = np.array([[True, True], [False, False]])

    data = _sample_data(
        signals,
        SimpleNamespace(n_target=1, n_sky=0),
        0 * u.electron,
        SimpleNamespace(method="in_field", mask=sky_mask),
        20_000,
        np.random.default_rng(8),
    )

    first = data[:, 1, 0, 0].value
    second = data[:, 1, 1, 0].value
    assert np.var(first) == pytest.approx(30.0, rel=0.03)
    assert np.cov(first, second, ddof=0)[0, 1] == pytest.approx(10.0, rel=0.08)


@pytest.mark.parametrize(
    ("center", "pixels", "expected"),
    [
        (5.0, 3, (4, 7)),
        (5.5, 3, (5, 8)),
        (5.0, 2, (5, 7)),
        (5.5, 2, (5, 7)),
    ],
)
def test_centered_bounds_preserve_approved_rounding(
    center: float,
    pixels: int,
    expected: tuple[int, int],
) -> None:
    assert _centered_bounds(center, pixels) == expected


@pytest.mark.parametrize(
    ("psf_shape", "model_shape"),
    [
        ((10, 12), (21, 25)),
        ((11, 13), (20, 24)),
        ((32, 36), (21, 25)),
        ((31, 35), (20, 24)),
    ],
)
def test_prepare_psf_preserves_center_across_padding_cropping_and_parity(
    psf_shape: tuple[int, int],
    model_shape: tuple[int, int],
) -> None:
    y, x = np.indices(psf_shape)
    psf = np.exp(
        -0.5
        * (
            ((x - (psf_shape[1] - 1) / 2) / 1.4) ** 2
            + ((y - (psf_shape[0] - 1) / 2) / 2.1) ** 2
        )
    )
    psf.setflags(write=False)

    prepared = _prepare_psf(psf, *model_shape)

    center_x, center_y = _pixel_center_centroid(prepared)
    assert prepared.shape == model_shape
    assert center_x == pytest.approx(model_shape[1] / 2, abs=1e-3)
    assert center_y == pytest.approx(model_shape[0] / 2, abs=1e-3)
    assert prepared.sum() == pytest.approx(1.0)


def test_spatial_convolution_and_detector_resampling_preserve_fractional_center() -> (
    None
):
    profile = np.zeros((15, 21))
    profile[9:11, 12:14] = 0.25
    y, x = np.indices((9, 9))
    psf = np.exp(-0.5 * (((x - 4) / 1.2) ** 2 + ((y - 4) / 1.8) ** 2))

    convolved = _convolve_spatial(profile, psf)
    selection = SimpleNamespace(
        scale=SimpleNamespace(
            spaxels_y=3,
            spaxels_x=4,
            spaxel_scale=50 * u.mas,
        )
    )
    detector, _ = _resample_spatial(convolved, 10 * u.mas, selection)

    assert _pixel_center_centroid(convolved) == pytest.approx((13.0, 10.0))
    assert _pixel_center_centroid(detector) == pytest.approx((2.5, 2.0))
    assert convolved.sum() == pytest.approx(1.0)
    assert detector.sum() == pytest.approx(1.0)


def test_doppler_shift_matches_independent_interpolation_and_conserves_flux() -> None:
    wavelength = np.linspace(0.95, 1.05, 2001) * u.micron
    spectrum = np.exp(-0.5 * ((wavelength.to_value(u.micron) - 1.0) / 0.004) ** 2) * u.J
    velocity = np.array([[-180.0, 0.0, 220.0]]) * u.km / u.s

    actual = _shift_spectrum_by_velocity(
        wavelength,
        spectrum,
        velocity,
        wavelength,
    )

    expected = np.empty(actual.shape)
    for index, value in enumerate(velocity[0]):
        factor = 1 + value.to_value(u.km / u.s) / 299792.458
        expected[0, index] = (
            np.interp(
                wavelength.to_value(u.micron) / factor,
                wavelength.to_value(u.micron),
                spectrum.value,
                left=0.0,
                right=0.0,
            )
            / factor
        )

    assert np.allclose(actual.value, expected, rtol=2e-11, atol=2e-14)
    integrals = actual.sum(axis=2)
    assert np.allclose(integrals.value, spectrum.sum().value, rtol=3e-6)
    centroids = (actual * wavelength[None, None, :]).sum(axis=2) / actual.sum(axis=2)
    assert centroids[0, 0] < centroids[0, 1] < centroids[0, 2]


def test_varying_velocity_operator_matches_independent_small_array() -> None:
    high_wavelength = np.arange(0.97, 1.031, 0.002) * u.micron
    wavelength = np.array([0.985, 1.005, 1.025]) * u.micron
    edges = np.array([0.975, 0.995, 1.015, 1.035]) * u.micron
    grid = SpectralGrid(
        wavelength=wavelength,
        edges=edges,
        step=0.02 * u.micron,
        high_wavelength=high_wavelength,
        high_step=0.002 * u.micron,
        resolution_fwhm=0.004 * u.micron,
        high_sigma_pixels=0.8,
        photon_energy=np.ones(3) * u.J,
    )
    scale = SimpleNamespace(
        spaxels_y=3,
        spaxels_x=3,
        spaxel_scale=1 * u.arcsec,
    )
    selection = SimpleNamespace(scale=scale)
    spatial = np.zeros((3, 3))
    spatial[1, 0] = 0.5
    spatial[1, 2] = 0.5
    spectrum = (
        np.exp(-0.5 * ((high_wavelength.to_value(u.micron) - 1.0) / 0.003) ** 2) * u.J
    )
    velocity = np.zeros((3, 3)) * u.km / u.s
    velocity[1, 0] = -900 * u.km / u.s
    velocity[1, 2] = 900 * u.km / u.s
    psf = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.5, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )

    actual = _assemble_varying_velocity_cube(
        selection,
        grid,
        spatial,
        1 * u.arcsec,
        spectrum,
        velocity,
        psf,
    )

    intrinsic = np.zeros((3, 3, len(high_wavelength)))
    for y in range(3):
        for x in range(3):
            factor = 1 + velocity[y, x].to_value(u.km / u.s) / 299792.458
            intrinsic[y, x] = (
                spatial[y, x]
                * np.interp(
                    high_wavelength.to_value(u.micron) / factor,
                    high_wavelength.to_value(u.micron),
                    spectrum.value,
                    left=0.0,
                    right=0.0,
                )
                / factor
            )
    spatially_convolved = np.stack(
        [
            fftconvolve(intrinsic[:, :, index], psf, mode="same")
            for index in range(len(high_wavelength))
        ],
        axis=2,
    )
    convolved = ndimage.gaussian_filter1d(
        spatially_convolved,
        sigma=grid.high_sigma_pixels,
        axis=2,
        mode="reflect",
    )
    convolved *= spatially_convolved.sum() / convolved.sum()
    expected = np.zeros((3, 3, 3))
    bins = np.digitize(high_wavelength.value, edges.value, right=True) - 1
    for index, spectral_bin in enumerate(bins):
        if 0 <= spectral_bin < 3:
            expected[:, :, spectral_bin] += convolved[:, :, index]
    expected *= grid.high_step.to_value(u.micron) / grid.step.to_value(u.micron)

    assert np.allclose(actual.value, expected, rtol=2e-13, atol=2e-15)
    blue_x = np.sum(actual.value * wavelength.value[None, None, :], axis=2)
    assert blue_x[1, 0] < blue_x[1, 2]

    unshifted = _assemble_varying_velocity_cube(
        selection,
        grid,
        spatial,
        1 * u.arcsec,
        spectrum,
        np.zeros((3, 3)) * u.km / u.s,
        psf,
    ).value
    detector_shifted = np.zeros_like(unshifted)
    for y in range(3):
        for x in range(3):
            factor = 1 + velocity[y, x].to_value(u.km / u.s) / 299792.458
            detector_shifted[y, x] = (
                np.interp(
                    wavelength.to_value(u.micron) / factor,
                    wavelength.to_value(u.micron),
                    unshifted[y, x],
                    left=0.0,
                    right=0.0,
                )
                / factor
            )
    ordering_difference = np.abs(actual.value - detector_shifted).sum()
    ordering_difference /= np.abs(actual.value).sum()
    assert ordering_difference == pytest.approx(0.2220087519, rel=2e-9)


def test_velocity_grid_discards_margins_without_changing_default_edge_clipping() -> (
    None
):
    grid = SpectralGrid(
        wavelength=np.array([1.0, 1.1]) * u.micron,
        edges=np.array([0.95, 1.05, 1.15]) * u.micron,
        step=0.1 * u.micron,
        high_wavelength=np.array([0.9, 1.0, 1.1, 1.2]) * u.micron,
        high_step=0.1 * u.micron,
        resolution_fwhm=0.1 * u.micron,
        high_sigma_pixels=1.0,
        photon_energy=np.ones(2) * u.J,
    )
    spectrum = np.ones(4) * u.J

    clipped = _resample_spectrum(grid, spectrum)
    cropped = _resample_spectrum(grid, spectrum, discard_outside=True)

    assert np.array_equal(clipped.value, [2.0, 2.0])
    assert np.array_equal(cropped.value, [1.0, 1.0])


def _pixel_center_centroid(array: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(array.shape)
    total = array.sum()
    return (
        float(((x + 0.5) * array).sum() / total),
        float(((y + 0.5) * array).sum() / total),
    )
