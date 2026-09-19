"""Public target-model contract tests."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import QTable

import cubesim


def test_air_wavelength_is_converted_before_redshift() -> None:
    model = cubesim.GaussianLines(
        rest_wavelength=6562.8 * u.angstrom,
        redshift=2.0,
        medium="air",
        flux=1e-17 * u.erg / (u.s * u.cm**2),
        dispersion=40 * u.km / u.s,
    )

    assert model.rest_wavelength[0].to_value(u.angstrom) == pytest.approx(
        6564.61298,
        rel=2e-8,
    )
    assert model.wavelength[0] == 3 * model.rest_wavelength[0]


def test_spectral_models_own_read_only_quantities() -> None:
    wavelength = np.array([1.0, 1.1]) * u.micron
    density = (
        np.array([1.0, 2.0])
        * 1e-20
        * u.erg
        / (u.s * u.cm**2 * u.micron)
    )
    tabulated = cubesim.TabulatedSpectrum(wavelength=wavelength, flux=density)
    lines = cubesim.GaussianLines(
        wavelength=np.array([1.05, 1.1]) * u.micron,
        flux=np.array([1.0, 2.0]) * 1e-17 * u.erg / (u.s * u.cm**2),
        dispersion=40 * u.km / u.s,
    )

    wavelength[0] = 2 * u.micron
    density[0] *= 3

    assert tabulated.wavelength[0] == 1 * u.micron
    assert tabulated.flux[0].to_value(density.unit) == 1e-20
    for quantity in (
        tabulated.wavelength,
        tabulated.flux,
        lines.wavelength,
        lines.flux,
        lines.dispersion,
    ):
        assert not quantity.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        lines.flux[0] *= 2


@pytest.mark.parametrize("velocity", [-299792.458, 299792.458])
def test_constant_velocity_rejects_speed_of_light(velocity: float) -> None:
    with pytest.raises(ValueError, match="less than the speed of light"):
        cubesim.ConstantVelocity(velocity * u.km / u.s)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"effective_radius": 0 * u.arcsec, "index": 1.0},
        {"effective_radius": 1 * u.arcsec, "index": 0.0},
        {"effective_radius": 1 * u.arcsec, "index": 1.0, "axis_ratio": 1.1},
    ],
)
def test_sersic_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        cubesim.Sersic(**kwargs)


def test_uniform_surface_brightness_runs_without_psf(instrument_data) -> None:
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Uniform(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2 * u.arcsec**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_exposure(time=100 * u.s, n_target=2)

    result = etc.run(include_models=True)

    assert result.options.psf_path is None
    assert result.options.psf_pixel_scale is None
    assert np.array_equal(result.models.targets[0].low.spatial, np.ones((3, 4)))
    assert np.all(result.models.combined == result.models.combined[0, 0])


def test_uniform_and_integrated_flux_pairing_is_enforced(instrument_data) -> None:
    etc = _base_etc(instrument_data)
    integrated = cubesim.GaussianLines(
        wavelength=1.1 * u.micron,
        flux=1e-17 * u.erg / (u.s * u.cm**2),
        dispersion=40 * u.km / u.s,
    )

    with pytest.raises(u.UnitConversionError, match="surface-brightness"):
        etc.add_target(
            position=(0 * u.arcsec, 0 * u.arcsec),
            spatial=cubesim.Uniform(),
            spectrum=integrated,
        )


def test_sersic_and_constant_velocity_use_existing_compute_operators(
    instrument_data,
) -> None:
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Sersic(
            effective_radius=0.08 * u.arcsec,
            index=1.0,
            axis_ratio=0.7,
            position_angle=25 * u.deg,
        ),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
        velocity=cubesim.ConstantVelocity(offset=100 * u.km / u.s),
    )
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    result = etc.run(include_models=True)
    target = result.models.targets[0]

    assert 0 < target.high.spatial.sum() < 1
    assert np.all(target.high.velocity == 100 * u.km / u.s)
    assert np.all(target.low.velocity == 100 * u.km / u.s)
    peak = target.high.wavelength[np.argmax(target.high.spectrum)]
    expected = 1.1 * u.micron * (1 + (100 * u.km / u.s) / (299792.458 * u.km / u.s))
    assert abs(peak - expected) < 0.5 * np.diff(target.high.wavelength[:2])[0]


def test_analytic_spatial_profile_retains_only_in_field_flux(instrument_data) -> None:
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Gaussian(fwhm=10 * u.arcsec),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_psf(np.ones((1, 1)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    target = etc.run(include_models=True).models.targets[0]

    assert np.isfinite(target.high.spatial).all()
    assert 0 < target.high.spatial.sum() < 0.001


def test_out_of_band_gaussian_line_contributes_zero(instrument_data) -> None:
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=100 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_psf(np.ones((1, 1)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    result = etc.run(include_models=True)

    assert np.isfinite(result.models.combined).all()
    assert np.all(result.models.combined == 0)
    assert np.all(result.snr == 0)


def test_partially_clipped_point_preserves_resampled_flux(instrument_data) -> None:
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(-0.102 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_psf(np.ones((1, 1)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    target = etc.run(include_models=True).models.targets[0]

    assert target.high.spatial.sum() == pytest.approx(0.8)
    assert target.low.spatial.sum() == pytest.approx(0.4)


def test_clipped_point_velocity_cube_preserves_resampled_flux(instrument_data) -> None:
    results = []
    for velocity in (
        cubesim.ConstantVelocity(0 * u.km / u.s),
        cubesim.VelocityField(
            np.zeros((43, 43)) * u.km / u.s,
            pixel_scale=10 * u.mas,
        ),
    ):
        etc = _base_etc(instrument_data)
        etc.add_target(
            position=(-0.102 * u.arcsec, 0 * u.arcsec),
            spatial=cubesim.Point(),
            spectrum=cubesim.GaussianLines(
                wavelength=1.1 * u.micron,
                flux=1e-17 * u.erg / (u.s * u.cm**2),
                dispersion=40 * u.km / u.s,
            ),
            velocity=velocity,
        )
        etc.set_psf(np.ones((1, 1)), pixel_scale=10 * u.mas)
        etc.set_exposure(time=100 * u.s, n_target=2)
        results.append(etc.run(include_models=True))

    constant, field = results
    reference = np.abs(constant.models.combined.value).sum()
    difference = np.abs(
        field.models.combined.value - constant.models.combined.value
    ).sum()

    assert field.models.targets[0].low.spatial.sum() == pytest.approx(0.4)
    assert difference / reference < 0.005


def test_tabulated_spectrum_runs_on_independent_grid(instrument_data) -> None:
    wavelength = np.linspace(0.94, 1.36, 50) * u.micron
    spectrum = cubesim.TabulatedSpectrum(
        wavelength=wavelength,
        flux=np.ones(50) * 1e-20 * u.erg / (u.s * u.cm**2 * u.micron),
    )
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=spectrum,
    )
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    result = etc.run(include_models=True)

    high = result.models.targets[0].high.spectrum
    assert np.allclose(high.value, high.value[0])


def test_tabulated_spectrum_reads_ecsv_medium_metadata(tmp_path) -> None:
    path = tmp_path / "spectrum.ecsv"
    table = QTable(
        [
            [0.9, 1.0, 1.1] * u.micron,
            np.array([1, 2, 1]) * 1e-20 * u.erg / (u.s * u.cm**2 * u.micron),
        ],
        names=("wavelength", "flux"),
        meta={"medium": "air"},
    )
    table.write(path, format="ascii.ecsv")

    spectrum = cubesim.TabulatedSpectrum(path)

    assert spectrum.medium == "air"
    assert np.all(spectrum.wavelength > table["wavelength"])
    with pytest.raises(ValueError, match="conflicts"):
        cubesim.TabulatedSpectrum(path, medium="vacuum")


def test_tabulated_spectrum_reads_fits_coordinate_metadata(tmp_path) -> None:
    path = tmp_path / "spectrum.fits"
    table = QTable(
        [
            [0.9, 1.0, 1.1] * u.micron,
            np.array([1, 2, 1]) * 1e-20 * u.erg / (u.s * u.cm**2 * u.micron),
        ],
        names=("wavelength", "flux"),
    )
    table.write(path, format="fits")
    with fits.open(path, mode="update") as hdus:
        hdus[1].header["TCTYP1"] = "AWAV"

    spectrum = cubesim.TabulatedSpectrum(path)

    assert spectrum.medium == "air"
    assert np.all(spectrum.wavelength > table["wavelength"])
    with pytest.raises(ValueError, match="conflicts"):
        cubesim.TabulatedSpectrum(path, medium="vacuum")


def test_tabulated_spectrum_must_cover_selected_disperser(instrument_data) -> None:
    spectrum = cubesim.TabulatedSpectrum(
        wavelength=np.linspace(1.0, 1.2, 10) * u.micron,
        flux=np.ones(10) * 1e-20 * u.erg / (u.s * u.cm**2 * u.micron),
    )
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=spectrum,
    )
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    with pytest.raises(ValueError, match="does not cover"):
        etc.run()


def test_spatial_image_is_copied_normalized_and_sky_oriented(
    instrument_data,
) -> None:
    image = np.zeros((3, 3))
    image[2, 1] = 2.0
    original = image.copy()
    model = cubesim.SpatialImage(
        image,
        pixel_scale=10 * u.mas,
        position_angle=90 * u.deg,
    )
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=model,
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_psf(np.ones((3, 3)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    target = etc.run(include_models=True).models.targets[0]
    y, x = np.unravel_index(np.argmax(target.high.spatial), target.high.spatial.shape)

    assert np.array_equal(image, original)
    assert model.data.sum() == pytest.approx(1.0)
    assert x < (target.high.spatial.shape[1] - 1) / 2
    assert y == pytest.approx((target.high.spatial.shape[0] - 1) / 2, abs=1)


def test_spatial_image_fits_wcs_must_match_pixel_scale(tmp_path) -> None:
    path = tmp_path / "spatial.fits"
    header = fits.Header(
        {
            "PIXSCALE": 10.0,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 120.0,
            "CRVAL2": 25.0,
            "CDELT1": -20 / 3_600_000,
            "CDELT2": 20 / 3_600_000,
        }
    )
    fits.writeto(path, np.ones((3, 3)), header=header)

    with pytest.raises(ValueError, match="conflicts"):
        cubesim.SpatialImage(path)


def test_target_position_rotates_with_ifu_orientation(instrument_data) -> None:
    north_up = _positioned_spatial(
        instrument_data,
        cubesim.Point(),
        position=(0.05 * u.arcsec, 0 * u.arcsec),
        pointing_angle=0 * u.deg,
    )
    north_right = _positioned_spatial(
        instrument_data,
        cubesim.Point(),
        position=(0.05 * u.arcsec, 0 * u.arcsec),
        pointing_angle=90 * u.deg,
    )
    first_y, first_x = _centroid(north_up)
    second_y, second_x = _centroid(north_right)
    center_y, center_x = (np.array(north_up.shape) - 1) / 2

    assert first_x == pytest.approx(center_x - 5)
    assert first_y == pytest.approx(center_y)
    assert second_x == pytest.approx(center_x)
    assert second_y == pytest.approx(center_y + 5)


def test_point_target_centroid_is_preserved_through_model_stages(instrument_data) -> (
    None
):
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(-0.025 * u.arcsec, 0.025 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    y, x = np.indices((9, 9))
    psf = np.exp(-0.5 * (((x - 4) / 1.2) ** 2 + ((y - 4) / 1.8) ** 2))
    etc.set_psf(psf, pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    target = etc.run(include_models=True).models.targets[0]

    assert _pixel_center_centroid(target.high.spatial) == pytest.approx((13.0, 10.0))
    assert _pixel_center_centroid(target.high.spatial_convolved) == pytest.approx(
        (13.0, 10.0)
    )
    assert _pixel_center_centroid(target.low.spatial) == pytest.approx((2.5, 2.0))
    assert target.high.spatial.sum() == pytest.approx(1.0)
    assert target.high.spatial_convolved.sum() == pytest.approx(1.0)
    assert target.low.spatial.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("model_name", ["gaussian", "sersic"])
def test_analytic_morphology_rotates_with_ifu_orientation(
    instrument_data,
    model_name: str,
) -> None:
    if model_name == "gaussian":
        spatial = cubesim.Gaussian(
            fwhm=0.08 * u.arcsec,
            axis_ratio=0.4,
            position_angle=0 * u.deg,
        )
    else:
        spatial = cubesim.Sersic(
            effective_radius=0.06 * u.arcsec,
            index=1.0,
            axis_ratio=0.4,
            position_angle=0 * u.deg,
        )
    north_up = _positioned_spatial(
        instrument_data,
        spatial,
        position=(0 * u.arcsec, 0 * u.arcsec),
        pointing_angle=0 * u.deg,
    )
    north_right = _positioned_spatial(
        instrument_data,
        spatial,
        position=(0 * u.arcsec, 0 * u.arcsec),
        pointing_angle=90 * u.deg,
    )
    first_y, first_x = _variances(north_up)
    second_y, second_x = _variances(north_right)

    assert first_y > first_x
    assert second_x > second_y


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "maximum_velocity": 0 * u.km / u.s,
            "turnover_radius": 0.1 * u.arcsec,
            "inclination": 45 * u.deg,
            "position_angle": 0 * u.deg,
        },
        {
            "maximum_velocity": 200 * u.km / u.s,
            "turnover_radius": 0 * u.arcsec,
            "inclination": 45 * u.deg,
            "position_angle": 0 * u.deg,
        },
        {
            "maximum_velocity": 200 * u.km / u.s,
            "turnover_radius": 0.1 * u.arcsec,
            "inclination": 90 * u.deg,
            "position_angle": 0 * u.deg,
        },
    ],
)
def test_rotating_disk_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        cubesim.RotatingDisk(**kwargs)


def test_velocity_field_supports_memory_npy_and_fits(tmp_path) -> None:
    source = np.arange(25, dtype=float).reshape(5, 5)
    memory = cubesim.VelocityField(source * u.km / u.s, pixel_scale=10 * u.mas)
    source[0, 0] = -1

    npy_path = tmp_path / "velocity.npy"
    np.save(npy_path, np.arange(25, dtype=float).reshape(5, 5))
    npy = cubesim.VelocityField(
        npy_path,
        unit=u.m / u.s,
        pixel_scale=20 * u.mas,
        position_angle=30 * u.deg,
    )

    fits_path = tmp_path / "velocity.fits"
    header = fits.Header({"BUNIT": "km s-1", "PIXSCALE": 15.0})
    fits.writeto(fits_path, np.ones((5, 5)), header=header)
    fits_field = cubesim.VelocityField(fits_path)

    assert memory.data[0, 0] == 0 * u.km / u.s
    assert not memory.data.flags.writeable
    assert npy.data[0, 1].to_value(u.km / u.s) == pytest.approx(0.001)
    assert npy.pixel_scale == 20 * u.mas
    assert npy.position_angle == 30 * u.deg
    assert fits_field.data.unit == u.km / u.s
    assert fits_field.pixel_scale == 15 * u.mas


def test_velocity_field_rejects_incomplete_or_invalid_inputs(tmp_path) -> None:
    npy_path = tmp_path / "velocity.npy"
    np.save(npy_path, np.ones((3, 3)))
    with pytest.raises(ValueError, match="require unit and pixel_scale"):
        cubesim.VelocityField(npy_path, pixel_scale=10 * u.mas)
    with pytest.raises(u.UnitConversionError, match="velocity unit"):
        cubesim.VelocityField(
            npy_path,
            unit=u.m,
            pixel_scale=10 * u.mas,
        )
    with pytest.raises(ValueError, match="finite"):
        cubesim.VelocityField(
            np.array([[np.nan]]) * u.km / u.s,
            pixel_scale=10 * u.mas,
        )


def test_rotating_disk_sign_tracks_ifu_orientation(instrument_data) -> None:
    disk = cubesim.RotatingDisk(
        maximum_velocity=200 * u.km / u.s,
        turnover_radius=0.05 * u.arcsec,
        inclination=60 * u.deg,
        position_angle=0 * u.deg,
        systemic_velocity=20 * u.km / u.s,
    )
    north_up = _velocity_etc(instrument_data, disk, pointing_angle=0 * u.deg)
    north_right = _velocity_etc(instrument_data, disk, pointing_angle=90 * u.deg)

    first = north_up.run(include_models=True).models.targets[0].high.velocity
    second = north_right.run(include_models=True).models.targets[0].high.velocity
    cy, cx = np.array(first.shape) // 2

    assert first[cy + 3, cx] > 20 * u.km / u.s
    assert first[cy - 3, cx] < 20 * u.km / u.s
    assert second[cy, cx + 3] > 20 * u.km / u.s
    assert second[cy, cx - 3] < 20 * u.km / u.s


def test_supplied_velocity_field_rotates_and_must_cover_ifu(instrument_data) -> None:
    field = np.repeat(np.arange(-20, 21)[:, None], 41, axis=1) * u.km / u.s
    model = cubesim.VelocityField(
        field,
        pixel_scale=10 * u.mas,
        position_angle=0 * u.deg,
    )
    etc = _velocity_etc(instrument_data, model, pointing_angle=90 * u.deg)

    velocity = etc.run(include_models=True).models.targets[0].high.velocity
    cy, cx = np.array(velocity.shape) // 2

    assert velocity[cy, cx + 3] > velocity[cy, cx - 3]
    assert velocity[cy + 3, cx] == pytest.approx(velocity[cy - 3, cx])

    too_small = cubesim.VelocityField(
        np.zeros((3, 3)) * u.km / u.s,
        pixel_scale=10 * u.mas,
    )
    with pytest.raises(ValueError, match="does not cover"):
        _velocity_etc(instrument_data, too_small).run()


def test_uniform_velocity_field_matches_constant_velocity_within_sampling(
    instrument_data,
) -> None:
    constant = cubesim.ConstantVelocity(offset=120 * u.km / u.s)
    field = cubesim.VelocityField(
        np.full((41, 41), 120.0) * u.km / u.s,
        pixel_scale=10 * u.mas,
    )
    constant_result = _velocity_etc(instrument_data, constant).run(include_models=True)
    field_result = _velocity_etc(instrument_data, field).run(include_models=True)

    difference = np.abs(
        field_result.models.combined.value - constant_result.models.combined.value
    ).sum()
    reference = np.abs(constant_result.models.combined.value).sum()
    assert difference / reference < 0.005
    assert np.isclose(
        field_result.models.combined.sum().value,
        constant_result.models.combined.sum().value,
        rtol=2e-4,
    )
    with pytest.raises(ValueError, match="read-only"):
        field_result.options.targets[0].velocity.data[0, 0] = 0 * u.km / u.s


def test_velocity_margin_requires_tabulated_spectrum_coverage(instrument_data) -> None:
    spectrum = cubesim.TabulatedSpectrum(
        wavelength=np.linspace(0.95, 1.35, 100) * u.micron,
        flux=np.ones(100) * 1e-20 * u.erg / (u.s * u.cm**2 * u.micron),
    )
    field = cubesim.VelocityField(
        np.zeros((41, 41)) * u.km / u.s,
        pixel_scale=10 * u.mas,
    )
    etc = _base_etc(instrument_data)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Gaussian(fwhm=0.08 * u.arcsec),
        spectrum=spectrum,
        velocity=field,
    )
    etc.set_psf(np.ones((3, 3)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    with pytest.raises(ValueError, match="does not cover"):
        etc.run()


def _base_etc(instrument_data) -> cubesim.Etc:
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    return etc


def _velocity_etc(
    instrument_data,
    velocity,
    *,
    pointing_angle=0 * u.deg,
) -> cubesim.Etc:
    etc = _base_etc(instrument_data)
    etc.set_pointing(position_angle=pointing_angle)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Gaussian(fwhm=0.08 * u.arcsec),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
        velocity=velocity,
    )
    etc.set_psf(np.ones((3, 3)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    return etc


def _positioned_spatial(
    instrument_data,
    spatial,
    *,
    position,
    pointing_angle,
) -> np.ndarray:
    etc = _base_etc(instrument_data)
    etc.set_pointing(position_angle=pointing_angle)
    etc.add_target(
        position=position,
        spatial=spatial,
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    psf = np.zeros((3, 3))
    psf[1, 1] = 1
    etc.set_psf(psf, pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    return etc.run(include_models=True).models.targets[0].high.spatial


def _centroid(profile: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(profile.shape)
    return float((y * profile).sum()), float((x * profile).sum())


def _pixel_center_centroid(profile: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(profile.shape)
    total = profile.sum()
    return (
        float(((x + 0.5) * profile).sum() / total),
        float(((y + 0.5) * profile).sum() / total),
    )


def _variances(profile: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(profile.shape)
    center_y, center_x = _centroid(profile)
    return (
        float(((y - center_y) ** 2 * profile).sum()),
        float(((x - center_x) ** 2 * profile).sum()),
    )
