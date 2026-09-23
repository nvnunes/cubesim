"""Telescope, IFU, and target coordinate contracts."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import cubesim


def _etc(instrument_data) -> cubesim.Etc:
    etc = cubesim.Etc(instrument_data)
    etc.configure(scale="50mas", disperser="r3000.yj", atmosphere="airmass10_pwv10")
    psf = np.ones((1, 1))
    etc.set_psf(psf, pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    return etc


def _add_point(etc: cubesim.Etc, **position) -> None:
    etc.add_target(
        **position,
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )


def _centroid(image: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(image.shape)
    return float((x * image).sum() / image.sum()), float((y * image).sum() / image.sum())


def test_positive_ifu_axes_follow_columns_and_rows(instrument_data) -> None:
    etc = _etc(instrument_data)
    _add_point(etc, ifu_offset=(0 * u.mas, 0 * u.mas))
    _add_point(etc, ifu_offset=(10 * u.mas, 0 * u.mas))
    _add_point(etc, ifu_offset=(0 * u.mas, 10 * u.mas))
    _add_point(etc, pointing_offset=(10 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(0 * u.mas, 10 * u.mas))
    models = etc.run(include_models=True).models.targets
    center, along_x, along_y, pointing_x, pointing_y = (
        _centroid(model.high.spatial) for model in models
    )
    assert along_x[0] > center[0]
    assert along_x[1] == pytest.approx(center[1], abs=1e-12)
    assert along_y[1] > center[1]
    assert along_y[0] == pytest.approx(center[0], abs=1e-12)
    assert pointing_x == pytest.approx(along_x)
    assert pointing_y == pytest.approx(along_y)


def test_polar_pointing_offsets_match_cartesian_with_rotated_pointing(instrument_data) -> None:
    etc = _etc(instrument_data)
    etc.set_pointing(position_angle=35 * u.deg)
    etc.set_ifu_position(pointing_offset=(0.05 * u.arcsec, 30 * u.deg))
    _add_point(etc, pointing_offset=(0.07 * u.arcsec, 120 * u.deg))
    _add_point(
        etc,
        pointing_offset=(-0.035 * u.arcsec, 0.0606217782649107 * u.arcsec),
    )
    result = etc.run(include_models=True)

    assert result.options.ifu_position.pointing_offset[0].to_value(u.arcsec) == pytest.approx(
        0.04330127018922193
    )
    assert result.options.ifu_position.pointing_offset[1].to_value(u.arcsec) == pytest.approx(
        0.025
    )
    np.testing.assert_allclose(
        result.models.targets[0].high.spatial,
        result.models.targets[1].high.spatial,
        atol=1e-12,
    )


def test_pointing_offset_unit_pair_selects_polar_form(instrument_data) -> None:
    etc = _etc(instrument_data)
    etc.set_ifu_position(pointing_offset=(0.1 * u.arcsec, 0.2 * u.arcsec))
    _add_point(etc, ifu_offset=(0 * u.arcsec, 0 * u.arcsec))
    result = etc.run()
    assert result.options.ifu_position.pointing_offset[0] == 0.1 * u.arcsec
    assert result.options.ifu_position.pointing_offset[1] == 0.2 * u.arcsec

    with pytest.raises(ValueError, match="r must be nonnegative"):
        etc.set_ifu_position(pointing_offset=(-1 * u.arcsec, 30 * u.deg))
    with pytest.raises(ValueError, match="theta must be finite"):
        etc.set_ifu_position(pointing_offset=(1 * u.arcsec, np.nan * u.deg))


def test_relative_ifu_rotation_maps_pointing_axes(instrument_data) -> None:
    etc = _etc(instrument_data)
    etc.set_pointing(position_angle=30 * u.deg)
    etc.set_ifu_position(pointing_offset=(0 * u.mas, 0 * u.mas), rotation=90 * u.deg)
    _add_point(etc, pointing_offset=(0 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(10 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(0 * u.mas, 10 * u.mas))
    models = etc.run(include_models=True).models.targets
    center, pointing_x, pointing_y = (_centroid(model.high.spatial) for model in models)
    assert pointing_x[0] == pytest.approx(center[0], abs=1e-12)
    assert pointing_x[1] < center[1]
    assert pointing_y[0] > center[0]
    assert pointing_y[1] == pytest.approx(center[1], abs=1e-12)


def test_pointing_and_ifu_offsets_are_equivalent_at_zero_rotation(instrument_data) -> None:
    etc = _etc(instrument_data)
    _add_point(etc, ifu_offset=(10 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(10 * u.mas, 0 * u.mas))
    models = etc.run(include_models=True).models.targets
    np.testing.assert_array_equal(models[0].high.spatial, models[1].high.spatial)


def test_target_forms_and_ifu_placement_resolve_at_run(instrument_data) -> None:
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    etc = _etc(instrument_data)
    etc.set_pointing(sky_position=center, position_angle=30 * u.deg)
    _add_point(etc, ifu_offset=(0 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(0 * u.mas, 0 * u.mas))
    _add_point(etc, sky_position=center)
    first = etc.run(include_models=True)
    etc.set_ifu_position(pointing_offset=(10 * u.mas, 0 * u.mas), rotation=20 * u.deg)
    second = etc.run(include_models=True)

    assert first.options.ifu_center.separation(center) < 1e-6 * u.mas
    assert second.options.ifu_position_angle == 50 * u.deg
    first_positions = [_centroid(model.high.spatial) for model in first.models.targets]
    second_positions = [_centroid(model.high.spatial) for model in second.models.targets]
    assert second_positions[0] == pytest.approx(first_positions[0])
    assert second_positions[1][0] < first_positions[1][0]
    assert second_positions[2] == pytest.approx(second_positions[1], abs=1e-5)
    assert first.options.ifu_position.pointing_offset[0] == 0 * u.deg
    with pytest.raises(ValueError, match="read-only"):
        second.options.ifu_position.pointing_offset[0][...] = 1 * u.deg

    east = -10 * u.mas * np.cos(np.deg2rad(30))
    north = 10 * u.mas * np.sin(np.deg2rad(30))
    expected_center = center.spherical_offsets_by(east, north)
    assert second.options.ifu_center.separation(expected_center) < 1e-6 * u.mas

    etc.set_ifu_position(sky_position=expected_center, rotation=20 * u.deg)
    sky_placed = etc.run(include_models=True)
    np.testing.assert_allclose(
        sky_placed.models.targets[1].high.spatial,
        second.models.targets[1].high.spatial,
        atol=1e-9,
    )


def test_target_form_follows_its_reference_frame(instrument_data) -> None:
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    etc = _etc(instrument_data)
    etc.set_pointing(sky_position=center)
    _add_point(etc, ifu_offset=(10 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(10 * u.mas, 0 * u.mas))
    _add_point(etc, sky_position=center.spherical_offsets_by(-10 * u.mas, 0 * u.mas))
    first = etc.run(include_models=True)
    etc.set_pointing(sky_position=center, position_angle=90 * u.deg)
    second = etc.run(include_models=True)
    assert _centroid(first.models.targets[0].high.spatial) == pytest.approx(
        _centroid(second.models.targets[0].high.spatial)
    )
    assert _centroid(first.models.targets[1].high.spatial) == pytest.approx(
        _centroid(second.models.targets[1].high.spatial)
    )
    assert _centroid(first.models.targets[2].high.spatial) != pytest.approx(
        _centroid(second.models.targets[2].high.spatial)
    )


def test_non_cardinal_off_axis_geometry_matches_sky_and_detector_values(
    instrument_data, tmp_path
) -> None:
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    expected_target = center.spherical_offsets_by(
        -38.30127018922194 * u.mas, 33.66025403784438 * u.mas
    )
    etc = _etc(instrument_data)
    etc.set_pointing(sky_position=center, position_angle=30 * u.deg)
    etc.set_ifu_position(pointing_offset=(40 * u.mas, -20 * u.mas), rotation=20 * u.deg)
    _add_point(etc, pointing_offset=(50 * u.mas, 10 * u.mas))
    _add_point(
        etc,
        ifu_offset=(19.657530507629147 * u.mas, 24.770577190320566 * u.mas),
    )
    _add_point(etc, sky_position=expected_target)
    result = etc.run(include_models=True)

    # The target is (10, 30) mas from the IFU in pointing axes. Rotating
    # through -20 degrees places it at these detector-axis offsets.
    detector_x_mas = 19.657530507629147
    detector_y_mas = 24.770577190320566
    image = result.models.targets[0].high.spatial
    assert _centroid(image) == pytest.approx(
        (
            (image.shape[1] - 1) / 2 + detector_x_mas / 10,
            (image.shape[0] - 1) / 2 + detector_y_mas / 10,
        ),
        abs=1e-10,
    )
    for model in result.models.targets[1:]:
        assert _centroid(model.high.spatial) == pytest.approx(_centroid(image), abs=1e-5)

    # Independent sky offsets from the telescope center, east then north.
    expected_ifu = center.spherical_offsets_by(
        -44.64101615137755 * u.mas, 2.679491924311222 * u.mas
    )
    assert result.options.ifu_center.separation(expected_ifu) < 1e-6 * u.mas

    path = tmp_path / "off_axis.fits"
    result.save(path)
    with fits.open(path) as hdus:
        wcs = WCS(hdus["SNR"].header)
        reference = np.array(
            [0, (result.snr.shape[1] - 1) / 2, (result.snr.shape[0] - 1) / 2]
        )
        pixels = np.array(
            [
                reference,
                reference + [0, 1, 0],
                reference + [0, 0, 1],
                reference + [0, detector_x_mas / 50, detector_y_mas / 50],
            ]
        )
        world = wcs.all_pix2world(pixels, 0)
    sky = SkyCoord(world[:, 1] * u.deg, world[:, 2] * u.deg, frame="icrs")
    assert sky[0].separation(expected_ifu) < 1e-5 * u.mas
    assert sky[3].separation(expected_target) < 1e-4 * u.mas

    column_east, column_north = expected_ifu.spherical_offsets_to(sky[1])
    row_east, row_north = expected_ifu.spherical_offsets_to(sky[2])
    assert column_east.to_value(u.mas) == pytest.approx(-32.13938048432697, abs=1e-5)
    assert column_north.to_value(u.mas) == pytest.approx(38.302222155948904, abs=1e-5)
    assert row_east.to_value(u.mas) == pytest.approx(38.302222155948904, abs=1e-5)
    assert row_north.to_value(u.mas) == pytest.approx(32.13938048432697, abs=1e-5)


def test_sky_and_pointing_positions_agree_across_field_of_regard(instrument_data) -> None:
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    ifu_sky = center.spherical_offsets_by(
        -35.98076211353316 * u.arcsec, -2.3205080756887764 * u.arcsec
    )
    target_sky = center.spherical_offsets_by(
        -36.01906338372238 * u.arcsec, -2.2868478216509303 * u.arcsec
    )
    etc = _etc(instrument_data)
    etc.set_pointing(sky_position=center, position_angle=30 * u.deg)
    etc.set_ifu_position(
        pointing_offset=(30 * u.arcsec, -20 * u.arcsec), rotation=20 * u.deg
    )
    _add_point(etc, pointing_offset=(30.05 * u.arcsec, -19.99 * u.arcsec))
    _add_point(etc, sky_position=target_sky)
    result = etc.run(include_models=True)
    pointing_pixel, sky_pixel = (
        _centroid(model.high.spatial) for model in result.models.targets
    )
    # Spherical offsets and local tangent-plane differences need not be
    # identical across the field, but should agree far below a high-res pixel.
    assert sky_pixel == pytest.approx(pointing_pixel, abs=0.002)

    etc.set_ifu_position(sky_position=ifu_sky, rotation=20 * u.deg)
    sky_placed = etc.run(include_models=True)
    for before, after in zip(result.models.targets, sky_placed.models.targets):
        assert _centroid(after.high.spatial) == pytest.approx(
            _centroid(before.high.spatial), abs=0.002
        )


def test_mixed_skycoord_frames_resolve_in_icrs(instrument_data) -> None:
    pointing = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    ifu = pointing.spherical_offsets_by(-20 * u.mas, 10 * u.mas)
    target = pointing.spherical_offsets_by(-10 * u.mas, 15 * u.mas)
    results = []
    for pointing_input, ifu_input, target_input in (
        (pointing, ifu, target),
        (pointing, ifu.transform_to("fk5"), target.transform_to("fk5")),
        (pointing.transform_to("fk5"), ifu, target),
    ):
        etc = _etc(instrument_data)
        etc.set_pointing(sky_position=pointing_input, position_angle=30 * u.deg)
        etc.set_ifu_position(sky_position=ifu_input, rotation=20 * u.deg)
        _add_point(etc, sky_position=target_input)
        results.append(etc.run(include_models=True))

    reference = results[0].models.targets[0].high.spatial
    for result in results[1:]:
        np.testing.assert_allclose(result.models.targets[0].high.spatial, reference, atol=1e-9)
        assert result.options.ifu_center.frame.name == "icrs"
    assert results[1].options.ifu_position.sky_position.frame.name == "fk5"
    assert results[1].options.targets[0].sky_position.frame.name == "fk5"
    assert results[2].options.pointing_center.frame.name == "fk5"


def test_sky_fixed_target_moves_when_telescope_pointing_moves(instrument_data) -> None:
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    fixed_target = center.spherical_offsets_by(-10 * u.mas, 10 * u.mas)
    etc = _etc(instrument_data)
    etc.set_pointing(sky_position=center)
    _add_point(etc, ifu_offset=(0 * u.mas, 0 * u.mas))
    _add_point(etc, pointing_offset=(0 * u.mas, 0 * u.mas))
    _add_point(etc, sky_position=fixed_target)
    first = etc.run(include_models=True)

    shifted_pointing = center.spherical_offsets_by(20 * u.mas, 0 * u.mas)
    etc.set_pointing(sky_position=shifted_pointing)
    second = etc.run(include_models=True)
    first_positions = [_centroid(model.high.spatial) for model in first.models.targets]
    second_positions = [_centroid(model.high.spatial) for model in second.models.targets]
    assert second_positions[0] == pytest.approx(first_positions[0])
    assert second_positions[1] == pytest.approx(first_positions[1])
    assert first_positions[2][0] == pytest.approx(first_positions[0][0] + 1, abs=1e-5)
    assert first_positions[2][1] == pytest.approx(first_positions[0][1] + 1, abs=1e-5)
    assert second_positions[2][0] == pytest.approx(first_positions[2][0] + 2, abs=1e-5)
    assert second_positions[2][1] == pytest.approx(first_positions[2][1], abs=1e-5)
    assert first.options.pointing_center.separation(center) < 1e-6 * u.mas
    assert second.options.pointing_center.separation(shifted_pointing) < 1e-6 * u.mas


def test_off_axis_fits_wcs_uses_ifu_center_and_combined_angle(
    instrument_data, tmp_path
) -> None:
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    etc = _etc(instrument_data)
    etc.set_pointing(sky_position=center, position_angle=30 * u.deg)
    etc.set_ifu_position(pointing_offset=(10 * u.mas, 5 * u.mas), rotation=20 * u.deg)
    _add_point(etc, ifu_offset=(0 * u.mas, 0 * u.mas))
    result = etc.run()
    path = tmp_path / "result.fits"
    result.save(path)
    with fits.open(path) as hdus:
        header = hdus["SNR"].header
        assert hdus[0].header["POSANGLE"] == pytest.approx(50)
        assert header["CRVAL2"] == pytest.approx(result.options.ifu_center.icrs.ra.deg)
        assert header["CRVAL3"] == pytest.approx(result.options.ifu_center.icrs.dec.deg)
        scale = (50 * u.mas).to_value(u.deg)
        assert header["CD2_2"] == pytest.approx(-scale * np.cos(np.deg2rad(50)))
        assert header["CD3_2"] == pytest.approx(scale * np.sin(np.deg2rad(50)))
        reference_pixel = [[0, (result.snr.shape[1] - 1) / 2, (result.snr.shape[0] - 1) / 2]]
        world = WCS(header).all_pix2world(reference_pixel, 0)[0]
        assert world[1] == pytest.approx(result.options.ifu_center.icrs.ra.deg)
        assert world[2] == pytest.approx(result.options.ifu_center.icrs.dec.deg)


def test_direct_psf_stays_in_ifu_detector_axes(instrument_data) -> None:
    psf = np.zeros((5, 5))
    psf[2, 2] = 2
    psf[3, 1] = 1
    results = []
    for rotation in (0, 90):
        etc = _etc(instrument_data)
        etc.set_ifu_position(
            pointing_offset=(0 * u.mas, 0 * u.mas), rotation=rotation * u.deg
        )
        etc.set_psf(psf, pixel_scale=10 * u.mas)
        _add_point(etc, ifu_offset=(0 * u.mas, 0 * u.mas))
        results.append(etc.run())
    np.testing.assert_array_equal(results[0].psf.data, results[1].psf.data)
    assert results[0].psf.data[3, 1] > 0


def test_position_forms_are_exclusive_and_absolute_requires_pointing(instrument_data) -> None:
    etc = _etc(instrument_data)
    center = SkyCoord(120 * u.deg, 25 * u.deg, frame="icrs")
    with pytest.raises(ValueError, match="exactly one"):
        etc.set_ifu_position()
    with pytest.raises(ValueError, match="exactly one"):
        etc.set_ifu_position(pointing_offset=(0 * u.deg, 0 * u.deg), sky_position=center)
    with pytest.raises(ValueError, match="absolute telescope pointing"):
        etc.set_ifu_position(sky_position=center)
    with pytest.raises(ValueError, match="exactly one"):
        _add_point(etc)
    with pytest.raises(ValueError, match="exactly one"):
        _add_point(etc, ifu_offset=(0 * u.deg, 0 * u.deg), sky_position=center)
    with pytest.raises(ValueError, match="absolute telescope pointing"):
        _add_point(etc, sky_position=center)
