"""Public forward-calculation API tests."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import cubesim


def test_package_exposes_etc_entrypoint(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    assert etc is not None
    assert not hasattr(cubesim, "EtcOptions")
    assert not hasattr(cubesim, "compute")


def test_run_requires_resolved_calculation_state(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match=r"configure\(\)"):
        etc.run()

    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    with pytest.raises(ValueError, match="at least one target"):
        etc.run()

    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=_line(),
    )
    with pytest.raises(ValueError, match=r"set_exposure\(\)"):
        etc.run()

    etc.set_exposure(time=100 * u.s, n_target=2)
    with pytest.raises(ValueError, match=r"set_psf\(\)"):
        etc.run()


def test_run_returns_requested_groups_and_immutable_snapshot(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(
        name="line",
        size=(1, 1, 3),
        center=(1, 2, 1.1 * u.micron),
    )

    result = etc.run(include_models=True)
    sample = result.sample(n=2, seed=42)
    single_sample = result.sample(seed=42)
    aperture_sample = result.apertures[0].sample(
        n=2,
        seed=42,
    )
    single_aperture_sample = result.apertures[0].sample(
        seed=42,
    )

    assert result.snr.shape[:2] == (3, 4)
    assert result.wavelength.shape == result.snr.shape[2:]
    assert sample.data.shape == (2, *result.snr.shape)
    assert sample.data.unit == u.electron
    assert sample.seed == 42
    assert single_sample.data.shape == result.snr.shape
    assert result.signals.target.unit == u.electron
    assert result.variances.total.unit == u.electron**2
    assert result.models.combined.shape == result.snr.shape
    assert result.models.transmission.shape == result.wavelength.shape
    assert result.models.sky.shape == result.wavelength.shape
    assert result.models.thermal.shape == result.wavelength.shape
    radiance = u.W / (u.m**2 * u.sr * u.m)
    assert result.models.sky.unit == radiance
    assert result.models.thermal.unit == radiance
    assert len(result.models.targets) == 1
    assert result.options.exposure.n == 4
    assert result.options.exposure.n_target == 2
    assert result.options.exposure.n_sky == 2
    assert aperture_sample.data.shape == (2,)
    assert aperture_sample.seed == 42
    assert aperture_sample.name == "line"
    assert np.array_equal(aperture_sample.mask, result.apertures[0].mask)
    assert single_aperture_sample.data.isscalar
    assert result.apertures[0].mask.sum() == 3
    assert result.psf.data.shape == (5, 5)
    assert result.psf.data.sum() == pytest.approx(1.0)
    assert result.psf.pixel_scale == 10 * u.mas
    assert result.psf.wavelength is None
    assert result.psf.pupil is None
    assert result.psf.telescope_diameter == 8 * u.m

    aperture = result.apertures[0]
    assert aperture.spectra.snr.shape == result.wavelength.shape
    assert aperture.maps.snr.shape == result.snr.shape[:2]
    for field in aperture.signals.__dataclass_fields__:
        integrated = getattr(aperture.signals, field)
        expected = getattr(result.signals, field)[aperture.mask].sum()
        assert integrated.to_value(expected.unit) == pytest.approx(expected.value)
        assert np.allclose(
            getattr(aperture.spectra.signals, field).sum().to_value(integrated.unit),
            integrated.value,
        )
        assert np.allclose(
            getattr(aperture.maps.signals, field).sum().to_value(integrated.unit),
            integrated.value,
        )
    for field in aperture.variances.__dataclass_fields__:
        integrated = getattr(aperture.variances, field)
        assert np.allclose(
            getattr(aperture.spectra.variances, field).sum().to_value(
                integrated.unit
            ),
            integrated.value,
        )
        assert np.allclose(
            getattr(aperture.maps.variances, field).sum().to_value(integrated.unit),
            integrated.value,
        )

    with pytest.raises(ValueError, match="read-only"):
        result.snr[0, 0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.options.exposure.time[...] = 1 * u.s
    with pytest.raises(ValueError, match="read-only"):
        result.options.targets[0].spectrum.wavelength[0] = 1.2 * u.micron
    with pytest.raises(ValueError, match="read-only"):
        result.psf.data[0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.psf.pixel_scale[...] = 20 * u.mas
    with pytest.raises(AttributeError):
        result.psf.data = np.ones((5, 5))
    with pytest.raises(ValueError, match="read-only"):
        result.apertures[0].spectra.snr[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.apertures[0].maps.signals.target[0, 0] = 0 * u.electron
    with pytest.raises(ValueError, match="read-only"):
        result.models.transmission[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.models.sky[0] = 0 * result.models.sky.unit
    with pytest.raises(ValueError, match="read-only"):
        result.models.thermal[0] = 0 * result.models.thermal.unit


def test_result_snapshot_freezes_nested_model_and_pointing_data(
    instrument_data,
) -> None:
    etc = _base_etc(instrument_data)
    etc.set_pointing(
        sky_position=SkyCoord(ra=120 * u.deg, dec=25 * u.deg, frame="icrs")
    )
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.SpatialImage(np.ones((3, 3)), pixel_scale=10 * u.mas),
        spectrum=_line(),
    )
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)

    result = etc.run()

    with pytest.raises(ValueError, match="read-only"):
        result.options.targets[0].spatial.pixel_scale[...] = 20 * u.mas
    with pytest.raises(ValueError, match="read-only"):
        result.options.pointing_center.data.lon[...] = 121 * u.deg


@pytest.mark.parametrize("method", ["nodding", "in_field"])
def test_aperture_samples_match_predicted_signal_and_variance(
    instrument_data,
    method: str,
) -> None:
    etc = _configured_etc(instrument_data)
    if method == "in_field":
        sky_mask = np.zeros((3, 4), dtype=bool)
        sky_mask[0] = True
        etc.set_sky_subtraction(method="in_field", mask=sky_mask)
    etc.add_aperture(
        name="source",
        size=(1, 2, 3),
        center=(1, 1.5, 1.1 * u.micron),
    )
    result = etc.run()
    aperture = result.apertures[0]

    samples = aperture.sample(n=20_000, seed=91)
    noise = samples.data - aperture.signals.target
    predicted = aperture.variances.total.to_value(u.electron**2)

    assert abs(noise.mean().to_value(u.electron)) < 4 * np.sqrt(
        predicted / len(samples.data)
    )
    assert noise.var(ddof=1).to_value(u.electron**2) == pytest.approx(
        predicted,
        rel=0.04,
    )


def test_default_result_includes_core_groups_but_not_models(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(name="voxel", size=(1, 1, 1), start=(0, 0, 0))
    result = etc.run()

    for name in ("models", "data"):
        assert not hasattr(result, name)
    aperture = result.apertures[0]
    assert result.signals.total.shape == result.snr.shape
    assert result.variances.total.shape == result.snr.shape
    assert aperture.signals.total.isscalar
    assert aperture.variances.total.isscalar
    assert aperture.spectra.signals.total.shape == result.wavelength.shape
    assert aperture.spectra.variances.total.shape == result.wavelength.shape
    assert aperture.maps.signals.total.shape == result.snr.shape[:2]
    assert aperture.maps.variances.total.shape == result.snr.shape[:2]


def test_uniform_target_result_has_no_psf_snapshot(instrument_data) -> None:
    etc = _base_etc(instrument_data)
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

    result = etc.run()

    assert not hasattr(result, "psf")


def test_sampling_validates_request(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(name="voxel", size=(1, 1, 1), start=(0, 0, 0))
    result = etc.run()

    cube_sample = result.sample()
    aperture_sample = result.apertures[0].sample()
    assert cube_sample.data.shape == result.snr.shape
    assert isinstance(cube_sample.seed, int)
    assert aperture_sample.data.isscalar
    assert isinstance(aperture_sample.seed, int)
    assert np.array_equal(
        cube_sample.data,
        result.sample(seed=cube_sample.seed).data,
    )
    assert np.array_equal(
        aperture_sample.data,
        result.apertures[0].sample(seed=aperture_sample.seed).data,
    )
    with pytest.raises(ValueError, match="positive integer"):
        result.sample(n=0)
    with pytest.raises(ValueError, match="positive integer"):
        result.apertures[0].sample(n=True)
    with pytest.raises(TypeError, match="non-negative integer"):
        result.sample(seed=object())
    with pytest.raises(ValueError, match="between 0"):
        result.sample(seed=-1)


def test_signal_identities_and_seeded_samples_are_reproducible(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    first = etc.run()
    second = etc.run()

    first_sample = first.sample(n=2, seed=17)
    second_sample = second.sample(n=2, seed=17)
    assert np.array_equal(first_sample.data.value, second_sample.data.value)
    assert not hasattr(first.signals, "background")
    assert np.allclose(
        first.signals.total.value,
        (
            first.signals.target
            + first.signals.sky
            + first.signals.thermal
            + first.signals.dark
        ).value,
    )
    assert np.allclose(
        first.variances.total.value,
        (
            first.variances.target
            + first.variances.sky
            + first.variances.thermal
            + first.variances.dark
            + first.variances.read
        ).value,
    )


def test_abba_resolves_total_and_target_exposure_counts(instrument_data) -> None:
    total_count = _configured_etc(instrument_data)
    total_count.set_sky_subtraction(method="nodding", sequence="ABBA")
    total_count.set_exposure(time=30 * u.s, n=20)
    total_result = total_count.run()

    target_count = _configured_etc(instrument_data)
    target_count.set_sky_subtraction(method="nodding", sequence="ABBA")
    target_count.set_exposure(time=30 * u.s, n_target=10)
    target_result = target_count.run()

    equal_weight = _configured_etc(instrument_data)
    equal_weight.set_exposure(time=30 * u.s, n_target=10)
    equal_weight_result = equal_weight.run()

    assert total_result.options.exposure == target_result.options.exposure
    assert total_result.options.exposure.n == 20
    assert total_result.options.exposure.n_target == 10
    assert total_result.options.exposure.n_sky == 10
    assert total_result.options.exposure.target_time == 300 * u.s
    assert total_result.options.exposure.sky_time == 300 * u.s
    assert total_result.options.exposure.total_time == 600 * u.s
    assert np.allclose(
        total_result.variances.total.value,
        target_result.variances.total.value,
    )
    assert np.allclose(
        total_result.variances.total.value,
        equal_weight_result.variances.total.value,
    )


@pytest.mark.parametrize(("count", "value"), [("n", 19), ("n_target", 9)])
def test_partial_nodding_sequence_is_rejected(
    instrument_data,
    count: str,
    value: int,
) -> None:
    etc = _configured_etc(instrument_data)
    etc.set_sky_subtraction(method="nodding", sequence="ABBA")
    etc.set_exposure(time=30 * u.s, **{count: value})

    with pytest.raises(ValueError, match="whole number"):
        etc.run()


def test_in_field_subtraction_propagates_aperture_covariance(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    sky_mask = np.zeros((3, 4), dtype=bool)
    sky_mask[0] = True
    etc.set_sky_subtraction(method="in_field", mask=sky_mask)
    etc.add_aperture(
        name="source",
        size=(1, 2, 1),
        center=(1, 1.5, 1.1 * u.micron),
    )

    result = etc.run()
    aperture = result.apertures[0]

    assert result.options.exposure.n == 2
    assert result.options.exposure.n_target == 2
    assert result.options.exposure.n_sky == 0
    assert aperture.variances.total > result.variances.total[aperture.mask].sum()
    assert np.allclose(
        aperture.spectra.variances.total.sum().to_value(
            aperture.variances.total.unit
        ),
        aperture.variances.total.value,
    )
    marginal_variance = result.variances.total[aperture.mask].sum()
    assert np.allclose(
        aperture.maps.variances.total.sum().to_value(marginal_variance.unit),
        marginal_variance.value,
    )
    assert aperture.maps.variances.total.sum() < aperture.variances.total
    for field in aperture.signals.__dataclass_fields__:
        integrated = getattr(aperture.signals, field)
        expected = getattr(result.signals, field)[aperture.mask].sum()
        assert integrated.to_value(expected.unit) == pytest.approx(expected.value)
        assert np.allclose(
            getattr(aperture.spectra.signals, field).sum().to_value(integrated.unit),
            integrated.value,
        )
        assert np.allclose(
            getattr(aperture.maps.signals, field).sum().to_value(integrated.unit),
            integrated.value,
        )


def test_in_field_subtraction_treats_sky_mask_as_target_free(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    sky_mask = np.zeros((3, 4), dtype=bool)
    sky_mask[0] = True
    etc.set_sky_subtraction(
        method="in_field",
        mask=sky_mask,
    )

    result = etc.run()
    data = result.sample(seed=4).data

    assert np.all(result.signals.target[sky_mask] == 0 * u.electron)
    assert np.all(result.snr[sky_mask] == 0)
    assert np.allclose(data[sky_mask].sum(axis=0).value, 0.0, atol=1e-12)


def test_in_field_subtraction_rejects_aperture_sky_overlap(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    sky_mask = np.zeros((3, 4), dtype=bool)
    sky_mask[0] = True
    etc.set_sky_subtraction(method="in_field", mask=sky_mask)
    etc.add_aperture(
        name="overlap",
        size=(1, 1, 1),
        start=(0, 1, 1.1 * u.micron),
    )

    with pytest.raises(
        ValueError,
        match="Aperture 'overlap' overlaps the in-field sky mask",
    ):
        etc.run()


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 4), dtype=bool),
        np.ones((3, 4), dtype=int),
        np.ones(4, dtype=bool),
    ],
)
def test_in_field_sky_mask_rejects_invalid_arrays(instrument_data, mask) -> None:
    etc = _configured_etc(instrument_data)

    with pytest.raises(ValueError, match="nonempty 2D Boolean"):
        etc.set_sky_subtraction(method="in_field", mask=mask)


def test_nodding_sequence_rejects_non_string_input(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    with pytest.raises(TypeError, match="sequence must be a string"):
        etc.set_sky_subtraction(method="nodding", sequence=3)


def test_aperture_rejects_malformed_coordinate_tuples(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    with pytest.raises(TypeError, match="size must be a .* tuple"):
        etc.add_aperture(name="scalar-size", size=3)
    with pytest.raises(ValueError, match="size must contain"):
        etc.add_aperture(name="short-size", size=(1, 1))
    with pytest.raises(TypeError, match="start must be a three-coordinate tuple"):
        etc.add_aperture(name="scalar-start", size=(1, 1, 1), start=3)


def test_aperture_rejects_empty_custom_mask(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    with pytest.raises(ValueError, match="nonempty three-dimensional Boolean"):
        etc.add_aperture(name="empty", mask=np.zeros((3, 4, 5), dtype=bool))


def test_run_rejects_non_boolean_model_selection(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    with pytest.raises(TypeError, match="include_models must be a Boolean"):
        etc.run(include_models=1)


def test_retained_state_is_revalidated_after_scale_change(instrument_data) -> None:
    ini_path = Path(instrument_data) / "etc.ini"
    ini_path.write_text(
        ini_path.read_text(encoding="utf-8")
        + """

[scale.100mas]
spaxels_x = 2
spaxels_y = 2
spaxel_scale = 100
""",
        encoding="utf-8",
    )
    etc = _configured_etc(instrument_data)
    original = etc.run()
    etc.set_sky_subtraction(
        method="in_field",
        mask=np.ones((3, 4), dtype=bool),
    )
    etc.configure(
        scale="100mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )

    with pytest.raises(ValueError, match=r"mask shape must be \(2, 2\)"):
        etc.run()
    assert original.options.scale == "50mas"
    assert original.options.sky_subtraction.method == "nodding"


def test_repeated_runs_preserve_results_and_combine_targets(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    first = etc.run(include_models=True)
    first_combined = first.models.combined.copy()

    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=_line(),
    )
    etc.set_exposure(time=200 * u.s, n_target=4)
    second = etc.run(include_models=True)

    assert len(first.options.targets) == 1
    assert first.options.exposure.n_target == 2
    assert np.array_equal(first.models.combined.value, first_combined.value)
    assert len(second.options.targets) == 2
    assert second.options.exposure.n_target == 4
    assert np.allclose(second.models.combined.value, 2 * first_combined.value)


def test_caller_owned_arrays_are_copied(instrument_data) -> None:
    psf = np.zeros((5, 5))
    psf[2, 2] = 1
    sky_mask = np.zeros((3, 4), dtype=bool)
    sky_mask[0] = True
    etc = _base_etc(instrument_data)
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=_line(),
    )
    etc.set_psf(psf, pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    etc.set_sky_subtraction(method="in_field", mask=sky_mask)

    psf[:] = np.nan
    sky_mask[:] = False
    result = etc.run()

    assert result.options.sky_subtraction.mask.sum() == 4
    assert np.isfinite(result.snr).all()


def test_aperture_placement_uses_one_rounding_rule_on_all_axes(
    instrument_data,
) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(
        name="half",
        size=(2, 2, 2),
        center=(1.5, 1.5, 5.5),
    )
    etc.add_aperture(
        name="integer",
        size=(1, 1, 1),
        center=(1, 2, 4),
    )
    etc.add_aperture(
        name="start",
        size=(1, 2, 3),
        start=(0, 1, 2),
    )

    half, integer, started = etc.run().apertures

    assert np.array_equal(np.argwhere(half.mask).min(axis=0), [1, 1, 5])
    assert np.array_equal(np.argwhere(half.mask).max(axis=0), [2, 2, 6])
    assert np.array_equal(np.argwhere(integer.mask), [[1, 2, 4]])
    assert np.array_equal(np.argwhere(started.mask).min(axis=0), [0, 1, 2])
    assert np.array_equal(np.argwhere(started.mask).max(axis=0), [0, 2, 4])


@pytest.mark.parametrize(
    ("placement", "axis"),
    [
        ({"start": (-1, 0, 0)}, 0),
        ({"start": (0, 4, 0)}, 1),
        ({"start": (0, 0, 1_000_000)}, 2),
        ({"center": (-1, 1, 4)}, 0),
        ({"center": (1, -1, 4)}, 1),
        ({"center": (1, 1, -1)}, 2),
    ],
)
def test_out_of_bounds_aperture_is_rejected(
    instrument_data,
    placement: dict[str, tuple[int, int, int]],
    axis: int,
) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(name="outside", size=(1, 1, 1), **placement)

    with pytest.raises(ValueError, match=f"extends beyond axis {axis}"):
        etc.run()


@pytest.mark.parametrize(
    ("placement", "wavelength"),
    [
        ("center", 0.1 * u.micron),
        ("center", 2.0 * u.micron),
        ("start", 0.1 * u.micron),
        ("start", 2.0 * u.micron),
    ],
)
def test_out_of_bounds_aperture_wavelength_is_rejected(
    instrument_data,
    placement: str,
    wavelength: u.Quantity,
) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(
        name="outside",
        size=(1, 1, 1),
        **{placement: (1, 1, wavelength)},
    )

    with pytest.raises(ValueError, match="outside the selected disperser"):
        etc.run()


def test_disperser_must_produce_multiple_detector_samples(instrument_data) -> None:
    ini_path = Path(instrument_data) / "etc.ini"
    ini_path.write_text(
        ini_path.read_text(encoding="utf-8").replace(
            "resolving_power = 3000",
            "resolving_power = 1",
        ),
        encoding="utf-8",
    )
    etc = _configured_etc(instrument_data)

    with pytest.raises(ValueError, match="fewer than two detector"):
        etc.run()


def test_custom_aperture_mask_is_revalidated_against_cube(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    etc.add_aperture(
        name="wrong-shape",
        mask=np.ones((3, 4, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="mask shape must be"):
        etc.run()


def _configured_etc(instrument_data) -> cubesim.Etc:
    etc = _base_etc(instrument_data)
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=_line(),
    )
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    return etc


def _base_etc(instrument_data) -> cubesim.Etc:
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    return etc


def _line() -> cubesim.GaussianLines:
    return cubesim.GaussianLines(
        wavelength=1.1 * u.micron,
        flux=1e-17 * u.erg / (u.s * u.cm**2),
        dispersion=40 * u.km / u.s,
    )
