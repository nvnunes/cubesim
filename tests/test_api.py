"""Public forward-calculation API tests."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

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
        position=(0 * u.arcsec, 0 * u.arcsec),
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

    result = etc.run(
        include_models=True,
        include_signals=True,
        include_variances=True,
        include_data=True,
        n_cubes=2,
        rng=np.random.default_rng(42),
    )

    assert result.snr.shape[:2] == (3, 4)
    assert result.wavelength.shape == result.snr.shape[2:]
    assert result.data.shape == (2, *result.snr.shape)
    assert result.data.unit == u.electron
    assert result.signals.target.unit == u.electron
    assert result.variances.total.unit == u.electron**2
    assert result.models.combined.shape == result.snr.shape
    assert len(result.models.targets) == 1
    assert result.options.exposure.n == 4
    assert result.options.exposure.n_target == 2
    assert result.options.exposure.n_sky == 2
    assert result.options.n_cubes == 2
    assert result.apertures[0].data.shape == (2,)
    assert result.apertures[0].mask.sum() == 3

    with pytest.raises(ValueError, match="read-only"):
        result.snr[0, 0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        result.options.exposure.time[...] = 1 * u.s
    with pytest.raises(ValueError, match="read-only"):
        result.options.targets[0].spectrum.wavelength[0] = 1.2 * u.micron
    with pytest.raises(ValueError, match="read-only"):
        result.apertures[0].data[0] = 0 * u.electron
    with pytest.raises(AttributeError, match="immutable"):
        result.data = result.data


def test_optional_groups_are_absent_by_default(instrument_data) -> None:
    result = _configured_etc(instrument_data).run()

    for name in ("models", "signals", "variances", "data"):
        assert not hasattr(result, name)


def test_realization_options_require_data(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    with pytest.raises(ValueError, match="include_data=True"):
        etc.run(n_cubes=2)


def test_signal_identities_and_seeded_data_are_reproducible(instrument_data) -> None:
    etc = _configured_etc(instrument_data)

    first = etc.run(
        include_signals=True,
        include_variances=True,
        include_data=True,
        n_cubes=2,
        rng=np.random.default_rng(17),
    )
    second = etc.run(
        include_signals=True,
        include_variances=True,
        include_data=True,
        n_cubes=2,
        rng=np.random.default_rng(17),
    )

    assert np.array_equal(first.data.value, second.data.value)
    assert np.allclose(
        first.signals.background.value,
        (first.signals.sky + first.signals.thermal + first.signals.dark).value,
    )
    assert np.allclose(
        first.signals.total.value,
        (first.signals.target + first.signals.background).value,
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
    total_result = total_count.run(include_variances=True)

    target_count = _configured_etc(instrument_data)
    target_count.set_sky_subtraction(method="nodding", sequence="ABBA")
    target_count.set_exposure(time=30 * u.s, n_target=10)
    target_result = target_count.run(include_variances=True)

    equal_weight = _configured_etc(instrument_data)
    equal_weight.set_exposure(time=30 * u.s, n_target=10)
    equal_weight_result = equal_weight.run(include_variances=True)

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

    result = etc.run(include_signals=True, include_variances=True)
    aperture = result.apertures[0]

    assert result.options.exposure.n == 2
    assert result.options.exposure.n_target == 2
    assert result.options.exposure.n_sky == 0
    assert aperture.variances.total > result.variances.total[aperture.mask].sum()


def test_in_field_subtraction_applies_target_estimate_to_snr(instrument_data) -> None:
    etc = _configured_etc(instrument_data)
    etc.set_sky_subtraction(
        method="in_field",
        mask=np.ones((3, 4), dtype=bool),
    )
    etc.add_aperture(
        name="field",
        size=(3, 4, 1),
        start=(0, 0, 1.1 * u.micron),
    )

    result = etc.run(
        include_signals=True,
        include_data=True,
        rng=np.random.default_rng(4),
    )

    assert result.snr.min() < 0
    assert result.snr.max() > 0
    assert result.apertures[0].snr == pytest.approx(0.0, abs=1e-7)
    assert result.apertures[0].signals.target.value == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.data.sum(axis=(0, 1)).value, 0.0, atol=1e-12)


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
        position=(0 * u.arcsec, 0 * u.arcsec),
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
        position=(0 * u.arcsec, 0 * u.arcsec),
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
        position=(0 * u.arcsec, 0 * u.arcsec),
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
