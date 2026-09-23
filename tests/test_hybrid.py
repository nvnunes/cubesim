"""Hybrid PSF integration and pointing-to-detector geometry."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from scipy import ndimage

import cubesim
from cubesim._hybrid import rotate_to_ifu
from cubesim._psf import load_psf


def _bundle(root: Path, *, zeropoint: str = "1966817941") -> Path:
    with (root / "etc.ini").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n[hybrid]\n"
            "mastsel_ini_file = mastsel.ini\n"
            "science_ho_interpolator_file = science.pkl\n"
            "ngs_ho_interpolator_file = ngs.pkl\n"
            f"ngs_magnitude_zeropoint = {zeropoint}\n"
        )
    for filename in ("mastsel.ini", "science.pkl", "ngs.pkl"):
        (root / filename).write_text("test asset", encoding="utf-8")
    return root


def _etc(root: Path) -> cubesim.Etc:
    etc = cubesim.Etc(root)
    etc.configure(scale="50mas", disperser="r3000.yj", atmosphere="airmass10_pwv10")
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.um,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_exposure(time=100 * u.s, n_target=2)
    return etc


def _set_hybrid(etc: cubesim.Etc, **kwargs) -> None:
    etc.set_hybrid_psf(
        ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
        ngs_magnitudes=np.array([12.0]) * u.mag,
        wavelength=1.1 * u.um,
        zenith_angle=20 * u.deg,
        **kwargs,
    )


def _fake_engine(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(requests=[], calls=0)
    image = np.zeros((9, 13), dtype=float)
    image[4, 6] = 10
    image[4, 9] = 2
    image[2, 6] = 1
    state.image = image
    state.pupil = np.ones((7, 7)) * u.one

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            state.requests.append(self)

    def simulate(request, science, ngs):
        state.calls += 1
        assert science.name == "science.pkl"
        assert ngs.name == "ngs.pkl"
        return SimpleNamespace(
            psfs=np.array([image]),
            metadata=SimpleNamespace(
                pixel_scale=10 * u.mas,
                wavelength=u.Quantity([request.wavelength.to_value(u.um)], u.um),
                tel_pupil=state.pupil,
                tel_diameter=7.9 * u.m,
            ),
            ngs_flux=np.array([123.0]) * u.photon / u.s,
        )

    monkeypatch.setitem(
        sys.modules,
        "hybrid_ao_psf",
        SimpleNamespace(
            HybridRequest=Request,
            load_science_ho_psf_interpolator=lambda path: path,
            load_ngs_ho_metric_interpolator=lambda path: path,
            simulate=simulate,
        ),
    )
    return state


@pytest.mark.parametrize("zeropoint", ["", "abc", "nan", "0", "-1"])
def test_hybrid_zeropoint_is_required_and_positive(instrument_data, zeropoint) -> None:
    _bundle(instrument_data, zeropoint=zeropoint)
    with pytest.raises(ValueError, match="ngs_magnitude_zeropoint"):
        cubesim.Etc(instrument_data)


def test_hybrid_rejects_complex_ngs_magnitudes(instrument_data) -> None:
    _bundle(instrument_data)
    etc = _etc(instrument_data)

    with pytest.raises(ValueError, match="ngs_magnitudes must be real"):
        etc.set_hybrid_psf(
            ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
            ngs_magnitudes=u.Quantity([12 + 1j], u.mag),
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )


def test_pending_hybrid_is_lazy_and_setter_order_wins(instrument_data, monkeypatch) -> None:
    _bundle(instrument_data)
    state = _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    _set_hybrid(etc)
    assert state.calls == 0
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    direct = etc.run()
    assert direct.options.hybrid is None
    assert state.calls == 0
    _set_hybrid(etc)
    first = etc.run()
    second = etc.run()
    assert state.calls == 2
    assert first.options.hybrid.coordinate_form == "pointing_offsets"
    assert first.options.hybrid.ngs_flux[0] == 123 * u.photon / u.s
    assert first.options.hybrid.ngs_magnitude_zeropoint == 1966817941 * u.photon / (u.m**2 * u.s)
    assert second.psf is not first.psf
    assert np.array_equal(direct.psf.data, np.ones((5, 5)) / 25)
    assert direct.psf.wavelength is None
    assert direct.psf.pupil is None
    assert direct.psf.telescope_diameter == 8 * u.m
    assert first.psf.wavelength == 1.1 * u.um
    assert first.psf.wavelength.isscalar
    assert first.psf.telescope_diameter == 7.9 * u.m
    np.testing.assert_array_equal(first.psf.pupil.value, np.ones((7, 7)))
    restored = pickle.loads(pickle.dumps(first))
    assert restored.psf.wavelength == first.psf.wavelength
    assert restored.psf.telescope_diameter == first.psf.telescope_diameter
    np.testing.assert_array_equal(restored.psf.pupil.value, first.psf.pupil.value)
    with pytest.raises(ValueError, match="read-only"):
        restored.psf.pupil[0, 0] = 0 * u.one


def test_hybrid_request_resolves_current_geometry_and_snapshots(instrument_data, monkeypatch) -> None:
    _bundle(instrument_data)
    state = _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    magnitudes = np.array([12.0]) * u.mag
    etc.set_hybrid_psf(
        ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
        ngs_magnitudes=magnitudes,
        wavelength=1.1 * u.um,
        zenith_angle=20 * u.deg,
    )
    magnitudes[0] = 20 * u.mag
    etc.set_ifu_position(pointing_offset=(2 * u.arcsec, 3 * u.arcsec), rotation=90 * u.deg)
    result = etc.run()
    request = state.requests[0]
    assert getattr(request, "ngs_flux", None) is None
    assert request.ngs_magnitude[0] == 12 * u.mag
    assert request.ngs_magnitude_zeropoint == 1966817941 * u.photon / (u.m**2 * u.s)
    assert request.science_x[0] == 2 * u.arcsec
    assert request.science_y[0] == 3 * u.arcsec
    assert request.ngs_x[0] == 10 * u.arcsec
    assert request.ngs_y[0] == 0 * u.arcsec
    assert result.options.hybrid.science_position[0].to_value(u.arcsec) == pytest.approx(2)
    assert result.options.hybrid.science_position[1].to_value(u.arcsec) == pytest.approx(3)
    assert np.allclose(result.psf.data, load_psf(
        rotate_to_ifu(state.image, 90 * u.deg),
        pixel_scale=10 * u.mas,
        instrument_root=instrument_data,
    ).data)
    state.pupil[0, 0] = 0 * u.one
    assert result.psf.pupil[0, 0] == 1 * u.one
    with pytest.raises(ValueError, match="read-only"):
        result.psf.pupil[0, 0] = 0 * u.one
    with pytest.raises(ValueError, match="read-only"):
        result.options.hybrid.ngs_magnitudes[0] = 9 * u.mag


def test_hybrid_polar_ngs_offsets_resolve_in_pointing_axes(instrument_data, monkeypatch) -> None:
    _bundle(instrument_data)
    state = _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    etc.set_pointing(position_angle=40 * u.deg)
    etc.set_hybrid_psf(
        ngs_pointing_offsets=(
            (30 * u.arcsec, 0 * u.deg),
            (30 * u.arcsec, 120 * u.deg),
            (30 * u.arcsec, 240 * u.deg),
        ),
        ngs_magnitudes=np.array([17, 17, 17]) * u.mag,
        wavelength=1.1 * u.um,
        zenith_angle=20 * u.deg,
    )
    result = etc.run()
    request = state.requests[0]
    np.testing.assert_allclose(request.ngs_x.to_value(u.arcsec), [30, -15, -15], atol=1e-12)
    np.testing.assert_allclose(
        request.ngs_y.to_value(u.arcsec),
        [0, 15 * np.sqrt(3), -15 * np.sqrt(3)],
        atol=1e-12,
    )
    assert result.options.hybrid.coordinate_form == "pointing_offsets"
    assert result.options.hybrid.ngs_pointing_offsets[1][0].to_value(u.arcsec) == pytest.approx(-15)


def test_sky_coordinates_and_ifu_rotation_share_pointing_frame(
    instrument_data, monkeypatch
) -> None:
    _bundle(instrument_data)
    state = _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    center = SkyCoord(ra=120 * u.deg, dec=25 * u.deg)
    etc.set_pointing(sky_position=center, position_angle=30 * u.deg)
    etc.set_ifu_position(
        pointing_offset=(2 * u.arcsec, 3 * u.arcsec), rotation=37 * u.deg
    )
    star = center.spherical_offsets_by(-10 * u.arcsec, 0 * u.arcsec)
    etc.set_hybrid_psf(
        ngs_sky_positions=SkyCoord([star]).galactic,
        ngs_magnitudes=np.array([12.0]) * u.mag,
        wavelength=1.1 * u.um,
        zenith_angle=20 * u.deg,
    )
    result = etc.run()
    assert result.options.hybrid.coordinate_form == "sky_positions"
    assert state.requests[-1].ngs_x[0].to_value(u.arcsec) == pytest.approx(10 * np.cos(np.deg2rad(30)))
    assert state.requests[-1].ngs_y[0].to_value(u.arcsec) == pytest.approx(-10 * np.sin(np.deg2rad(30)))
    assert state.requests[-1].science_x[0] == 2 * u.arcsec
    assert state.requests[-1].science_y[0] == 3 * u.arcsec
    np.testing.assert_allclose(
        result.psf.data,
        load_psf(
            rotate_to_ifu(state.image, 37 * u.deg),
            pixel_scale=10 * u.mas,
            instrument_root=instrument_data,
        ).data,
    )


def test_invalid_target_is_rejected_before_hybrid_modelling(
    instrument_data, monkeypatch
) -> None:
    _bundle(instrument_data)
    state = _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    center = SkyCoord(ra=120 * u.deg, dec=25 * u.deg)
    etc.set_pointing(sky_position=center)
    etc.add_target(
        sky_position=center,
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.um,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    _set_hybrid(etc)
    etc.set_pointing(position_angle=10 * u.deg)

    with pytest.raises(ValueError, match="Target sky_position requires"):
        etc.run()
    assert state.calls == 0


@pytest.mark.parametrize("angle", [0, 90, 37])
def test_rotation_preserves_flux_and_direction(angle) -> None:
    image = np.zeros((7, 11))
    image[3, 5] = 10
    image[3, 8] = 2
    image[1, 5] = 1
    rotated = rotate_to_ifu(image, angle * u.deg)
    assert rotated.sum() == pytest.approx(image.sum())
    center = np.unravel_index(np.argmax(rotated), rotated.shape)
    assert np.linalg.norm(np.asarray(center) - np.asarray(rotated.shape) // 2) <= 1
    if angle == 0:
        assert np.array_equal(rotated, image)
    elif angle == 90:
        assert rotated[center[0] - 3, center[1]] > 0
        assert rotated[center[0], center[1] - 2] > 0
    elif angle == 37:
        impulse = np.zeros((7, 11))
        impulse[3, 8] = 1
        transformed = rotate_to_ifu(impulse, 37 * u.deg)
        y, x = ndimage.center_of_mass(transformed)
        mid_y, mid_x = (np.asarray(transformed.shape) - 1) / 2
        assert x - mid_x == pytest.approx(3 * np.cos(np.deg2rad(37)), abs=0.2)
        assert y - mid_y == pytest.approx(-3 * np.sin(np.deg2rad(37)), abs=0.2)


def test_hybrid_validation_and_missing_dependency(instrument_data, monkeypatch) -> None:
    _bundle(instrument_data)
    etc = _etc(instrument_data)
    with pytest.raises(ValueError, match="exactly one"):
        etc.set_hybrid_psf(
            ngs_magnitudes=np.array([12]) * u.mag,
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )
    with pytest.raises(ValueError, match="one value per NGS"):
        etc.set_hybrid_psf(
            ngs_pointing_offsets=((0 * u.arcsec, 0 * u.arcsec),),
            ngs_magnitudes=np.array([12, 13]) * u.mag,
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )
    _set_hybrid(etc)
    etc.configure(scale="50mas", disperser="r3000.yj", atmosphere="airmass10_pwv10")
    etc.set_hybrid_psf(
        ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
        ngs_magnitudes=np.array([12]) * u.mag,
        wavelength=1.4 * u.um,
        zenith_angle=20 * u.deg,
    )
    with pytest.raises(ValueError, match="selected disperser range"):
        etc.run()
    _set_hybrid(etc)
    monkeypatch.setitem(sys.modules, "hybrid_ao_psf", None)
    with pytest.raises(ImportError, match=r"cubesim\[hybrid\]"):
        etc.run()


def test_hybrid_missing_asset_and_bad_ngs_inputs(instrument_data, monkeypatch) -> None:
    _bundle(instrument_data)
    _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    with pytest.raises(TypeError, match="1D magnitude"):
        etc.set_hybrid_psf(
            ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
            ngs_magnitudes=12 * u.mag,
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )
    with pytest.raises(u.UnitConversionError, match="mag units"):
        etc.set_hybrid_psf(
            ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
            ngs_magnitudes=np.array([12]) * u.s,
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )
    with pytest.raises(ValueError, match="finite"):
        etc.set_hybrid_psf(
            ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
            ngs_magnitudes=np.array([np.nan]) * u.mag,
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )
    with pytest.raises(ValueError, match="absolute telescope pointing"):
        etc.set_hybrid_psf(
            ngs_sky_positions=SkyCoord(["12h00m00s +30d00m00s"]),
            ngs_magnitudes=np.array([12]) * u.mag,
            wavelength=1.1 * u.um,
            zenith_angle=20 * u.deg,
        )
    _set_hybrid(etc)
    (instrument_data / "science.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="science.pkl"):
        etc.run()


def test_hybrid_wavelength_endpoints_and_direct_psf_equivalence(
    instrument_data, monkeypatch
) -> None:
    _bundle(instrument_data)
    state = _fake_engine(monkeypatch)
    etc = _etc(instrument_data)
    etc.set_ifu_position(pointing_offset=(0 * u.arcsec, 0 * u.arcsec), rotation=37 * u.deg)
    for wavelength in (0.95 * u.um, 1.35 * u.um):
        etc.set_hybrid_psf(
            ngs_pointing_offsets=((10 * u.arcsec, 0 * u.arcsec),),
            ngs_magnitudes=np.array([12]) * u.mag,
            wavelength=wavelength,
            zenith_angle=35 * u.deg,
        )
        hybrid_result = etc.run()
        assert hybrid_result.options.hybrid.zenith_angle == 35 * u.deg
    transformed = rotate_to_ifu(state.image, 37 * u.deg)
    etc.set_psf(transformed, pixel_scale=10 * u.mas)
    direct_result = etc.run()
    np.testing.assert_allclose(direct_result.psf.data, hybrid_result.psf.data)
    np.testing.assert_allclose(direct_result.snr, hybrid_result.snr)
    assert direct_result.options.hybrid is None
    assert hybrid_result.options.hybrid is not None


def test_real_hybrid_models_one_science_psf_and_resolves_photon_rates(instrument_data) -> None:
    hybrid = pytest.importorskip("hybrid_ao_psf")
    from ngs_photometry import WFSPhotometryConfig, magnitudes_to_effective_photons_per_second

    _bundle(instrument_data)
    (instrument_data / "mastsel.ini").write_text(
        (Path(__file__).parent / "data" / "hybrid_mastsel.ini").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    axis = np.arange(17) - 8
    x, y = np.meshgrid(axis, axis)
    psf = np.exp(-0.5 * (x**2 / 1.5**2 + y**2 / 1.8**2))
    psf = (psf / psf.sum()).astype(np.float32)
    hybrid.save_science_ho_psf_interpolator(
        hybrid.build_science_ho_psf_interpolator(
            hybrid.ScienceHoPsfSamples(
                zenith_angle=20 * u.deg,
                wavelength=1.1 * u.um,
                x=np.array([0, 36, 60]) * u.arcsec,
                y=np.zeros(3) * u.arcsec,
                psfs=np.stack([psf, psf, psf]),
                pixel_scale=10 * u.mas,
                tel_diameter=7.9 * u.m,
                tel_pupil=np.ones((17, 17), dtype=np.float32) * u.one,
            )
        ),
        instrument_data / "science.pkl",
        overwrite=True,
    )
    theta = np.deg2rad([0, 120, 240])
    hybrid.save_ngs_ho_metric_interpolator(
        hybrid.build_ngs_ho_metric_interpolator(
            hybrid.NgsHoMetricSamples(
                zenith_angle=20 * u.deg,
                x=30 * np.cos(theta) * u.arcsec,
                y=30 * np.sin(theta) * u.arcsec,
                ee=np.array([0.42, 0.40, 0.38]) * u.one,
                fwhm=np.array([76, 80, 84]) * u.mas,
                sr=np.array([0.24, 0.22, 0.20]) * u.one,
            ),
            interpolation_config=hybrid.RbfInterpolationConfig(smoothing=0),
        ),
        instrument_data / "ngs.pkl",
        overwrite=True,
    )
    etc = _etc(instrument_data)
    magnitudes = np.array([12.0, 12.0, 12.0]) * u.mag
    etc.set_hybrid_psf(
        ngs_pointing_offsets=tuple(
            (30 * np.cos(angle) * u.arcsec, 30 * np.sin(angle) * u.arcsec)
            for angle in theta
        ),
        ngs_magnitudes=magnitudes,
        wavelength=1.1 * u.um,
        zenith_angle=20 * u.deg,
    )
    result = etc.run()
    expected_flux = magnitudes_to_effective_photons_per_second(
        magnitudes,
        WFSPhotometryConfig(
            telescope_diameter=7.9 * u.m,
            n_channels=1,
            frame_rate=500 * u.Hz,
            zeropoint=1966817941 * u.photon / (u.m**2 * u.s),
        ),
    )
    assert result.psf.data.ndim == 2
    assert result.psf.data.sum() == pytest.approx(1)
    assert result.psf.pixel_scale == 10 * u.mas
    assert result.psf.wavelength == 1.1 * u.um
    assert result.psf.telescope_diameter == 7.9 * u.m
    np.testing.assert_array_equal(result.psf.pupil.value, np.ones((17, 17)))
    np.testing.assert_allclose(
        result.options.hybrid.ngs_flux.to_value(u.photon / u.s),
        expected_flux.to_value(u.photon / u.s),
    )
