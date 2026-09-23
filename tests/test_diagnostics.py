"""PSF diagnostics over retained CubeSim results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import sys

import astropy.units as u
import numpy as np
import pytest

import cubesim
import cubesim.diagnostics as diagnostics


@pytest.fixture
def psf_result(instrument_data):
    etc = cubesim.Etc(instrument_data)
    etc.configure(scale="50mas", disperser="r3000.yj", atmosphere="airmass10_pwv10")
    y, x = np.indices((41, 41))
    image = np.exp(-0.5 * (((y - 20) / 2.5) ** 2 + ((x - 20) / 3.0) ** 2))
    etc.set_psf(image, pixel_scale=10 * u.mas)
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.um,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_exposure(time=100 * u.s, n_target=1)
    return etc.run().psf


def test_direct_psf_stats_use_pixel_scale_without_strehl(psf_result) -> None:
    pytest.importorskip("ao_stats")
    from ao_stats import compute_psf_stats

    apertures = np.array([20, 50, 100]) * u.mas

    measured = diagnostics.psf_stats(psf_result, ee_apertures=apertures)
    expected_ee, expected_fwhm = compute_psf_stats(
        psf_result.data,
        pixel_scale=psf_result.pixel_scale,
        ee_apertures=apertures,
        ee_geometry="ensquared",
        metrics=("ee", "fwhm"),
    )

    assert measured.sr is None
    assert measured.ee_apertures.unit == u.mas
    np.testing.assert_array_equal(measured.ee.value, expected_ee.value)
    assert measured.fwhm == expected_fwhm
    assert np.isfinite(measured.fwhm.value)
    with pytest.raises(ValueError, match="read-only"):
        measured.ee[0] = 0 * u.one
    with pytest.raises(FrozenInstanceError):
        measured.sr = 0 * u.one


def test_psf_stats_use_full_metadata_for_strehl(psf_result) -> None:
    pytest.importorskip("ao_stats")
    from ao_stats import PsfMetadata, compute_psf_stats

    pupil = np.ones((32, 32)) * u.one
    hybrid_psf = replace(psf_result, wavelength=1.1 * u.um, pupil=pupil)
    apertures = np.array([20, 50, 100]) * u.mas

    measured = diagnostics.psf_stats(hybrid_psf, ee_apertures=apertures)
    expected_sr, expected_ee, expected_fwhm = compute_psf_stats(
        hybrid_psf.data,
        PsfMetadata(
            wavelength=hybrid_psf.wavelength,
            pixel_scale=hybrid_psf.pixel_scale,
            tel_diameter=hybrid_psf.telescope_diameter,
            tel_pupil=hybrid_psf.pupil,
        ),
        ee_apertures=apertures,
        ee_geometry="ensquared",
    )

    assert measured.sr == expected_sr
    np.testing.assert_array_equal(measured.ee.value, expected_ee.value)
    assert measured.fwhm == expected_fwhm
    assert np.array_equal(psf_result.data, hybrid_psf.data)


def test_psf_stats_validate_input_and_apertures(psf_result) -> None:
    pytest.importorskip("ao_stats")
    with pytest.raises(TypeError, match="structured result.psf"):
        diagnostics.psf_stats(psf_result.data, ee_apertures=[20] * u.mas)
    with pytest.raises(TypeError, match="angular Quantity"):
        diagnostics.psf_stats(psf_result, ee_apertures=[20])
    with pytest.raises(ValueError, match="one-dimensional"):
        diagnostics.psf_stats(psf_result, ee_apertures=20 * u.mas)
    with pytest.raises(ValueError, match="finite values > 0"):
        diagnostics.psf_stats(psf_result, ee_apertures=[0] * u.mas)


def test_psf_stats_reports_missing_optional_dependency(psf_result, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ao_stats", None)

    with pytest.raises(ImportError, match=r"cubesim\[stats\]"):
        diagnostics.psf_stats(psf_result, ee_apertures=[100] * u.mas)
