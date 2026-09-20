"""Result persistence contract tests."""

from __future__ import annotations

import pickle

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

import cubesim


def test_pickle_round_trips_complete_result(instrument_data, tmp_path) -> None:
    result = _result(instrument_data)
    path = tmp_path / "result.pkl"

    result.save(path)
    with path.open("rb") as stream:
        restored = pickle.load(stream)

    assert np.array_equal(restored.snr, result.snr)
    assert np.array_equal(restored.data, result.data)
    assert restored.options.instrument == result.options.instrument
    assert restored.options.scale == result.options.scale
    assert restored.options.exposure == result.options.exposure
    assert restored.apertures[0].name == "line"
    assert np.array_equal(restored.models.transmission, result.models.transmission)
    assert np.array_equal(restored.models.sky, result.models.sky)
    assert np.array_equal(restored.models.thermal, result.models.thermal)
    with pytest.raises(ValueError, match="read-only"):
        restored.data[0, 0, 0, 0] = 0 * u.electron
    with pytest.raises(FileExistsError):
        result.save(path)
    result.save(path, overwrite=True)


def test_fits_stores_datacubes_metadata_masks_and_apertures(
    instrument_data,
    tmp_path,
) -> None:
    result = _result(instrument_data)
    path = tmp_path / "result.fits"

    result.save(path)

    with fits.open(path, checksum=True) as hdus:
        assert {hdu.name for hdu in hdus} == {
            "PRIMARY",
            "WAVELEN",
            "SNR",
            "MODEL",
            "TRANSMIS",
            "SKYMODEL",
            "THERMAL",
            "SIGTARG",
            "SIGSKY",
            "SIGTHERM",
            "SIGDARK",
            "SIGBKG",
            "SIGTOTAL",
            "VARTARG",
            "VARSKY",
            "VARTHERM",
            "VARDARK",
            "VARREAD",
            "VARTOTAL",
            "DATA",
            "APMASK0",
            "APERTURE",
        }
        assert hdus[0].header["NTARGET"] == 2
        assert hdus[0].header["NSKY"] == 2
        assert hdus[0].header["NCUBES"] == 2
        assert hdus["SNR"].header["BUNIT"] == "1"
        assert hdus["TRANSMIS"].header["BUNIT"] == "1"
        assert hdus["SKYMODEL"].header["BUNIT"] == "W sr-1 m-3"
        assert hdus["THERMAL"].header["BUNIT"] == "W sr-1 m-3"
        assert hdus["SIGTARG"].header["BUNIT"] == "electron"
        assert hdus["VARTOTAL"].header["BUNIT"] == "electron2"
        assert hdus["SNR"].header["CTYPE2"] == "XOFFSET"
        pixel_scale = (50 * u.mas).to_value(u.deg)
        angle = np.deg2rad(30)
        assert hdus["SNR"].header["CD2_2"] == pytest.approx(
            -pixel_scale * np.cos(angle)
        )
        assert hdus["SNR"].header["CD2_3"] == pytest.approx(pixel_scale * np.sin(angle))
        assert hdus["SNR"].header["CD3_2"] == pytest.approx(pixel_scale * np.sin(angle))
        assert hdus["SNR"].header["CD3_3"] == pytest.approx(pixel_scale * np.cos(angle))
        assert hdus["SNR"].data.shape == result.snr.shape
        assert hdus["TRANSMIS"].data.shape == result.wavelength.shape
        assert hdus["SKYMODEL"].data.shape == result.wavelength.shape
        assert hdus["THERMAL"].data.shape == result.wavelength.shape
        assert hdus["DATA"].data.shape == result.data.shape
        assert hdus["APMASK0"].data.sum() == 3
        assert hdus["APERTURE"].data["NAME"][0].rstrip() == "line"


def test_fits_uses_celestial_wcs_for_absolute_pointing(
    instrument_data,
    tmp_path,
) -> None:
    center = SkyCoord(ra=120 * u.deg, dec=25 * u.deg, frame="icrs")
    result = _result(instrument_data, center=center)
    path = tmp_path / "celestial.fits"

    result.save(path)

    with fits.open(path) as hdus:
        header = hdus["SNR"].header
        assert header["CTYPE2"] == "RA---TAN"
        assert header["CTYPE3"] == "DEC--TAN"
        assert header["CRVAL2"] == pytest.approx(120.0)
        assert header["CRVAL3"] == pytest.approx(25.0)


def test_save_rejects_unknown_extension(instrument_data, tmp_path) -> None:
    with pytest.raises(ValueError, match="PKL or FITS"):
        _result(instrument_data).save(tmp_path / "result.npy")


def _result(instrument_data, *, center=None):
    etc = cubesim.Etc(instrument_data)
    etc.configure(
        scale="50mas",
        disperser="r3000.yj",
        atmosphere="airmass10_pwv10",
    )
    etc.set_pointing(position_angle=30 * u.deg, center=center)
    etc.add_target(
        position=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.1 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_psf(np.ones((5, 5)), pixel_scale=10 * u.mas)
    etc.set_exposure(time=100 * u.s, n_target=2)
    etc.add_aperture(
        name="line",
        size=(1, 1, 3),
        center=(1, 2, 1.1 * u.micron),
    )
    return etc.run(
        include_models=True,
        include_signals=True,
        include_variances=True,
        include_data=True,
        n_cubes=2,
        rng=np.random.default_rng(9),
    )
