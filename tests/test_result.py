"""Result persistence contract tests."""

from __future__ import annotations

import pickle

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import cubesim


def test_pickle_round_trips_complete_result(instrument_data, tmp_path) -> None:
    result = _result(instrument_data)
    path = tmp_path / "result.pkl"

    result.save(path)
    with path.open("rb") as stream:
        restored = pickle.load(stream)

    assert np.array_equal(restored.snr, result.snr)
    assert np.array_equal(
        restored.sample(n=2, seed=9).data,
        result.sample(n=2, seed=9).data,
    )
    assert np.array_equal(
        restored.apertures[0].sample(n=2, seed=9).data,
        result.apertures[0].sample(n=2, seed=9).data,
    )
    assert restored.options.instrument == result.options.instrument
    assert restored.options.scale == result.options.scale
    assert restored.options.exposure == result.options.exposure
    assert restored.apertures[0].name == "line"
    assert np.array_equal(restored.psf, result.psf)
    assert restored.psf_pixel_scale == result.psf_pixel_scale
    assert np.array_equal(
        restored.apertures[0].spectra.snr,
        result.apertures[0].spectra.snr,
    )
    assert np.array_equal(
        restored.apertures[0].maps.signals.target,
        result.apertures[0].maps.signals.target,
    )
    assert np.array_equal(restored.models.transmission, result.models.transmission)
    assert np.array_equal(restored.models.sky, result.models.sky)
    assert np.array_equal(restored.models.thermal, result.models.thermal)
    with pytest.raises(FileExistsError):
        result.save(path)
    result.save(path, overwrite=True)


def test_fits_stores_key_datacubes_and_metadata(
    instrument_data,
    tmp_path,
) -> None:
    result = _result(instrument_data)
    path = tmp_path / "result.fits"

    result.save(path)

    with fits.open(path, checksum=True) as hdus:
        expected = {
            "PRIMARY",
            "WAVELEN",
            "SNR",
            "SIGNAL",
            "BACKGROUND",
            "VARIANCE",
        }
        assert {hdu.name for hdu in hdus} == expected
        assert hdus[0].header["NTARGET"] == 2
        assert hdus[0].header["NSKY"] == 2
        assert "NCUBES" not in hdus[0].header
        assert "PSFSCALE" not in hdus[0].header
        assert "PSFFILE" not in hdus[0].header
        assert hdus["SNR"].header["BUNIT"] == "1"
        assert hdus["SIGNAL"].header["BUNIT"] == "electron"
        assert hdus["BACKGROUND"].header["BUNIT"] == "electron"
        assert hdus["VARIANCE"].header["BUNIT"] == "electron2"
        assert hdus["SNR"].header["CTYPE2"] == "XOFFSET"
        wcs = WCS(hdus["SNR"].header)
        spectral_pixels = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        spectral_world = wcs.all_pix2world(spectral_pixels, 0)
        assert np.allclose(
            spectral_world[:, 0],
            result.wavelength[:2].to_value(u.m),
        )
        reference_pixel = np.array(
            [[0, (result.snr.shape[1] - 1) / 2, (result.snr.shape[0] - 1) / 2]]
        )
        reference_world = wcs.all_pix2world(reference_pixel, 0)[0]
        assert reference_world[0] == pytest.approx(
            result.wavelength[0].to_value(u.m)
        )
        assert reference_world[1] == pytest.approx(0.0, abs=1e-15)
        assert reference_world[2] == pytest.approx(0.0, abs=1e-15)
        pixel_scale = (50 * u.mas).to_value(u.deg)
        angle = np.deg2rad(30)
        assert hdus["SNR"].header["CD2_2"] == pytest.approx(
            -pixel_scale * np.cos(angle)
        )
        assert hdus["SNR"].header["CD2_3"] == pytest.approx(pixel_scale * np.sin(angle))
        assert hdus["SNR"].header["CD3_2"] == pytest.approx(pixel_scale * np.sin(angle))
        assert hdus["SNR"].header["CD3_3"] == pytest.approx(pixel_scale * np.cos(angle))
        assert hdus["SNR"].data.shape == result.snr.shape
        assert np.array_equal(hdus["SIGNAL"].data, result.signals.target.value)
        assert np.array_equal(
            hdus["BACKGROUND"].data,
            (
                result.signals.sky
                + result.signals.thermal
                + result.signals.dark
            ).value,
        )
        assert np.array_equal(
            hdus["VARIANCE"].data,
            result.variances.total.value,
        )


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
        wcs = WCS(header)
        reference_pixel = np.array(
            [[0, (result.snr.shape[1] - 1) / 2, (result.snr.shape[0] - 1) / 2]]
        )
        reference_world = wcs.all_pix2world(reference_pixel, 0)[0]
        assert reference_world[0] == pytest.approx(
            result.wavelength[0].to_value(u.m)
        )
        assert reference_world[1] == pytest.approx(120.0)
        assert reference_world[2] == pytest.approx(25.0)


def test_sampled_cube_saves_data_seed_metadata_and_wcs(
    instrument_data,
    tmp_path,
) -> None:
    result = _result(instrument_data)
    sample = result.sample(n=2, seed=42)
    path = tmp_path / "cube-samples.fits"

    sample.save(path)

    with fits.open(path, checksum=True) as hdus:
        assert {hdu.name for hdu in hdus} == {"PRIMARY", "WAVELEN", "DATA"}
        assert hdus[0].header["PRODUCT"] == "SAMPLED CUBE"
        assert hdus[0].header["RNGSEED"] == 42
        assert hdus[0].header["NREAL"] == 2
        assert hdus[0].header["INSTRUME"] == result.options.instrument
        assert hdus["DATA"].header["BUNIT"] == "electron"
        assert hdus["DATA"].header["CTYPE4"] == "REALIZATION"
        wcs = WCS(hdus["DATA"].header)
        spectral_pixels = np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0]],
            dtype=float,
        )
        spectral_world = wcs.all_pix2world(spectral_pixels, 0)
        assert np.allclose(
            spectral_world[:, 0],
            sample.wavelength[:2].to_value(u.m),
        )
        assert np.array_equal(hdus["DATA"].data, sample.data.value)
        assert np.array_equal(
            hdus["WAVELEN"].data,
            sample.wavelength.value,
        )

    with pytest.raises(ValueError, match="read-only"):
        sample.data[0, 0, 0, 0] = 0 * u.electron
    with pytest.raises(FileExistsError):
        sample.save(path)
    sample.save(path, overwrite=True)


def test_sampled_aperture_saves_data_seed_and_definition(
    instrument_data,
    tmp_path,
) -> None:
    result = _result(instrument_data)
    sample = result.apertures[0].sample(n=3, seed=91)
    path = tmp_path / "aperture-samples.fits"

    sample.save(path)

    with fits.open(path, checksum=True) as hdus:
        assert {hdu.name for hdu in hdus} == {
            "PRIMARY",
            "WAVELEN",
            "DATA",
            "MASK",
        }
        assert hdus[0].header["PRODUCT"] == "SAMPLED APERTURE"
        assert hdus[0].header["APERTURE"] == "line"
        assert hdus[0].header["RNGSEED"] == 91
        assert hdus[0].header["NREAL"] == 3
        assert hdus["DATA"].header["BUNIT"] == "electron"
        assert np.array_equal(hdus["DATA"].data, sample.data.value)
        assert np.array_equal(hdus["MASK"].data.astype(bool), sample.mask)
        assert np.array_equal(
            hdus["WAVELEN"].data,
            sample.wavelength.value,
        )

    with pytest.raises(ValueError, match="read-only"):
        sample.mask[0, 0, 0] = True


def test_sample_save_rejects_unknown_extension(instrument_data, tmp_path) -> None:
    result = _result(instrument_data)

    with pytest.raises(ValueError, match="FITS"):
        result.sample(seed=1).save(tmp_path / "sample.npy")
    with pytest.raises(ValueError, match="FITS"):
        result.apertures[0].sample(seed=1).save(tmp_path / "sample.pkl")


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
    etc.set_pointing(position_angle=30 * u.deg, sky_position=center)
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
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
    return etc.run(include_models=True)
