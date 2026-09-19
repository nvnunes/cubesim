"""Direct PSF input contract tests."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

import cubesim


def test_set_psf_copies_centers_and_normalizes_array(instrument_data) -> None:
    source = np.zeros((5, 5), dtype=float)
    source[0, 1] = 1.0
    original = source.copy()
    etc = cubesim.Etc(instrument_data)

    etc.set_psf(source, pixel_scale=10 * u.mas)

    assert etc._psf is not None
    assert np.array_equal(source, original)
    assert np.isclose(etc._psf.data.sum(), 1.0)
    assert np.unravel_index(etc._psf.data.argmax(), etc._psf.data.shape) == (2, 2)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1.902207761e-7, 9.809694281587807e-1, 0, 5.54340893505e-5],
            [0, 0, 0, 1.61017081762e-5, 0],
            [0, 3.6761206e-9, 1.89577708526932e-2, 0, 1.0712941027e-6],
        ]
    )
    assert np.allclose(etc._psf.data, expected, rtol=1e-11, atol=1e-14)
    assert not etc._psf.data.flags.writeable
    assert etc._psf.pixel_scale == 10 * u.mas


@pytest.mark.parametrize("shape", [(31, 35), (32, 36)])
def test_set_psf_centers_smooth_asymmetric_odd_and_even_arrays(
    instrument_data,
    shape: tuple[int, int],
) -> None:
    y, x = np.indices(shape)
    source = np.exp(
        -0.5
        * (
            ((x - ((shape[1] - 1) / 2 + 0.7)) / 1.3) ** 2
            + ((y - ((shape[0] - 1) / 2 - 0.45)) / 1.7) ** 2
        )
    )
    etc = cubesim.Etc(instrument_data)

    etc.set_psf(source, pixel_scale=10 * u.mas)

    assert etc._psf is not None
    center_x, center_y = _pixel_center_centroid(etc._psf.data)
    assert center_x == pytest.approx(shape[1] / 2, abs=1e-5)
    assert center_y == pytest.approx(shape[0] / 2, abs=1e-5)
    assert etc._psf.data.sum() == pytest.approx(1.0)


def test_set_psf_loads_relative_npy(instrument_data) -> None:
    np.save(instrument_data / "psf.npy", np.ones((3, 3)))
    etc = cubesim.Etc(instrument_data)

    etc.set_psf("psf.npy", pixel_scale=0.01 * u.arcsec)

    assert etc._psf is not None
    assert etc._psf.path == (instrument_data / "psf.npy").resolve()
    assert np.isclose(etc._psf.pixel_scale.to_value(u.mas), 10.0)


def test_set_psf_loads_fits_pixel_scale(instrument_data) -> None:
    path = instrument_data / "psf.fits"
    fits.writeto(path, np.ones((3, 3)), header=fits.Header({"PIXSCALE": 4.0}))
    etc = cubesim.Etc(instrument_data)

    etc.set_psf(path)

    assert etc._psf is not None
    assert etc._psf.pixel_scale == 4 * u.mas


def test_set_psf_rejects_fits_without_pixel_scale(instrument_data) -> None:
    path = instrument_data / "psf.fits"
    fits.writeto(path, np.ones((3, 3)))
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match="missing PIXSCALE"):
        etc.set_psf(path)


def test_set_psf_rejects_pickle_backed_npy(instrument_data) -> None:
    path = instrument_data / "psf.npy"
    np.save(path, np.array([[object()]], dtype=object), allow_pickle=True)
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match="allow_pickle=False"):
        etc.set_psf(path, pixel_scale=1 * u.mas)


@pytest.mark.parametrize(
    ("array", "error"),
    [
        (np.ones(3), "2D"),
        (np.empty((0, 0)), "non-empty"),
        (np.array([[True]]), "real numeric"),
        (np.array([[1 + 1j]]), "real numeric"),
        (np.array([["1"]]), "real numeric"),
        (np.array([[np.nan]]), "finite"),
        (np.array([[-1.0, 2.0]]), "non-negative"),
        (np.zeros((2, 2)), "strictly positive"),
    ],
)
def test_set_psf_rejects_invalid_arrays(instrument_data, array, error) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises((TypeError, ValueError), match=error):
        etc.set_psf(array, pixel_scale=1 * u.mas)


@pytest.mark.parametrize(
    "pixel_scale",
    [1.0, True * u.mas, (1 + 1j) * u.mas, 0 * u.mas, np.inf * u.mas, 1 * u.s],
)
def test_set_psf_rejects_invalid_pixel_scale(instrument_data, pixel_scale) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises((TypeError, ValueError, u.UnitConversionError)):
        etc.set_psf(np.ones((2, 2)), pixel_scale=pixel_scale)


def test_set_psf_rejects_pickle_file(instrument_data) -> None:
    path = instrument_data / "psf.pkl"
    path.write_bytes(b"not loaded")
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(ValueError, match="FITS or NPY"):
        etc.set_psf(path, pixel_scale=1 * u.mas)


def _pixel_center_centroid(array: np.ndarray) -> tuple[float, float]:
    y, x = np.indices(array.shape)
    total = array.sum()
    return (
        float(((x + 0.5) * array).sum() / total),
        float(((y + 0.5) * array).sum() / total),
    )
