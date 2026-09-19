"""Loading and normalization for the direct PSF input contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class Psf:
    """Validated, centred, unit-normalized PSF state."""

    data: np.ndarray
    pixel_scale: u.Quantity
    path: Path | None


def load_psf(
    source: str | Path | Any,
    *,
    pixel_scale: Any | None,
    instrument_root: Path,
) -> Psf:
    """Load and validate one direct PSF input."""

    path: Path | None = None
    if isinstance(source, (str, Path)):
        path = _resolve_psf_path(source, instrument_root)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            if pixel_scale is None:
                raise ValueError("NPY PSFs require pixel_scale.")
            data = np.load(path, allow_pickle=False)
        elif suffix in {".fit", ".fits", ".fts"}:
            if pixel_scale is not None:
                raise ValueError(
                    "FITS PSFs read pixel scale from PIXSCALE; do not pass pixel_scale."
                )
            data, header = fits.getdata(path, header=True)
            if "PIXSCALE" not in header:
                raise ValueError("FITS PSF header is missing PIXSCALE in mas per pixel.")
            header_scale = np.asarray(header["PIXSCALE"])
            if header_scale.shape != () or header_scale.dtype.kind not in {
                "f",
                "i",
                "u",
            }:
                raise TypeError("FITS PIXSCALE must be a real numeric scalar.")
            pixel_scale = header_scale.item() * u.mas
        else:
            raise ValueError("PSF files must use FITS or NPY format.")
    else:
        if pixel_scale is None:
            raise ValueError("In-memory PSFs require pixel_scale.")
        data = source

    data = _validate_array(data)
    pixel_scale = _validate_pixel_scale(pixel_scale)
    data = _center_psf(data)
    data /= data.sum()
    data.setflags(write=False)
    return Psf(data=data, pixel_scale=pixel_scale, path=path)


def _resolve_psf_path(source: str | Path, instrument_root: Path) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = instrument_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PSF file does not exist: {path}")
    return path


def _validate_array(data: Any) -> np.ndarray:
    source = np.asarray(data)
    if source.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("PSF must be a real numeric 2D array.")
    try:
        array = np.array(source, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("PSF must be a numeric 2D array.") from exc

    if array.ndim != 2:
        raise ValueError("PSF must be a 2D array.")
    if array.size == 0:
        raise ValueError("PSF must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError("PSF must contain only finite values.")
    if np.any(array < 0):
        raise ValueError("PSF must be non-negative.")
    if array.sum() <= 0:
        raise ValueError("PSF total flux must be strictly positive.")
    return array


def _validate_pixel_scale(pixel_scale: Any) -> u.Quantity:
    if not isinstance(pixel_scale, u.Quantity):
        raise TypeError("PSF pixel_scale must be an angular quantity.")
    if not pixel_scale.isscalar:
        raise TypeError("PSF pixel_scale must be scalar.")
    value = np.asarray(pixel_scale.value)
    if value.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("PSF pixel_scale must be a real numeric quantity.")
    try:
        pixel_scale = pixel_scale.to(u.mas)
    except u.UnitConversionError as exc:
        raise u.UnitConversionError(
            "PSF pixel_scale must have angular units."
        ) from exc
    if not np.isfinite(pixel_scale.value):
        raise ValueError("PSF pixel_scale must be finite.")
    if pixel_scale.value <= 0:
        raise ValueError("PSF pixel_scale must be strictly positive.")
    return pixel_scale


def _center_psf(data: np.ndarray) -> np.ndarray:
    filtered = ndimage.gaussian_filter(data, sigma=0.5)
    y_indices, x_indices = np.indices(data.shape)
    centroid_x = np.sum((x_indices + 0.5) * filtered) / filtered.sum()
    centroid_y = np.sum((y_indices + 0.5) * filtered) / filtered.sum()

    shift_y = data.shape[0] / 2 - centroid_y
    shift_x = data.shape[1] / 2 - centroid_x
    if abs(shift_y) < 0.01 and abs(shift_x) < 0.01:
        return data

    centered = ndimage.shift(
        data,
        shift=(shift_y, shift_x),
        order=3,
        mode="constant",
        cval=0.0,
    )
    return np.clip(centered, 0, None)
