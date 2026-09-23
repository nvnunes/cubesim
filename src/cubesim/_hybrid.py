"""Lazy boundary between CubeSim's PSF contract and Hybrid AO PSF."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from scipy import ndimage

from cubesim._instrument import HybridDefinition, _resolve_reference
from cubesim._psf import Psf, load_psf
from cubesim._result import HybridOptions


def model_psf(
    *,
    root: Path,
    definition: HybridDefinition,
    coordinate_form: str,
    ngs_offsets: tuple[tuple[u.Quantity, u.Quantity], ...],
    ngs_magnitudes: u.Quantity,
    science_offset: tuple[u.Quantity, u.Quantity],
    wavelength: u.Quantity,
    zenith_angle: u.Quantity,
    ifu_rotation: u.Quantity,
) -> tuple[Psf, HybridOptions]:
    """Model one science position, then hand the image to the direct PSF path."""

    try:
        from hybrid_ao_psf import (
            HybridRequest,
            load_ngs_ho_metric_interpolator,
            load_science_ho_psf_interpolator,
            simulate,
        )
    except ImportError as exc:
        if exc.name == "hybrid_ao_psf":
            raise ImportError(
                "Hybrid PSF modelling requires the optional 'cubesim[hybrid]' "
                "extra. Install it with pip install 'cubesim[hybrid]'."
            ) from exc
        raise

    mastsel_path = _resolve_reference(root, definition.mastsel_ini_file)
    science_path = _resolve_reference(root, definition.science_ho_interpolator_file)
    ngs_path = _resolve_reference(root, definition.ngs_ho_interpolator_file)
    request = HybridRequest(
        science_x=u.Quantity([science_offset[0].to_value(u.arcsec)], u.arcsec),
        science_y=u.Quantity([science_offset[1].to_value(u.arcsec)], u.arcsec),
        ngs_x=u.Quantity(
            [point[0].to_value(u.arcsec) for point in ngs_offsets], u.arcsec
        ),
        ngs_y=u.Quantity(
            [point[1].to_value(u.arcsec) for point in ngs_offsets], u.arcsec
        ),
        ngs_magnitude=ngs_magnitudes,
        ngs_magnitude_zeropoint=definition.ngs_magnitude_zeropoint,
        wavelength=wavelength,
        zenith_angle=zenith_angle,
        mastsel_ini=mastsel_path.read_text(encoding="utf-8"),
    )
    result = simulate(
        request,
        load_science_ho_psf_interpolator(science_path),
        load_ngs_ho_metric_interpolator(ngs_path),
    )
    if result.psfs.shape[0] != 1:
        raise ValueError("Hybrid must return exactly one science PSF.")
    scale = result.metadata.pixel_scale
    if not scale.isscalar:
        scale = scale[0]
    image = rotate_to_ifu(result.psfs[0], ifu_rotation)
    psf = load_psf(image, pixel_scale=scale, instrument_root=root)
    options = HybridOptions(
        coordinate_form=coordinate_form,
        ngs_pointing_offsets=ngs_offsets,
        ngs_magnitudes=ngs_magnitudes,
        ngs_magnitude_zeropoint=definition.ngs_magnitude_zeropoint,
        ngs_flux=result.ngs_flux,
        science_position=science_offset,
        wavelength=wavelength,
        zenith_angle=zenith_angle,
        mastsel_ini_file=mastsel_path,
        science_ho_interpolator_file=science_path,
        ngs_ho_interpolator_file=ngs_path,
    )
    return psf, options


def rotate_to_ifu(image: np.ndarray, rotation: u.Quantity) -> np.ndarray:
    """Rotate a pointing-frame [y, x] image into IFU detector axes."""

    angle = rotation.to_value(u.deg) % 360
    if np.isclose(angle, 0):
        return np.array(image, copy=True)
    # ndimage's positive image rotation maps physical +x to -y in [y, x].
    transformed = ndimage.rotate(
        image,
        angle=angle,
        reshape=True,
        order=3,
        mode="constant",
        cval=0.0,
    )
    transformed = np.clip(transformed, 0, None)
    total = transformed.sum()
    if total <= 0:
        raise ValueError("Rotating the Hybrid PSF removed all flux.")
    return transformed * (np.sum(image) / total)
