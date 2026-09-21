"""Shared variance propagation for in-field sky subtraction."""

from __future__ import annotations

import numpy as np


def in_field_marginal_variance(
    raw_variance: np.ndarray,
    sky_mask: np.ndarray,
) -> np.ndarray:
    """Propagate one shared in-field sky estimate into voxel variances."""

    sky_count = int(sky_mask.sum())
    estimator_variance = raw_variance[sky_mask].sum(axis=0) / sky_count**2
    covariance = np.zeros_like(raw_variance)
    covariance[sky_mask] = 2 * raw_variance[sky_mask] / sky_count
    return raw_variance + estimator_variance[None, None, :] - covariance


def in_field_aperture_variance_spectrum(
    aperture_mask: np.ndarray,
    sky_mask: np.ndarray,
    raw_variance: np.ndarray,
) -> np.ndarray:
    """Propagate a shared in-field sky estimate into a spatial reduction."""

    sky_count = int(sky_mask.sum())
    aperture_count = aperture_mask.sum(axis=(0, 1))
    weights = aperture_mask.astype(float) - (
        aperture_count[None, None, :] / sky_count
    ) * sky_mask[:, :, None]
    return np.sum(weights**2 * raw_variance, axis=(0, 1))
