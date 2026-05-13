"""Public ETC API definitions."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PSF_KEY = "psf"
PIXEL_SCALE_KEY = "pixel_scale"


# Data structures
@dataclass(slots=True)
class EtcResult:
    """ETC result surface."""

    signal: Any | None = None
    noise_var: Any | None = None
    target_spatial_profile: Any | None = None
    target_wavelength: Any | None = None
    target_spectral_fwhm: Any | None = None


@dataclass(slots=True)
class EtcOptions:
    """ETC setup state."""

    instrument: str
    fov_config: dict[str, Any] = field(default_factory=dict)
    spec_config: dict[str, Any] = field(default_factory=dict)
    target_config: dict[str, Any] = field(default_factory=dict)
    atm_config: dict[str, Any] = field(default_factory=dict)
    analysis_config: dict[str, Any] = field(default_factory=dict)
    psf_data: dict[str, Any] | None = None
    psf_path: str | None = None
    is_finished: bool = False

    def config_fov(self, **kwargs: Any) -> None:
        self.fov_config.update(kwargs)

    def config_spec(self, **kwargs: Any) -> None:
        self.spec_config.update(kwargs)

    def config_target(self, **kwargs: Any) -> None:
        self.target_config.update(kwargs)

    def config_atm(self, **kwargs: Any) -> None:
        self.atm_config.update(kwargs)

    def config_analysis(self, **kwargs: Any) -> None:
        self.analysis_config.update(kwargs)

    def set_psf(self, psf: Any, pixel_scale: Any, **additional_keys: Any) -> None:
        """Attach validated PSF data to the options."""

        self.psf_data = _normalize_psf_payload(
            {
                PSF_KEY: psf,
                PIXEL_SCALE_KEY: pixel_scale,
                **additional_keys,
            }
        )
        self.psf_path = None

    def load_psf(self, path: str | Path) -> None:
        """Load persisted PSF data and attach it to the options."""

        payload = _load_psf_payload(path)
        self.psf_data = _normalize_psf_payload(payload)
        self.psf_path = str(path)

    def config_finish(self) -> None:
        self.is_finished = True


# Helper primitives
def _load_psf_payload(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PSF file does not exist: {path}")

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, Mapping):
        raise TypeError("Persisted PSF payload must be a mapping.")

    return payload


def _normalize_psf_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if PSF_KEY not in payload:
        raise KeyError(f"Persisted PSF payload is missing '{PSF_KEY}'.")
    if PIXEL_SCALE_KEY not in payload:
        raise KeyError(
            f"Persisted PSF payload is missing '{PIXEL_SCALE_KEY}'."
        )

    psf = _normalize_psf_array(payload[PSF_KEY])
    pixel_scale = _validate_positive_pixel_scale(payload[PIXEL_SCALE_KEY])

    normalized = dict(payload)
    normalized[PSF_KEY] = psf
    normalized[PIXEL_SCALE_KEY] = pixel_scale
    return normalized


def _normalize_psf_array(psf: Any) -> np.ndarray:
    array = np.asarray(psf, dtype=float)

    if array.ndim != 2:
        raise ValueError("PSF must be a 2D array.")
    if array.size == 0:
        raise ValueError("PSF must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError("PSF must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError("PSF must be non-negative.")

    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("PSF total flux must be strictly positive.")

    return array / total


def _validate_positive_pixel_scale(pixel_scale: Any) -> Any:
    if hasattr(pixel_scale, "to_value"):
        try:
            value = float(pixel_scale.to_value())
        except TypeError:
            pass
        else:
            if value <= 0.0:
                raise ValueError("PSF pixel_scale must be strictly positive.")
            return pixel_scale

    array = np.asarray(pixel_scale)
    if array.shape != ():
        raise TypeError("PSF pixel_scale must be scalar-like.")

    value = float(array)
    if value <= 0.0:
        raise ValueError("PSF pixel_scale must be strictly positive.")
    return pixel_scale


# Public entrypoints
def compute(options: EtcOptions) -> EtcResult:
    """Validate ETC setup state and run ETC compute.

    The compute implementation is not available yet.
    """

    if not isinstance(options, EtcOptions):
        raise TypeError("compute() requires an EtcOptions instance.")
    if options.psf_data is None:
        raise ValueError("EtcOptions must have PSF data before compute().")
    if not options.is_finished:
        raise ValueError("Call config_finish() before compute().")

    raise NotImplementedError("compute() is not implemented yet.")
