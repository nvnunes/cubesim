"""API tests for the Phase 1 cubesim scaffold."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import cubesim


def test_package_exposes_phase1_api() -> None:
    assert cubesim.EtcOptions is not None
    assert cubesim.EtcResult is not None
    assert cubesim.compute is not None


def test_options_load_psf_normalizes_payload(tmp_path) -> None:
    payload = {
        "psf": np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float),
        "pixel_scale": 5.0,
        "extra": "preserved",
    }
    path = tmp_path / "psf.pkl"
    path.write_bytes(pickle.dumps(payload))

    options = cubesim.EtcOptions("test-instrument")
    options.load_psf(path)

    assert options.psf_data is not None
    assert options.psf_path == str(path)
    assert np.isclose(options.psf_data["psf"].sum(), 1.0)
    assert options.psf_data["pixel_scale"] == 5.0
    assert options.psf_data["extra"] == "preserved"


def test_options_load_psf_rejects_missing_pixel_scale(tmp_path) -> None:
    path = tmp_path / "psf.pkl"
    path.write_bytes(pickle.dumps({"psf": np.ones((2, 2), dtype=float)}))

    options = cubesim.EtcOptions("test-instrument")

    with pytest.raises(KeyError):
        options.load_psf(path)


def test_compute_requires_finished_options_with_psf() -> None:
    options = cubesim.EtcOptions("test-instrument")
    options.set_psf(np.ones((2, 2), dtype=float), pixel_scale=1.0)

    with pytest.raises(ValueError):
        cubesim.compute(options)

    options.config_finish()

    with pytest.raises(NotImplementedError):
        cubesim.compute(options)
