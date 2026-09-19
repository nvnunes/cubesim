"""Public API tests for the Phase 2 scaffold."""

from __future__ import annotations

import pytest

import cubesim


def test_package_exposes_etc_entrypoint(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    assert etc is not None
    assert not hasattr(cubesim, "EtcOptions")
    assert not hasattr(cubesim, "compute")


def test_run_is_explicitly_deferred(instrument_data) -> None:
    etc = cubesim.Etc(instrument_data)

    with pytest.raises(
        NotImplementedError, match="ETC calculation is not implemented yet"
    ):
        etc.run()
