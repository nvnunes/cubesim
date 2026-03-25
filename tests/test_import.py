"""Smoke tests for the initial cubesim package scaffold."""

import cubesim


def test_package_exposes_version() -> None:
    assert isinstance(cubesim.__version__, str)
    assert cubesim.__version__
