"""Public package entrypoints for cubesim."""

from cubesim._version import __version__
from cubesim.etc import EtcOptions, EtcResult, compute

__all__ = ["__version__", "EtcOptions", "EtcResult", "compute"]
