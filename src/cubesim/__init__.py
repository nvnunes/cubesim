"""Public package entrypoints for cubesim."""

from cubesim._version import __version__
from cubesim.etc import Etc
from cubesim.models import (
    ConstantVelocity,
    Gaussian,
    GaussianLines,
    Point,
    RotatingDisk,
    Sersic,
    SpatialImage,
    TabulatedSpectrum,
    Uniform,
    VelocityField,
)

__all__ = [
    "ConstantVelocity",
    "Etc",
    "Gaussian",
    "GaussianLines",
    "Point",
    "RotatingDisk",
    "Sersic",
    "SpatialImage",
    "TabulatedSpectrum",
    "Uniform",
    "VelocityField",
    "__version__",
]
