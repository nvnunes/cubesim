# Target Model API

Target models are immutable validated configuration objects. All of the names
below are also exported from the `cubesim` package root. See the
[Python API guide](../api.md#build-targets) for composition and flux-unit rules.

::: cubesim.models
    options:
      members:
        - Point
        - Uniform
        - Gaussian
        - Sersic
        - SpatialImage
        - GaussianLines
        - TabulatedSpectrum
        - ConstantVelocity
        - RotatingDisk
        - VelocityField
      members_order: source
      show_root_heading: true
      show_source: false
