# PSF Diagnostics

Import diagnostics from its public submodule:

```python
import cubesim.diagnostics as diagnostics
```

`psf_stats()` requires the optional `cubesim[stats]` extra and measures the
retained `result.psf` without changing it. The
returned `PsfStats` contains Strehl when full Hybrid metadata is available,
geometric-mean FWHM, and ensquared energy at the requested square-aperture
widths. See the [Python API guide](../api.md#measure-the-psf) for an example.

::: cubesim.diagnostics
    options:
      members_order: source
      show_root_heading: true
      show_source: false
