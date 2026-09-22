"""Validation for the bundled example instrument."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cubesim


PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLE_DIRECTORY = PROJECT_ROOT / "example"
EXAMPLE_ROOT = EXAMPLE_DIRECTORY / "instrument_data"


def test_example_instrument_loads_every_configuration() -> None:
    etc = cubesim.Etc(EXAMPLE_ROOT)

    for scale in ("25mas", "50mas", "100mas"):
        for disperser in (
            "r3000.yj",
            "r3000.jh",
            "r3000.hk",
            "r8000.j",
            "r8000.h",
            "r8000.k",
        ):
            for atmosphere in ("airmass10", "airmass20"):
                etc.configure(
                    scale=scale,
                    disperser=disperser,
                    atmosphere=atmosphere,
                )

    assert etc._instrument.detector.dark_current == 0.05 * u.electron / u.s


def test_example_psf_and_calculation_use_public_api() -> None:
    etc = cubesim.Etc(EXAMPLE_ROOT)
    etc.configure(
        scale="100mas",
        disperser="r3000.yj",
        atmosphere="airmass10",
    )
    etc.set_psf("psf.fits")
    etc.add_target(
        ifu_offset=(0 * u.arcsec, 0 * u.arcsec),
        spatial=cubesim.Point(),
        spectrum=cubesim.GaussianLines(
            wavelength=1.2 * u.micron,
            flux=1e-17 * u.erg / (u.s * u.cm**2),
            dispersion=40 * u.km / u.s,
        ),
    )
    etc.set_exposure(time=60 * u.s, n_target=2)

    result = etc.run()

    assert result.snr.shape[:2] == (40, 40)
    assert result.snr.shape[2:] == result.wavelength.shape
    assert np.isfinite(result.snr).all()
    assert result.psf_pixel_scale == 10 * u.mas
    assert result.psf.sum() == 1


def test_readme_quickstart_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    quickstart = readme.split("## Quickstart: Python API", maxsplit=1)[1]
    quickstart = quickstart.split("\n## ", maxsplit=1)[0]
    source = quickstart.split("```python\n", maxsplit=1)[1]
    source = source.split("\n```", maxsplit=1)[0]

    monkeypatch.chdir(PROJECT_ROOT)
    namespace = {"__name__": "__main__"}
    exec(compile(source, "README.md:quickstart", "exec"), namespace)

    result = namespace["result"]
    assert result.snr.shape[:2] == (40, 40)
    assert np.isfinite(result.snr).all()


@pytest.mark.parametrize(
    ("filename", "output_filename"),
    [
        ("etc.ipynb", "etc_result.fits"),
        ("sampling.ipynb", "sampled_cube.fits"),
    ],
)
def test_example_notebook_runs_from_clean_copy(
    filename: str,
    output_filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = tmp_path / "example"
    shutil.copytree(
        EXAMPLE_DIRECTORY,
        example,
        ignore=shutil.ignore_patterns(
            ".DS_Store",
            "__pycache__",
            "etc_result.fits",
            "sampled_cube.fits",
        ),
    )
    monkeypatch.chdir(example)
    monkeypatch.syspath_prepend(str(example))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    matplotlib.use("Agg", force=True)
    monkeypatch.setattr(plt, "show", lambda: None)
    sys.modules.pop("inclined_galaxy", None)

    notebook = json.loads((example / filename).read_text())
    namespace = {"__name__": "__main__"}
    try:
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            exec(
                compile(source, f"{filename}:cell-{index}", "exec"),
                namespace,
            )
    finally:
        plt.close("all")

    assert (example / output_filename).is_file()
