"""Stateful public entrypoint for cubesim calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cubesim._instrument import InstrumentSelection, load_instrument
from cubesim._psf import Psf, load_psf


class Etc:
    """Configure calculations for one instrument-data directory.

    The directory must contain ``etc.ini`` at its root. Instrument files named
    by that configuration are resolved only relative to the directory and are
    validated when the ETC is constructed.
    """

    def __init__(self, instrument_data: str | Path) -> None:
        self._instrument = load_instrument(instrument_data)
        self._selection: InstrumentSelection | None = None
        self._psf: Psf | None = None

    def configure(
        self,
        *,
        spatial_mode: str,
        spectral_mode: str,
        atmosphere_mode: str,
    ) -> None:
        """Select one exact spatial, spectral, and atmosphere mode."""

        self._selection = self._instrument.select(
            spatial_mode=spatial_mode,
            spectral_mode=spectral_mode,
            atmosphere_mode=atmosphere_mode,
        )

    def set_psf(
        self,
        psf: str | Path | Any,
        *,
        pixel_scale: Any | None = None,
    ) -> None:
        """Set the achromatic PSF used by the next calculation.

        ``psf`` may be a FITS image, an NPY file, or an in-memory 2D array.
        File paths may be absolute or relative to the instrument-data
        directory. FITS obtains its pixel scale from ``PIXSCALE`` in mas and
        rejects an explicit ``pixel_scale``. NPY and in-memory arrays require
        an explicit positive angular quantity.
        """

        self._psf = load_psf(
            psf,
            pixel_scale=pixel_scale,
            instrument_root=self._instrument.root,
        )

    def run(
        self,
        *,
        include_models: bool = False,
        include_signals: bool = False,
        include_variances: bool = False,
        include_data: bool = False,
        n_cubes: int = 1,
        rng: Any | None = None,
    ) -> Any:
        """Run the configured forward ETC calculation.

        Forward ETC calculation is not implemented.
        """

        raise NotImplementedError("ETC calculation is not implemented yet.")
