'''User interface of roble'''

from __future__ import annotations
from typing import Sequence
from logging import getLogger
from pathlib import Path
from astropy.table import QTable

from . import base
from .core import RobleCore
from .jwst import JwstCalData, NIRSpecIFU

__all__ = ['Barrel']

logger = getLogger(__name__)


##
class Barrel:
    '''Entry point of roble.'''

    def __init__(
        self,
        data: Sequence[base.BaseDetectorImage],
        instrument: base.BaseInstrument,
    ) -> None:
        self._core = RobleCore(data, instrument)

    def extract1d(self, aperture: base.BaseAperture, nchain: int = 2) -> dict:
        '''Extract 1d spectra.'''
        spec = self._core.extract1d(aperture, nchain)
        return self._core.change_outputunits(spec)

    def extract_chains(self, aperture: base.BaseAperture, nchain: int = 2) -> dict:
        '''Extract spectra with specified number of chains.'''
        chains = self._core.extract_chains(aperture, nchain)
        return self._core.change_outputunits(chains)

    @classmethod
    def from_jwst(cls, files: list[Path]) -> Barrel:
        data = [JwstCalData.from_file(f) for f in files]
        nirspec = NIRSpecIFU.from_file(files[0])
        return cls(data, nirspec)


class FlavorfulBarrel(Barrel):
    '''Entry point for detailed analyses.'''
