'''User interface of roble'''

from __future__ import annotations
from typing import Sequence
from logging import getLogger
from pathlib import Path
from astropy.table import QTable

from . import model
from .core import RobleCore
from .jwst import JwstCalData, NIRSpecIFU

__all__ = ['Barrel']

logger = getLogger(__name__)


##
class Barrel:
    '''Entry point of roble.'''

    def __init__(
        self,
        data: Sequence[model.BaseDetectorImage],
        instrument: model.BaseInstrument,
    ) -> None:
        self._core = RobleCore(data, instrument)

    def extract1d(self, aperture: model.BaseAperture, nchain: int = 2) -> dict:
        '''Extract 1d spectra with specified number of chains.'''
        spec = self._core.extract1d(aperture, nchain)
        return self._core.change_outputunits(spec)

    @classmethod
    def from_jwst(cls, files: list[Path]) -> Barrel:
        data = [JwstCalData.from_file(f) for f in files]
        nirspec = NIRSpecIFU.from_file(files[0])
        return cls(data, nirspec)


class FlavorfulBarrel(Barrel):
    '''Entry point for detailed analyses.'''
