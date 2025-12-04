'''User interface of roble'''

from __future__ import annotations
from logging import getLogger
from pathlib import Path
from astropy.table import QTable

from . import model
from .core import RobleCore
from .jwst import JwstCalData

__all__ = ['Barrel']

logger = getLogger(__name__)


##
class Barrel:
    '''Entry point of roble.'''

    def __init__(self, data: list[model.BaseDetectorImage]) -> None:
        self._core = RobleCore(data)

    def extract1d(self, aperture: model.BaseAperture, nchain: int = 2) -> dict:
        '''Extract 1d spectra with specified number of chains.'''
        return self._core.extract1d(aperture, nchain)

    @classmethod
    def from_jwst(cls, files: list[Path]) -> Barrel:
        return cls([JwstCalData.from_jwst(f) for f in files])


class FlavorfulBarrel(Barrel):
    '''Entry point for detailed analyses.'''
