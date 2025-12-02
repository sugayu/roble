'''User interface of roble'''

from __future__ import annotations
from logging import getLogger
from astropy.table import QTable

from . import model
from .core import RobleCore

__all__ = ['Barrel']

logger = getLogger(__name__)


##
class Barrel:
    '''Entry point of roble.'''

    def __init__(self, data: list[model.BaseDetectorImage]) -> None:
        self._core = RobleCore(data)

    def extract1d(self, aperture: model.BaseAperture, nchain: int = 2) -> QTable:
        '''Extract 1d spectra with specified number of chains.'''
        return self._core.extract1d(aperture, nchain)


class FlavorfulBarrel(Barrel):
    '''Entry point for detailed analyses.'''
