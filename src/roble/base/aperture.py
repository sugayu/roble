'''Model of apertures.
'''

from __future__ import annotations
from abc import ABC, abstractmethod
from logging import getLogger
import numpy as np
from .data import BaseDetectorImage

__all__ = ['BaseAperture']

logger = getLogger(__name__)


##
class BaseAperture(ABC):
    '''Base Aperture class.'''

    def __init__(self) -> None:
        pass

    @abstractmethod
    def include(self, data: BaseDetectorImage) -> np.ndarray:
        '''Check whether data is included in the aperture.'''
        raise NotImplementedError
