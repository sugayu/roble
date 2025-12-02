'''Abstract base classes of a data container.
'''

from __future__ import annotations
from abc import ABC, abstractmethod
from logging import getLogger
import numpy as np
import astropy.units as u

__all__ = ['BaseDetectorImage']

logger = getLogger(__name__)


##
class BaseDetectorImage(ABC):
    '''Base class of the 2D detector image.'''

    def __init__(self) -> None:
        self.intensity: u.Quantity
        self.wavelength: u.Quantity
        self.uncertainty: u.Quantity
        self.ra: u.Quantity
        self.dec: u.Quantity
        self.x: np.ndarray
        self.y: np.ndarray
