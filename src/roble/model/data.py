'''Abstract base classes of a data container.'''

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from logging import getLogger
import numpy as np
import astropy.units as u

__all__ = ['BaseDetectorImage', 'BaseInstrument']

logger = getLogger(__name__)


##
@dataclass
class BaseDetectorImage(ABC):
    '''Base class of the 2D detector image.'''

    wavelength: u.Quantity
    intensity: u.Quantity
    error: u.Quantity
    ra: u.Quantity
    dec: u.Quantity
    available: np.ndarray  # Pixels that are available. True is used.
    shape: tuple[int, int] = (0, 0)
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if (ndim := self.wavelength.ndim) != 2:
            raise ValueError(f'Input wavelength must have ndim of 2, but now {ndim}.')
        if (ndim := self.intensity.ndim) != 2:
            raise ValueError(f'Input intensity must have ndim of 2, but now {ndim}.')
        if (ndim := self.error.ndim) != 2:
            raise ValueError(f'Input error must have ndim of 2, but now {ndim}.')
        if (ndim := self.ra.ndim) != 2:
            raise ValueError(f'Input ra must have ndim of 2, but now {ndim}.')
        if (ndim := self.dec.ndim) != 2:
            raise ValueError(f'Input dec must have ndim of 2, but now {ndim}.')
        if (ndim := self.available.ndim) != 2:
            raise ValueError(f'Input available must have ndim of 2, but now {ndim}.')
        if (dtype := self.available.dtype) != bool:
            raise ValueError(f'Input available must be dtype=bool, but now {dtype}.')

        if self.shape == (0, 0):
            self.shape = self.intensity.shape
        if (self.x.size == 0) or (self.y.size == 0):
            self.y, self.x = np.mgrid[: self.shape[0], : self.shape[1]]


class BaseInstrument(ABC):
    '''Base class that contains information specific to instruments.'''

    def __init__(self) -> None:
        self._wavelength: u.Quantity
        self._dispersion: u.Quantity

    @property
    def wavelength(self) -> u.Quantity:
        '''Wavelength bins for 1d spec.'''
        return self._wavelength

    @property
    def dispersion(self) -> u.Quantity:
        '''Dispersion or wavelength difference between next pixels.'''
        return self._dispersion
