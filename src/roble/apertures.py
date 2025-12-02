'''Apertures.
'''

from __future__ import annotations
from logging import getLogger
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from .model.aperture import BaseAperture
from .model.data import BaseDetectorImage

__all__ = ['CircularAperture']

logger = getLogger(__name__)


##
class CircularAperture(BaseAperture):
    '''Base Aperture class.'''

    def __init__(self, position: SkyCoord, r: u.Quantity) -> None:
        self.position = position
        self.r = r.to(u.arcsec)

    def include(self, data: BaseDetectorImage) -> np.ndarray:
        dist = (data.ra - self.position.ra) ** 2 + (data.dec - self.position.dec) ** 2
        return dist.to(u.arcsec) < self.r**2
