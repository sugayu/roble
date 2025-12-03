'''I/O interface of jwst
'''

from __future__ import annotations
from logging import getLogger
from pathlib import Path
import numpy as np

import astropy.units as u
from jwst import datamodels

from ..model.data import BaseDetectorImage

__all__ = ['JwstCalData']

logger = getLogger(__name__)


##
class JwstCalData(BaseDetectorImage):
    '''JWST data container for cal.fits data.'''

    FLAG_DO_NOT_USE: list = ['DO_NOT_USE', 'NON_SCIENCE']

    @classmethod
    def from_jwst(cls, filename: str | Path) -> JwstCalData:
        data = datamodels.open(filename)
        if not isinstance(data, datamodels.IFUImageModel):
            raise TypeError(
                'The input data must be jwst.datamodels.IFUImageModel, '
                f'but {filename} contains {type(data)}.'
            )

        ra, dec, wave = cls.get_wcs(data)
        wcsunits = data.meta.wcs.world.unit

        return cls(
            wavelength=wave * wcsunits[2],
            intensity=data.data * u.Unit(data.meta.bunit_data),
            error=data.err * u.Unit(data.meta.bunit_err),
            ra=ra * wcsunits[0],
            dec=dec * wcsunits[1],
            available=cls.get_availablearray(data.dq),
        )

    @staticmethod
    def get_wcs(
        data: datamodels.IFUImageModel,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Get wcs from JWST data model.

        Args:
            data (datamodels.IFUImageModel): Input cal.fits data model.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: RA, Dec, and wavelengths.
                These data have the same data shape of the input data.data.

        Examples:
            >>> ra, dec, wave = get_wcs(data)
        '''
        shape = data.data.shape
        y, x = np.mgrid[: shape[0], : shape[1]]
        return data.meta.wcs(x, y)

    @classmethod
    def get_availablearray(cls, dq: np.ndarray) -> np.ndarray:
        '''Get available pixels by interpreting dqflags.

        Args:
            data (datamodels.IFUImageModel): Input cal.fits data model.

        Returns:
            np.ndarray: Boolean array, where True is available.
        '''
        unavailable = np.zeros_like(dq, dtype=bool)
        for flag in cls.FLAG_DO_NOT_USE:
            _unavail = np.bitwise_and(dq, datamodels.dqflags.pixel[flag]).astype(bool)
            unavailable |= _unavail
        return ~unavailable
