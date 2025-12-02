'''Core items of roble.'''

from __future__ import annotations
from logging import getLogger
import numpy as np
from astropy.table import QTable
from . import model

__all__ = ['RobleCore']

logger = getLogger(__name__)


##
class RobleCore:
    '''Core class for roble.'''

    def __init__(self, data: list[model.BaseDetectorImage]) -> None:
        self.datalist = data

    def extract1d(self, aperture: model.BaseAperture, nchain: int = 2) -> QTable:
        '''Extract 1d spectra with specified number of chains.'''

        new_wave = self.construct_wavebins(nchain)
        new_flux = np.zeros_like(new_wave)
        new_sigma2 = np.zeros_like(new_wave)

        for data in self.datalist:
            mask = aperture.include(data)
            resample = Resampler(data.wavelength[mask], new_wave.T.ravel())
            new_flux += resample(data.intensity[mask]).reshape(-1, nchain).T
            new_sigma2 += resample(data.uncertainty[mask] ** 2).reshape(-1, nchain).T

        new_uncertainty = np.sqrt(new_sigma2)
        return QTable(
            [new_wave, new_flux, new_uncertainty],
            names=['wavelength', 'flux', 'uncertainty'],
        )

    def construct_wavebins(self, chain: int) -> np.ndarray:
        '''Construct new wavelength bins.'''
        new_wave = np.array([0.0, 1, 2])
        return new_wave


class Resampler:
    '''Resampling class.

    Why is this class needed? because we want to hold a variable, argsort and idx,
    for sequential computations.
    This implements nearest neighbouring resampling base on searchsorted.
    '''

    def __init__(self, wavelength: np.ndarray, new_wave: np.ndarray) -> None:
        self.wavelength = wavelength
        self.new_wave = new_wave
        sep = (new_wave[1:] + new_wave[:-1]) / 2.0
        self.argsort = np.argsort(wavelength)
        self.idx = np.searchsorted(wavelength, sep)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        start = 0
        data = data[self.argsort]
        new_data = []
        for i in self.idx:
            new_data.append(np.mean(data[start:i]))
            start = i
        new_data.append(np.mean(data[i:]))
        return np.array(new_data)
