'''Core items of roble.'''

from __future__ import annotations
from logging import getLogger
import numpy as np
import astropy.units as u
from astropy.table import QTable
from . import model

__all__ = ['RobleCore', 'Resampler']

logger = getLogger(__name__)


##
class RobleCore:
    '''Core class for roble.'''

    def __init__(self, data: list[model.BaseDetectorImage]) -> None:
        self.datalist = data

    def extract1d(self, aperture: model.BaseAperture, nchain: int = 2) -> QTable:
        '''Extract 1d spectra with specified number of chains.'''

        new_wave = self.construct_wavebins(nchain)
        new_flux = np.zeros(new_wave.shape) * self.datalist[0].intensity.unit
        new_sigma2 = np.zeros(new_wave.shape) * self.datalist[0].error.unit**2

        for data in self.datalist:
            mask_aperture = aperture.include(data)
            available = data.available & mask_aperture
            resample = Resampler(data.wavelength[available], new_wave.T.ravel())
            new_flux += resample(data.intensity[available]).reshape(-1, nchain).T
            new_sigma2 += resample(data.error[available] ** 2).reshape(-1, nchain).T
        new_error = np.sqrt(new_sigma2)

        return {'wavelength': new_wave, 'flux': new_flux, 'uncertainty': new_error}

    def construct_wavebins(self, nchain: int) -> u.Quantity:
        '''Construct new wavelength bins.'''
        wbin = 1e-3
        new_wave = np.arange(2.8, 5.3, wbin / nchain) * u.um
        # new_wave = np.array([0.0, 1, 2]) * u.um
        return new_wave.reshape(-1, 2).T


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
        self.idx = np.searchsorted(wavelength[self.argsort], sep)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        start = 0
        data = data[self.argsort]
        new_data = []
        for i in self.idx:
            new_data.append(np.nansum(data[start:i]))
            start = i
        new_data.append(np.nansum(data[i:]))
        return u.Quantity(new_data)
