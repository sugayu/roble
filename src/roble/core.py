'''Core items of roble.'''

from __future__ import annotations
from logging import getLogger
from typing import Sequence
import numpy as np
import astropy.units as u
from astropy.table import QTable
from . import model

__all__ = ['RobleCore', 'Resampler']

logger = getLogger(__name__)


##
class RobleCore:
    '''Core class for roble.'''

    def __init__(
        self,
        data: Sequence[model.BaseDetectorImage],
        instrument: model.BaseInstrument,
    ) -> None:
        self.datalist = data
        self.instrument = instrument

    def extract1d(
        self, aperture: model.BaseAperture, nchain: int = 2
    ) -> dict[str, u.Quantity]:
        '''Extract 1d spectra.'''
        chains = self.extract_chains(aperture, nchain)
        result = self.stack_chains(chains)
        return result

    def extract_chains(
        self, aperture: model.BaseAperture, nchain: int = 2
    ) -> dict[str, u.Quantity]:
        '''Extract 1d spectral chains.'''
        new_wave = self.construct_wavebins(nchain)
        new_flux = np.zeros(new_wave.shape) * self.datalist[0].intensity.unit
        new_sigma2 = np.zeros(new_wave.shape) * self.datalist[0].error.unit**2
        n = np.zeros(new_wave.shape, dtype=int)

        for data in self.datalist:
            mask_aperture = aperture.include(data)
            available = data.available & mask_aperture
            resample = Resampler(data.wavelength[available], new_wave.T.ravel())
            new_flux += resample(data.intensity[available]).reshape(-1, nchain).T
            new_sigma2 += resample(data.error[available] ** 2).reshape(-1, nchain).T
            n += resample.count_wherein().reshape(-1, nchain).T
        new_error = np.sqrt(new_sigma2)
        n[n == 0] = 1  # To avoid zero devision error

        return {
            'wavelength': new_wave,
            'flux': new_flux / n,
            'uncertainty': new_error / n,
        }

    def stack_chains(self, chains: dict[str, u.Quantity]) -> dict[str, u.Quantity]:
        '''Stack chains to produce 1d spectrum.'''
        if (chains['wavelength'].ndim == 1) or (chains['wavelength'].shape[0] == 1):
            logger.info('The input is already 1d. Return as it is.')
            return chains

        elif chains['wavelength'].shape[0] != 2:
            raise ValueError('nchain > 2 is not implemented yet.')

        wave = chains['wavelength']
        flux = chains['flux']
        unc = chains['uncertainty']

        w0 = wave.T.ravel()
        new_wave = (w0[1:] + w0[:-1]) / 2
        f0 = np.repeat(flux, 2, axis=1) / 2.0
        new_flux = f0[0, 1:] + f0[1, :-1]
        u0_2 = (np.repeat(unc, 2, axis=1) / 2.0) ** 2
        new_unc = np.sqrt(u0_2[0, 1:] + u0_2[1, :-1])

        return {
            'wavelength': new_wave,
            'flux': new_flux,
            'uncertainty': new_unc,
        }

    def construct_wavebins(self, nchain: int) -> u.Quantity:
        '''Construct new wavelength bins.'''
        wave_base = self.instrument.wavelength
        if nchain == 1:
            return wave_base
        else:
            dw = self.instrument.dispersion
            index = np.arange(len(wave_base))
            wlist = [wave_base]
            for i in range(1, nchain):
                step = i / nchain
                _wave = np.interp(index + step, index, wave_base)
                _wave[-1] = _wave[-2] + dw[-1]
                wlist.append(_wave)
        return np.vstack(wlist)

    def change_outputunits(
        self, spectra: dict[str, u.Quantity]
    ) -> dict[str, u.Quantity]:
        '''Change units of output spectra to erg/s/cm2/um.

        Currently, this assumes to recieve arguments given by extract1d.
        '''
        spectra['wavelength'] = spectra['wavelength'].to(u.um)
        spectra['flux'] = spectra['flux'] * self.instrument.pixelarea
        spectra['flux'] = spectra['flux'].to(
            u.erg / u.s / u.cm**2 / u.AA, u.spectral_density(spectra['wavelength'])
        )
        spectra['uncertainty'] = spectra['uncertainty'] * self.instrument.pixelarea
        spectra['uncertainty'] = spectra['uncertainty'].to(
            u.erg / u.s / u.cm**2 / u.AA, u.spectral_density(spectra['wavelength'])
        )
        return spectra


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

    def count_wherein(self, dtype: type = int) -> np.ndarray:
        '''Specify in which pixels the data stored.'''
        start = 0
        counts = []
        for i in self.idx:
            counts.append((i - start) > 0)
            start = i
        end = len(self.wavelength) - 1
        counts.append((end - i) > 0)
        return np.array(counts, dtype=dtype)
