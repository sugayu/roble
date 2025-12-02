'''Resampling modules.
'''

from __future__ import annotations
from logging import getLogger
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

__all__ = [
    'sorted_kdtree',
    'searchsorted',
    'searchsorted_unsort',
    'interpolate',
    'interpolate_unsort',
]

logger = getLogger(__name__)


##
def sorted_kdtree(
    wavelength: np.ndarray, flux: np.ndarray, new_wave: np.ndarray
) -> np.ndarray:
    '''Nearest neibors by KDTree for sorted data array.'''
    sep = (new_wave[1:] + new_wave[:-1]) / 2.0

    argsort = np.argsort(wavelength)
    flux = flux[argsort]
    wavelength = wavelength[argsort]

    tree = KDTree(wavelength[:, np.newaxis])
    _, idx = tree.query(sep[:, np.newaxis])

    start = 0
    new_flux = []
    for i, j in enumerate(idx):
        end = j + 1 if (sep[i] - wavelength[j]) >= 0 else j
        new_flux.append(np.mean(flux[start:end]))
        start = end
    new_flux.append(np.mean(flux[end:]))

    return np.array(new_flux)


def searchsorted(
    wavelength: np.ndarray, flux: np.ndarray, new_wave: np.ndarray
) -> np.ndarray:
    '''Nearest neibors by np.searchsorted for sorted data array.'''
    sep = (new_wave[1:] + new_wave[:-1]) / 2.0

    argsort = np.argsort(wavelength)
    flux = flux[argsort]
    wavelength = wavelength[argsort]

    idx = np.searchsorted(wavelength, sep)

    start = 0
    new_flux = []
    for i in idx:
        new_flux.append(np.mean(flux[start:i]))
        start = i
    new_flux.append(np.mean(flux[i:]))

    return np.array(new_flux)


def searchsorted_unsort(
    wavelength: np.ndarray, flux: np.ndarray, new_wave: np.ndarray
) -> np.ndarray:
    '''Nearest neibors by np.searchsorted for unsorted data array.'''

    sep = (new_wave[1:] + new_wave[:-1]) / 2.0
    idx = np.searchsorted(sep, wavelength)
    new_flux = [np.mean(flux[idx == i]) for i in range(len(new_wave))]
    return np.array(new_flux)


def interpolate(
    wavelength: np.ndarray, flux: np.ndarray, new_wave: np.ndarray
) -> np.ndarray:
    '''Nearest neibors by nearest interpolation for sorted data array.'''
    sep = (new_wave[1:] + new_wave[:-1]) / 2.0

    argsort = np.argsort(wavelength)
    flux = flux[argsort]
    wavelength = wavelength[argsort]
    indecies = np.arange(len(wavelength))

    interpolate = interp1d(wavelength, indecies, kind='nearest')
    idx = interpolate(sep).astype(int)

    start = 0
    new_flux = []
    for i, j in enumerate(idx):
        end = j + 1 if (sep[i] - wavelength[j]) >= 0 else j
        new_flux.append(np.mean(flux[start:end]))
        start = end
    new_flux.append(np.mean(flux[end:]))

    return np.array(new_flux)


def interpolate_unsort(
    wavelength: np.ndarray, flux: np.ndarray, new_wave: np.ndarray
) -> np.ndarray:
    '''Nearest neibors by nearest interpolation for unsorted data array.'''
    indecies = np.arange(len(new_wave))
    interpolate = interp1d(new_wave, indecies, kind='nearest')
    idx = interpolate(wavelength)
    new_flux = [np.mean(flux[idx == i]) for i in indecies]
    return np.array(new_flux)
