'''Produce wavelength bin arrays taking wave-dependence of dispersion into account.'''

from __future__ import annotations
from importlib import resources
from logging import getLogger
from pathlib import Path
import numpy as np
from astropy.table import QTable
import astropy.units as u

__all__ = ['produce_wavelengthbins', 'read_wavebins']

logger = getLogger(__name__)


##
def read_wavebins(key_disperser: str) -> QTable:
    '''Read wavelength bins from fits files.'''
    global fnames_wavelengths

    pdir = resources.files('roble.jwst.lib')
    if key_disperser in fnames_wavelengths.keys():
        fname_wave = fnames_wavelengths[key_disperser]
    else:
        raise KeyError(
            f'No dispersion key of {key_disperser}.'
            'Select from {fname_dispersion_curves.keys()}.'
        )
    return QTable.read(pdir / fname_wave)


def produce_wavelengthbins(key_disperser: str) -> None:
    '''Produce fiducial wavelength bins of roble 1d spectra.'''
    global fnames_dispersion_curves, fnames_wavelengths

    pdir = resources.files('roble.jwst.lib')
    if key_disperser in fnames_dispersion_curves.keys():
        fname_dispersion = fnames_dispersion_curves[key_disperser]
    else:
        raise KeyError(
            f'No dispersion key of {key_disperser}.'
            'Select from {fname_dispersion_curves.keys()}.'
        )

    tb = QTable.read(pdir / 'dispersion_curves' / fname_dispersion)
    wave = tb['WAVELENGTH'].value
    dispersion = tb['DLDS'].value
    crdspars = QTable.read(pdir / 'crds/jwst_nirspec_cubepar_0009.fits', 1)

    dwave = wave[1:] - wave[:-1]
    dwave = np.concatenate(([0], dwave))
    wmin = crdspars[crdspars['disperser'] == key_disperser]['wavemin'][0]
    wmax = crdspars[crdspars['disperser'] == key_disperser]['wavemax'][0]
    idx = np.searchsorted(wave, [wmin, wmax])
    s = slice(idx[0] - 1, idx[1] + 1)

    pix = np.cumsum(dwave[s] / dispersion[s])
    pix_offset = np.interp(wmin, wave[s], pix)
    length = int(np.interp(wmax, wave[s], pix) - pix_offset) + 1
    newwave = np.interp(np.arange(length) + pix_offset, pix, wave[s]) * u.um
    newdispersion = np.interp(np.arange(length) + pix_offset, pix, dispersion[s]) * u.um

    table = QTable([newwave, newdispersion], names=['wavelength', 'dispersion'])
    table.write(pdir / fnames_wavelengths[key_disperser])
    logger.info(f'Save: {pdir / fnames_wavelengths[key_disperser]}')


fnames_dispersion_curves = {
    'G140H': 'jwst_nirspec_g140h_disp.fits',
    'G140M': 'jwst_nirspec_g140m_disp.fits',
    'G235H': 'jwst_nirspec_g235h_disp.fits',
    'G235M': 'jwst_nirspec_g235m_disp.fits',
    'G395H': 'jwst_nirspec_g395h_disp.fits',
    'G395M': 'jwst_nirspec_g395m_disp.fits',
    'PRISM': 'jwst_nirspec_prism_disp.fits',
}

fnames_wavelengths = {
    'G140H': 'jwst_nirspec_g140h_wave.fits',
    'G140M': 'jwst_nirspec_g140m_wave.fits',
    'G235H': 'jwst_nirspec_g235h_wave.fits',
    'G235M': 'jwst_nirspec_g235m_wave.fits',
    'G395H': 'jwst_nirspec_g395h_wave.fits',
    'G395M': 'jwst_nirspec_g395m_wave.fits',
    'PRISM': 'jwst_nirspec_prism_wave.fits',
}

if __name__ == '__main__':
    for disperser in fnames_dispersion_curves.keys():
        produce_wavelengthbins(disperser)
