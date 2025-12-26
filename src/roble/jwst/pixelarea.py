'''Return pixel areas of JWST NIRSpec detector for IFU.'''

from __future__ import annotations
from logging import getLogger
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable

from .crds import CRDSContext

__all__ = ['make_pixareamap']

logger = getLogger(__name__)


##
def make_pixareamap(fname: Path) -> u.Quantity:
    '''Make a pixel area map from information contained in a fits file.'''

    with CRDSContext(fname) as crds:
        p_area = crds.get_path('AREA')
    tb = QTable.read(p_area, 1)

    slice_idmap = fits.getdata(fname, 'REGIONS')  # from 1 to 30
    pixarea = np.full_like(slice_idmap, np.nan, dtype=float)
    for row in tb:  # from 0 to 29
        pixarea[slice_idmap == (row['SLICE_ID'] + 1)] = row['PIXAREA']
    return pixarea * u.arcsec**2
