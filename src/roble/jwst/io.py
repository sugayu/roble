'''I/O interface of jwst
'''

from __future__ import annotations
from logging import getLogger
import numpy as np

import jwst

from ..model.data import BaseDetectorImage

__all__ = ['JwstCalData']

logger = getLogger(__name__)


##
class JwstCalData(BaseDetectorImage):
    '''JWST data container for cal.fits data.'''

    def __init__(self) -> None:
        pass
