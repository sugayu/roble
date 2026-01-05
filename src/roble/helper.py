'''Helper functions'''

from __future__ import annotations
from logging import getLogger
import numpy as np

__all__ = ['corrmatrix']

logger = getLogger(__name__)


##
def corrmatrix(*args, size: int) -> np.ndarray:
    '''Retrun correlation matrix.

    The input arguments indicate correlation value of n-th off-diagonal elements.
    '''
    corr = np.identity(size)
    for i, arg in enumerate(args, 1):
        r = np.full(size - i, arg)
        corr += np.diag(r, i) + np.diag(r, -i)
    return corr
