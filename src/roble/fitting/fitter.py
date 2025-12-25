'''Fitter of astropy'''

from __future__ import annotations
from logging import getLogger
from scipy.linalg import solve_triangular
from astropy.modeling.fitting import _NLLSQFitter
import numpy as np

__all__ = ['TRFSQFitter', 'DogBoxLSQFitter', 'LMLSQFitter']

logger = getLogger(__name__)


##
class _CovarWeight_NLLSQFitter(_NLLSQFitter):
    '''Add objective_function considering a covariance matrix as weight.'''

    def objective_function(self, fps, *args) -> np.ndarray:
        '''
        Function to minimize.

        See scipy.optimize.fitting._NonLinearLSQFitter for details.
        Part of this code is from scipy.optimize.curve_fit.
        The argument "weights" is not the same as sigma. If it is 2d array,
        it's assumed to be already transformed from sigma by the Cholesky decomposition.
        '''

        weights: None | np.ndarray = args[1]
        measurements = args[-1]

        if weights is not None:
            if weights.shape == (measurements.size, measurements.size):
                residuals = super().objective_function(fps, (args[0], None) + args[2:])
                return solve_triangular(weights, residuals, lower=True)

        return super().objective_function(fps, *args)


class TRFSQFitter(_CovarWeight_NLLSQFitter):
    '''Wrapper of TRFSQFitter in astropy.'''

    def __init__(self, calc_uncertainties=False, use_min_max_bounds=False) -> None:
        super().__init__("trf", calc_uncertainties, use_min_max_bounds)


class DogBoxLSQFitter(_CovarWeight_NLLSQFitter):
    '''Wrapper of DogBoxLSQFitter in astropy.'''

    def __init__(self, calc_uncertainties=False, use_min_max_bounds=False):
        super().__init__("dogbox", calc_uncertainties, use_min_max_bounds)


class LMLSQFitter(_CovarWeight_NLLSQFitter):
    '''Wrapper of LMLSQFitter in astropy.'''

    def __init__(self, calc_uncertainties=False):
        super().__init__("lm", calc_uncertainties, True)
