'''Fitter of astropy'''

from __future__ import annotations
from logging import getLogger
from scipy.linalg import solve_triangular, cholesky, LinAlgError
import astropy.units as u
from astropy.modeling import FittableModel
from astropy.modeling.fitting import _NLLSQFitter
import numpy as np

__all__ = ['TRFSQFitter', 'DogBoxLSQFitter', 'LMLSQFitter']

logger = getLogger(__name__)


##
class _CovarWeight_NLLSQFitter(_NLLSQFitter):
    '''Add objective_function considering a covariance matrix as weight.'''

    def __call__(
        self, *args, sigma: None | u.Quantity = None, **kwargs
    ) -> FittableModel:
        '''Wrapper of call method to recieve a new argument "sigma".'''
        x = np.asarray(args[1])

        if sigma is not None:
            if ('weights' in kwargs) and (kwargs['weights'] is not None):
                raise ValueError('Either sigma or weights can be input.')

            sigma = np.asarray(sigma)
            if sigma.shape == (x.size, x.size):
                try:
                    # scipy.linalg.cholesky requires lower=True to return L L^T = A
                    weights = cholesky(sigma, lower=True)
                except LinAlgError as e:
                    raise ValueError("`sigma` must be positive definite.") from e

            else:
                weights = 1 / sigma

            kwargs['weights'] = weights

        return super().__call__(*args, **kwargs)

    def objective_function(self, fps, *args, **kwargs) -> np.ndarray:
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
                residuals = super().objective_function(
                    fps, (args[0], None) + args[2:], **kwargs
                )
                return solve_triangular(weights, residuals, lower=True)

        return super().objective_function(fps, *args, **kwargs)


class TRFLSQFitter(_CovarWeight_NLLSQFitter):
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
