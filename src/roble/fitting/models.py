'''Fitting models'''

from __future__ import annotations
from logging import getLogger
import numpy as np
from astropy.modeling import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.modeling.functional_models import FLOAT_EPSILON

__all__ = ['AreaGaussian1D']

logger = getLogger(__name__)


##
class AreaGaussian1D(Fittable1DModel):
    """
    One dimensional Gaussian model with area as a parameter.

    Parameters
    ----------
    area : float or `~astropy.units.Quantity`.
        Integrated area
        Note: amplitude = area / (stddev * np.sqrt(2 * np.pi))
    mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian.
    stddev : float or `~astropy.units.Quantity`.
        Standard deviation of the Gaussian with FWHM = 2 * stddev * np.sqrt(2 * np.log(2)).

    Note
    -----
    This class is taken from a astropy document:
    https://docs.astropy.org/en/stable/modeling/jointfitter.html#example-spectral-line
    """

    area = Parameter(default=1)
    mean = Parameter(default=0)

    # Ensure stddev makes sense if its bounds are not explicitly set.
    # stddev must be non-zero and positive.
    stddev = Parameter(default=1, bounds=(FLOAT_EPSILON, None))

    @staticmethod
    def evaluate(x, area, mean, stddev):
        """
        AreaGaussian1D model function.
        """
        return (area / (stddev * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (x - mean) ** 2 / stddev**2
        )
