'''Fitting models'''

from __future__ import annotations
from logging import getLogger
import numpy as np
from astropy.modeling import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.modeling.functional_models import FLOAT_EPSILON, GAUSSIAN_SIGMA_TO_FWHM

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

    area = Parameter(default=1, description="Area of the Gaussian")
    mean = Parameter(default=0, description="Position of peak (Gaussian)")

    # Ensure stddev makes sense if its bounds are not explicitly set.
    # stddev must be non-zero and positive.
    stddev = Parameter(
        default=1,
        bounds=(FLOAT_EPSILON, None),
        description="Standard deviation of the Gaussian",
    )

    @staticmethod
    def evaluate(x, area, mean, stddev):
        """
        AreaGaussian1D model function.
        """
        return (area / (stddev * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (x - mean) ** 2 / stddev**2
        )

    @property
    def fwhm(self):
        """Gaussian full width at half maximum."""
        return self.stddev * GAUSSIAN_SIGMA_TO_FWHM

    @staticmethod
    def fit_deriv(x, area, mean, std):
        '''Gaussian1D model function derivatives.'''
        diff = x - mean
        d_area = np.exp(-0.5 / std**2 * diff**2) / (np.sqrt(2 * np.pi) * std)
        d_mean = area * d_area * diff / std**2
        d_std = area * d_area * ((diff**2 - std**2) / std**3)
        return [d_area, d_mean, d_std]

    @property
    def input_units(self):
        if self.mean.input_unit is None:
            return None
        return {self.inputs[0]: self.mean.input_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        '''inputs_unit = x, outputs_unit = y'''

        return {
            "area": outputs_unit[self.outputs[0]] * inputs_unit[self.inputs[0]],
            "mean": inputs_unit[self.inputs[0]],
            "stddev": inputs_unit[self.inputs[0]],
        }
