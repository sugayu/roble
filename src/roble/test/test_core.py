'''Test core functions'''

import numpy as np
import astropy.units as u
from ..core import Resampler


def test_Resampler() -> None:
    w0 = np.arange(1.0, 21, 2) * u.AA
    w1 = np.array([3.5, 10.0, 14.0]) * u.AA
    f0 = np.ones_like(w0) * u.uJy

    resample = Resampler(w0, w1)
    f1 = resample(f0)
    assert np.all(np.isclose(f1, [3.0, 3.0, 4.0] * u.uJy * u.AA))
