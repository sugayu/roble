'''Test for resmapling methods.'''

# import time
# from logging import getLogger
# import numpy as np
# from numpy.random import default_rng
# from ..resample import (
#     sorted_kdtree,
#     searchsorted,
#     searchsorted_unsort,
#     interpolate,
#     interpolate_unsort,
# )

# logger = getLogger(__name__)


# def test_resample() -> None:
#     '''Test all resampling methods'''
#     size = 1_000_000
#     rng = default_rng()
#     flux = rng.uniform(1.0, 10.0, size=size)
#     wavelength = rng.uniform(4.0, 5.0, size=size)
#     new_wave = np.linspace(4.0, 5.0, 1000)

#     t0 = time.perf_counter()
#     new_flux0 = sorted_kdtree(wavelength, flux, new_wave)
#     t1 = time.perf_counter()
#     new_flux1 = searchsorted(wavelength, flux, new_wave)
#     t2 = time.perf_counter()
#     new_flux2 = searchsorted_unsort(wavelength, flux, new_wave)
#     t3 = time.perf_counter()
#     new_flux3 = interpolate(wavelength, flux, new_wave)
#     t4 = time.perf_counter()
#     new_flux4 = interpolate_unsort(wavelength, flux, new_wave)
#     t5 = time.perf_counter()
#     logger.info(f'sorted_kdtree: {t1-t0}s')
#     logger.info(f'searchsorted: {t2-t1}s')
#     logger.info(f'searchsorted: {t3-t2}s')
#     logger.info(f'interpolate: {t4-t3}s')
#     logger.info(f'interpolate2: {t5-t4}s')
#     assert np.all(np.isclose(new_flux0, new_flux1))
#     assert np.all(np.isclose(new_flux1, new_flux2))
#     assert np.all(np.isclose(new_flux2, new_flux3))
#     assert np.all(np.isclose(new_flux3, new_flux4))
