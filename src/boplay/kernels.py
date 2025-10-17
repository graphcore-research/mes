import numpy as np

from functools import partial
from scipy.special import kv, gamma


def se_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    len_scale: float = 10.0,
    sigma_f: float = 1.0,
) -> np.ndarray:
    """
    Isotropic squared exponential kernel.

    Args:
        x1: np.ndarray, shape (n1, d)
        x2: np.ndarray, shape (n2, d)
        len_scale: float, length scale
        sigma_f: float, signal variance

    Returns:
        np.ndarray, shape (n1, n2)
    """
    assert len(x1.shape) == 2, "x1 must be a a matrix"
    assert len(x2.shape) == 2, "x2 must be a a matrix"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same dimension"

    # (n1, 1, d) - (1, n2, d) -> (n1, n2, d)
    diff = (x1[:, None, :] - x2[None, :, :]) ** 2
    diff = diff.sum(axis=2)
    scale = -0.5 / (len_scale * len_scale)
    return sigma_f**2 * np.exp(scale * diff)


def matern_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    len_scale: float = 1.0,
    sigma_f: float = 1.0,
    nu: float = 1.5,
) -> np.ndarray:
    """
    Vectorized Matérn kernel for arbitrary nu.
    Uses scipy.special.kv (modified Bessel function) and gamma.
    """
    assert len(x1.shape) == 2, "x1 must be a matrix"
    assert len(x2.shape) == 2, "x2 must be a matrix"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same dimension"

    # Pairwise Euclidean distance
    diff = (x1[:, None, :] - x2[None, :, :]) ** 2
    r = np.sqrt(diff.sum(axis=2)) / len_scale

    if nu == 0.5:
        # Special case: exponential kernel
        return sigma_f**2 * np.exp(-r)
    elif nu == 1.5:
        sqrt3_r = np.sqrt(3) * r
        return sigma_f**2 * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
    elif nu == 2.5:
        sqrt5_r = np.sqrt(5) * r
        return (
            sigma_f**2 * (1.0 + sqrt5_r + 5.0 * r**2 / 3.0) * np.exp(-sqrt5_r)
        )
    else:
        # General case
        factor = np.sqrt(2 * nu) * r / len_scale
        # Avoid division by zero when r = 0
        factor = np.where(r == 0.0, 1e-10, factor)
        prefactor = (2 ** (1.0 - nu)) / gamma(nu)
        return sigma_f**2 * prefactor * (factor**nu) * kv(nu, factor)


def min_kernel(x1, x2, **kwargs):
    """
    Min kernel.
    """
    assert len(x1.shape) == 2, "x1 must be a a matrix"
    assert len(x2.shape) == 2, "x2 must be a a matrix"
    assert x1.shape[1] == 1 and x2.shape[1] == 1, "x1 and x2 must be 1D arrays"

    _x1 = np.tile(x1.reshape(-1, 1, 1), (1, x2.shape[0], 1))
    _x2 = np.tile(x2.reshape(1, -1, 1), (x1.shape[0], 1, 1))

    diff = np.concatenate((_x1, _x2), axis=2)
    diff = np.min(diff, axis=2)

    assert diff.shape == (x1.shape[0], x2.shape[0])

    return diff


KERNELS = {
    "se": se_kernel,
    # "min": min_kernel,
    "matern-3/2": partial(matern_kernel, nu=1.5),
    "matern-5/2": partial(matern_kernel, nu=2.5),
}
