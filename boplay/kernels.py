import numpy as np


def se_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    len_scale: float = 10.0,
    sigma_f: float = 1.0,
) -> np.ndarray:
    """
    Isotropic squared exponential kernel.
    """
    assert len(x1.shape) == 2, "x1 must be a a matrix"
    assert len(x2.shape) == 2, "x2 must be a a matrix"
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same dimension"
    
    # (n1, 1, d) - (1, n2, d) -> (n1, n2, d)
    diff = (x1[:, None, :] - x2[None, :, :])**2
    diff = diff.sum(axis=2)
    scale = -0.5 / (len_scale * len_scale)
    return sigma_f**2 * np.exp(scale * diff)


KERNELS = {
    "se": se_kernel,
}