import numpy as np
from scipy.stats import norm


def expected_improvement(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    idx_train: np.ndarray,
    lr: float = 1e-2,
    wd: float = 0.0,
) -> np.ndarray:
    """
    Expected improvement acquisition function.

    Args:
        x_test: np.ndarray, shape (n_x, x_dim)
        mean: np.ndarray, shape (n_x,)
        cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        idx_train: np.ndarray, indices of the training points

    Returns:
        np.ndarray, shape (n_x,)
    """
    y_mean = y_mean.reshape(-1)

    sdev = np.sqrt(np.diag(y_cov))
    z = (y_mean - y_best) / sdev
    ei_vals = (y_mean - y_best) * norm.cdf(z) + sdev * norm.pdf(z)

    return ei_vals
