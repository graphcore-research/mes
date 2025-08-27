import numpy as np
from scipy.stats import norm


def probability_of_improvement(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    y_indiff:float=0.01,
    idx_train: np.ndarray,
    **unused,
) -> np.ndarray:
    """
    Probability of improvement acquisition function.

    Args:
        x_grid: np.ndarray, shape (n_x, x_dim)
        y_mean: np.ndarray, shape (n_x,)
        y_cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        y_indiff: float, the smallest y increase we care about
        idx_train: np.ndarray, indices of the training points

    Returns:
        np.ndarray, shape (n_x,)
    """
    y_mean = y_mean.reshape(-1)

    sdev = np.sqrt(np.clip(np.diag(y_cov), 1e-6, None))
    y_best = y_best + y_indiff
    z = (y_mean - y_best) / sdev
    pi_vals = norm.cdf(z)

    return pi_vals
