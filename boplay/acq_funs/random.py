import numpy as np


def random_search(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    idx_train: np.ndarray,
    **usused,
) -> np.ndarray:
    """
    Random search acquisition function.

    Args:
        x_test: np.ndarray, shape (n_x, x_dim)
        mean: np.ndarray, shape (n_x,)
        cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        idx_train: np.ndarray, indices of the training points
    Returns:
        np.ndarray, shape (n_x,)
    """
    return np.random.rand(x_grid.shape[0])
