import numpy as np
from typing import Callable

from boplay.acq_funs.mes_utils import sample_yn1_ymax, reconstruct_full_vector


def ves_1d_regression_base(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    n_yn1: int=10,
    n_ymax: int=30,
    batch_size: int=1e9,
    idx_train: np.ndarray,
    model_fit_fun: Callable,
    model_fit_fun_kwargs: dict=None,
    expand_y_n1: bool=True,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using any regression model
    to predict y* values from y_n1 values.

    Args:
        x_grid: np.ndarray, shape (n_x, x_dim)
        y_mean: np.ndarray, shape (n_x,)
        y_cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        n_yn1: int, number of y_n1 samples
        n_ymax: int, number of y_max samples
        batch_size: int, batch size for the optimizer
        idx_train: np.ndarray, indices of the training points
        model_fit_fun: Callable, function to fit a 1D regression model and return fitness scores
        expand_y_n1: bool, whether to expand the y_n1 values to match the y_max_samples
        model_fit_fun_kwargs: dict, keyword arguments to pass to the model_fit_fun
    
    Returns:
        np.ndarray, shape (n_x,)
    """
    idx_test = np.setdiff1d(np.arange(y_mean.shape[0]), idx_train)

    y_mean = y_mean.reshape(-1, 1)
    y_mean = y_mean[idx_test, :]
    y_cov = y_cov[idx_test, :][:, idx_test]

    y_n1_samples, _, y_max_samples, _ = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=batch_size,
    )

    n_x = len(idx_test)

    if expand_y_n1:
        # expand the y_n1 values to match the y_max_samples
        # (n_x, n_yn1, n_ymax) <- (n_x, n_yn1, 1)
        y_n1_samples = np.tile(y_n1_samples[:, :, None], (1, 1, n_ymax))

        # reshape the y_n1 and y^* values  so that each row is one
        # dataset for 1D regression.
        # (n_x, n_yn1 * n_ymax)
        y_n1_samples = y_n1_samples.reshape(n_x, n_yn1 * n_ymax)

        # (n_x, n_yn1 * n_ymax)
        y_max_samples = y_max_samples.reshape(n_x, n_yn1 * n_ymax)

    if model_fit_fun_kwargs is None:
        model_fit_fun_kwargs = {}

    # fit a 1D regression model to each row and measure model fitness
    model_fitness_scores = model_fit_fun(
        x_data=y_n1_samples,
        y_data=y_max_samples,
        **model_fit_fun_kwargs,
    )

    model_fitness_scores = model_fitness_scores.reshape(-1)

    assert model_fitness_scores.shape == (n_x,), (
        f"model_fitness_scores.shape: {model_fitness_scores.shape}"
        f"n_x: {n_x} mismatch"
    )

    acq_fun_vals = reconstruct_full_vector(
        acq_fun_vals_idx_test=model_fitness_scores,
        idx_test=idx_test,
        n_x=x_grid.shape[0],
    )

    return acq_fun_vals




