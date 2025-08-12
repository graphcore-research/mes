import numpy as np

from boplay.acq_funs.mes_utils import sample_yn1_ymax


def ves_mc_exponential(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    n_yn1: int=10,
    n_ymax: int=30,
    batch_size: int=1e9,
    idx_train: np.ndarray,
) -> np.ndarray:
    """
    Cheap Variational Entropy Search acquisition function.
    Compute log likelihoods by Monte-carlo using an exponential distribution
    for each y_n1 value within each x location.

    Args:
        x_grid: np.ndarray, shape (n_x, x_dim)
        y_mean: np.ndarray, shape (n_x,)
        y_cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        n_yn1: int, number of y_n1 samples
        n_ymax: int, number of y_max samples
        batch_size: int, batch size for the optimizer
        idx_train: np.ndarray, indices of the training points
    
    Returns:
        np.ndarray, shape (n_x,)
    """
    y_n1_samples, _, y_max_samples, _ = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=batch_size,
    )

    n_x = x_grid.shape[0]

    # (n_x, n_yn1)
    y_best_n1 = np.clip(y_n1_samples, min=y_best)

    # (n_x, n_yn1, n_ymax)
    y_max_shifted = y_max_samples - y_best_n1[:, :, None]

    # make sure they aren't exactly zero
    y_max_shifted = np.clip(y_max_shifted, min=1e-8)

    # (n_x * n_yn1, n_ymax) flatten so each row is samples from one distribution
    y_max_shifted = y_max_shifted.reshape(n_x * n_yn1, n_ymax)

    # (n_x * n_yn1, 1) estimate exponential lambda for each row
    lambda_values = 1.0 / y_max_shifted.mean(axis=1, keepdims=True)

    # (n_x * n_yn1, n_ymax)
    log_likelihood = np.log(lambda_values) - lambda_values * y_max_shifted

    # (n_x * n_yn1)
    log_likelihood = log_likelihood.sum(axis=1)

    # (n_x, n_yn1)
    log_likelihood = log_likelihood.reshape(n_x, n_yn1)

    # (n_x,)
    log_likelihood = log_likelihood.sum(axis=1)

    return log_likelihood



