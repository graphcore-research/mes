import numpy as np

from boplay.acq_funs.mes_utils import sample_yn1_ymax, reconstruct_full_vector
from boplay.acq_funs.gamma_distribution import estimate_gamma_params, gamma_log_likelihood


def gaussian_log_likelihood(
    *,
    x: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """
    Compute the log likelihood of a Gaussian distribution.
    Args:
        x: np.ndarray, shape (n_x, n_points)
        mu: np.ndarray, shape (n_x,)
        var: np.ndarray, shape (n_x,)

    Returns:
        np.ndarray, shape (n_x, n_points)
    """
    mu = mu.reshape(-1, 1)
    var = var.reshape(-1, 1)

    assert x.shape[0] == mu.shape[0] == var.shape[0], (
        f"x.shape: {x.shape}, mu.shape: {mu.shape}, var.shape: {var.shape}"
    )


    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu)**2 / var


def ves_mc_gaussian(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    y_noise_std: float,
    n_yn1: int=20,
    n_ymax: int=100,
    batch_size: int=1e9,
    idx_train: np.ndarray,
    lr: float = 1e-2,
    wd: float = 0.0,
) -> np.ndarray:
    """
    Cheap Variational Entropy Search acquisition function.
    Compute log likelihoods by Monte-carlo using a Gaussian distribution
    for each y_n1 value within each x location.

    Args:
        x_test: np.ndarray, shape (n_x, x_dim)
        mean: np.ndarray, shape (n_x,)
        cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        y_noise_std: float, noise standard deviation of y values for the objective function.
        n_yn1: int, number of y_n1 samples
        n_ymax: int, number of y_max samples
        batch_size: int, batch size for the optimizer
        idx_train: np.ndarray, indices of the training points

    Returns:
        np.ndarray, shape (n_x,)
    """
    idx_test = np.setdiff1d(np.arange(y_mean.shape[0]), idx_train)
    n_x = len(idx_test)

    # only compute acquisition funciotn values for the unused points
    y_mean = y_mean[idx_test]
    y_cov = y_cov[idx_test, :][:, idx_test]

    y_n1_samples, _, y_max_samples, _ = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        y_noise_std=y_noise_std,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=batch_size,
    )

    # (n_x * n_yn1, n_ymax), (n_x * n_yn1, 1)
    y_max_samples = y_max_samples.reshape(n_x * n_yn1, n_ymax)
    means = y_max_samples.mean(axis=1)[:, None]  # (n_x * n_yn1, 1)
    vars = y_max_samples.var(axis=1)[:, None]  # (n_x * n_yn1, 1)

    # (n_x * n_yn1, n_ymax)
    log_likelihood = gaussian_log_likelihood(x=y_max_samples, mu=means, var=vars)

    log_likelihood = log_likelihood.reshape(n_x, n_yn1 * n_ymax)

    # (n_x,): mean over all (y_n1, y_max) samples within each of the n_x locations.
    acq_fun_vals = log_likelihood.mean(axis=1)

    acq_fun_vals = reconstruct_full_vector(
        acq_fun_vals_idx_test=acq_fun_vals,
        idx_test=idx_test,
        n_x=x_grid.shape[0],
    ) 

    return acq_fun_vals






