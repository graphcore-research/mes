import numpy as np

from boplay.acq_funs.mes_utils import sample_yn1_ymax, reconstruct_full_vector
from boplay.acq_funs.gamma_distribution import estimate_gamma_params, gamma_log_likelihood


def ves_mc_gamma(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    n_yn1: int=20,
    n_ymax: int=100,
    batch_size: int=1e9,
    idx_train: np.ndarray,
    lr: float = 1e-2,
    wd: float = 0.0,
) -> np.ndarray:
    """
    Cheap Variational Entropy Search acquisition function.
    Compute log likelihoods by Monte-carlo using a Gamma distribution
    for each y_n1 value within each x location.

    Args:
        x_test: np.ndarray, shape (n_x, x_dim)
        mean: np.ndarray, shape (n_x,)
        cov: np.ndarray, shape (n_x, n_x)
        y_best: float, best observed value
        n_yn1: int, number of y_n1 samples
        n_ymax: int, number of y_max samples
        batch_size: int, batch size for the optimizer
        idx_train: np.ndarray, indices of the training points

    Returns:
        np.ndarray, shape (n_x,)
    """
    idx_test = np.setdiff1d(np.arange(y_mean.shape[0]), idx_train)

    # only compute acquisition funciotn values for the unused points
    y_mean = y_mean[idx_test]
    y_cov = y_cov[idx_test, :][:, idx_test]

    y_n1_samples, _, y_max_samples, _ = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=batch_size,
    )

    n_x = len(idx_test)

    # (n_x, n_yn1)
    y_best_n1 = np.clip(y_n1_samples, min=y_best)

    # (n_x, n_yn1, n_ymax)
    y_max_shifted = y_max_samples - y_best_n1[:, :, None]

    # make sure they aren't exactly zero
    y_max_shifted = np.clip(y_max_shifted, min=1e-8)

    # (n_x * n_yn1, n_ymax) flatten so each row is samples from one distribution
    y_max_shifted = y_max_shifted.reshape(n_x * n_yn1, n_ymax)

    # (n_x * n_yn1, ), (n_x * n_yn1, )
    k, theta = estimate_gamma_params(x=y_max_shifted, lr=lr, wd=wd)

    # (n_x * n_yn1, n_ymax): compute likelihood for each data y_max
    log_likelihood = gamma_log_likelihood(x=y_max_shifted, k=k, theta=theta)

    # (n_x, n_yn1 * n_ymax)
    log_likelihood = log_likelihood.reshape(n_x, n_yn1 * n_ymax)

    # (n_x,): mean over all (y_n1, y_max) samples within each of the n_x locations.
    acq_fun_vals = log_likelihood.mean(axis=1)

    acq_fun_vals = reconstruct_full_vector(
        acq_fun_vals_idx_test=acq_fun_vals,
        idx_test=idx_test,
        n_x=x_grid.shape[0],
    )

    return acq_fun_vals






