import numpy as np

from boplay.acq_funs.mes_utils import sample_yn1_ymax, reconstruct_full_vector
from boplay.acq_funs.gamma_distribution import estimate_gamma_params, gamma_log_likelihood
from scipy.special import gammaln
from scipy.stats import norm


def ves_gamma(
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
    Compute log likelihoods by Monte-carlo using a Gamma distribution
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

    y_mean = y_mean[idx_test]
    y_cov = y_cov[idx_test, :][:, idx_test]

    y_mean = y_mean.reshape(-1)
    y_n1_samples, _, y_max_samples, y_funcs = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        y_noise_std=y_noise_std,
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
    y_max_shifted = np.clip(y_max_shifted, min=1e-10)

    # (n_x , n_yn1 * n_ymax) flatten so each row is data to learn one (k, beta)
    y_max_shifted = y_max_shifted.reshape(n_x, n_yn1 * n_ymax)

    # (n_x, n_yn1*n_ymax), (n_x, n_yn1*n_ymax)
    k, theta = estimate_gamma_params(x=y_max_shifted, lr=lr, wd=wd)

    # NOTE: dead code here for future reference, we just use MC instead.
    # This is the analytic version of the function from the paper, however
    # this implementaion does not seem to work and their implementation
    # does not use this formula exactly :(
    # # expected improvement given by the formula
    # # E_y_n1[max(y_n1, y_best)] = mu + sig * pdf(z) + (y_best - mu) * cdf(z)
    # # where y_n1 ~ N(mu, sig^2) and z = (y_best - mu) / sig
    # y_sd = np.sqrt(np.diag(y_cov))
    # y_mean = y_mean.reshape(-1)
    # z_scores = (y_best - y_mean) / y_sd
    # ei_term = y_mean + y_sd * norm.pdf(z_scores) + (y_best - y_mean) * norm.cdf(z_scores)

    # # (n_x, )
    # # lets do this analytically
    # beta = 1 / theta
    # term_1 = k * np.log(beta)
    # term_2 = -gammaln(k)
    # term_3 = (k - 1) * np.log(y_max_shifted).mean(axis=1)
    # term_4 = -beta * y_funcs.max(axis=1).mean()  # scalar
    # term_5 = beta * ei_term

    # acq_fun_vals = term_1 + term_2 + term_3 + term_4 + term_5

    # (n_x, n_yn1 * n_ymax)
    log_likelihood = gamma_log_likelihood(x=y_max_shifted, k=k, theta=theta)

    # (n_x, )
    log_likelihood = log_likelihood.mean(axis=1)

    # (n_x,)
    acq_fun_vals = log_likelihood

    acq_fun_vals = reconstruct_full_vector(
        acq_fun_vals_idx_test=acq_fun_vals,
        idx_test=idx_test,
        n_x=x_grid.shape[0],
    )

    return acq_fun_vals






