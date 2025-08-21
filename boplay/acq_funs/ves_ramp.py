"""
This is the exponential ramp acquisition function for VES.

For any VES implementation we only need three functions:
 - initialize_exponential_ramp_params: numpy function
 - compute_exponential_lambda_values: torch function
 - compute_exponential_log_likelihood: torch function

"""

import numpy as np
import torch as pt

from boplay.acq_funs.ves_base import ves_base


def initialize_exponential_ramp_params(
    *,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_n1_samples: np.ndarray,
    y_max_samples: np.ndarray,
    y_max_shifted: np.ndarray,
    y_best: float,
) -> np.ndarray:
    """
    Initialize the parameters for the y_n1 -> lambda functions.
    This function is vectorised over the n_x and the y_n1 dimension.

    NOTE: THIS IS ALL IN NUMPY!!

    Args:
        y_n1_samples: np.ndarray, shape (n_x, n_yn1)
        y_max_samples: np.ndarray, shape (n_x, n_yn1, n_ymax)
        y_best: float, best observed value

    Returns:
        np.ndarray, shape (n_x, 4)
    """
    n_yn1 = y_n1_samples.shape[1]

    # use the 1/3 and 2/3 quantiles of the y_n1 samples to initialise the
    # lower and upper thresholds for the y_n1 -> lambda functions.
    idx_lo = int(n_yn1 / 3)
    idx_hi = int(2 * n_yn1 / 3)

    y_lo = y_n1_samples[:, idx_lo]
    y_hi = y_n1_samples[:, idx_hi]

    # (n_x, n_yn1/3) <- (n_x, n_yn1/3, n_ymax)
    # use the empirical mean of the ymax values of y_n1 < y_lo (and y_n1 > y_hi)
    # to initialise the lambda values.
    ymax_lo_mean = y_max_shifted[:, :idx_lo, :].mean(axis=2)
    ymax_hi_mean = y_max_shifted[:, idx_hi:, :].mean(axis=2)

    # lambda = 1/empirical mean
    # (n_x, n_yn1/3) <- (n_x, n_yn1/3)
    l_lo = 1.0 / ymax_lo_mean
    l_hi = 1.0 / ymax_hi_mean

    # (n_x,) <- (n_x, n_yn1/3)
    # get average empirical lambda from all wedge under/over threshold
    l_lo = np.log(l_lo.mean(axis=1)) # (n_x,)
    l_hi = np.log(l_hi.mean(axis=1)) # (n_x,)

    return np.stack([y_lo, l_lo, y_hi, l_hi], axis=1) # (n_x, 4)


def compute_exponential_lambda_values(
    *,
    y_n1_samples: pt.Tensor,
    params: pt.Tensor,
) -> pt.Tensor:
    """
    Compute lambda values for a given set of "ramp" parameters.
    This function is vectorised over the n_x and the y_n1 dimension.

    NOTE: THIS IS ALL IN TORCH!!

    Args:
        y_n1: pt.Tensor, shape (n_x, n_yn1)
        params: pt.Tensor, shape (n_x, 4)

    Returns:
        pt.Tensor, shape (n_x, n_yn1)
    """
    y_lo = params[:, 0, None]
    l_lo = pt.exp(params[:, 1, None])
    y_hi = params[:, 2, None]
    l_hi = pt.exp(params[:, 3, None])

    dy = y_hi - y_lo
    dl = l_hi - l_lo

    y_clipped = pt.clip(y_n1_samples, min=y_lo, max=y_hi)

    return l_lo + (y_clipped - y_lo) * dl / dy


def compute_exponential_log_likelihood(
    *,
    y_max_shifted: pt.Tensor,
    distro_params: pt.Tensor,
) -> pt.Tensor:
    """
    Compute the log likelihood of the exponential distribution for a given
    set of y_max samples where the y-max samples have already been shifted
    to have a lower bound of 0.

    This function is vectorised over the n_x and the y_n1 dimension.

    NOTE: THIS IS ALL IN TORCH!!

    Args:
        y_max_shifted: pt.Tensor, shape (n_x, n_yn1, n_ymax)
        distro_params: pt.Tensor, shape (n_x, n_yn1)

    Returns:
        lhood: pt.Tensor, shape (n_x, n_yn1)
    """
    # each lambda value is for n_ymax values -> prep for broadcasting over ymax
    distro_params = distro_params[:, :, None]

    # (n_x, n_yn1, n_ymax) <- (n_x, n_yn1, n_ymax) - (n_x, n_yn1, 1)
    lhood = pt.log(distro_params) - distro_params * y_max_shifted

    # get likelihood for each lambda value by summing over ymax dimension
    # (n_x, n_yn1)
    lhood = pt.sum(lhood, dim=2)

    return lhood


def ves_exponential_ramp(
    *,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    y_noise_std: float,
    n_yn1: int=10,
    n_ymax: int=30,
    batch_size: int=1e9,
    idx_train: np.ndarray,
    lr: float = 1e-2,
    wd: float = 0.,
) -> np.ndarray:
    """
    Variational Entropy search with the exponenital ramp to approximate how
    the y-max distribution changes as a function of y_n1.

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
    return ves_base(
        y_mean=y_mean,
        y_cov=y_cov,
        y_best=y_best,
        y_noise_std=y_noise_std,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=batch_size,
        initialize_params=initialize_exponential_ramp_params,
        compute_distro_params=compute_exponential_lambda_values,
        compute_log_likelihood=compute_exponential_log_likelihood,
        idx_train=idx_train,
        lr=lr,
        wd=wd,
    )
