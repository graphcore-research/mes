from typing import Callable
import numpy as np
import torch as pt

from boplay.acq_funs.ves_base import optimize_adam

from .utils import gaussian_log_likelihood


def fit_lr_het_model(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    trend_basis_fun: Callable = None,
    noise_basis_fun: Callable = None,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.

    Args:
        x_data: np.ndarray, shape (n_x, n_points)
        y_data: np.ndarray, shape (n_x, n_points)
        trend_basis_fun: Callable, the basis function for the trend
        noise_basis_fun: Callable, the basis function for the noise

    Returns:
        mse: np.ndarray, shape (n_x,)
    """
    if trend_basis_fun is None:
        trend_basis_fun = lambda x: x
    if noise_basis_fun is None:
        noise_basis_fun = lambda x: x

    n_x, n_points = x_data.shape

    # transform the input if necessary
    x_trend = trend_basis_fun(x_data)
    x_noise = noise_basis_fun(x_data)

    # ensure the shapes are unchanged
    assert x_data.shape == x_trend.shape == x_noise.shape, (
        f"x_data.shape: {x_data.shape}, "
        f"x_trend.shape: {x_trend.shape}, "
        f"x_noise.shape: {x_noise.shape}"
    )

    # (n_x, n_points)
    x_trend_pt = pt.tensor(x_trend, dtype=pt.float32)
    x_noise_pt = pt.tensor(x_noise, dtype=pt.float32)
    y_pt = pt.tensor(y_data, dtype=pt.float32)

    y_mean_emp = y_pt.mean(axis=1)[:, None]
    y_log_std_emp = pt.log(y_pt.std(axis=1)[:, None])

    # (n_x, 4): initialized constant mean and constant std
    params = pt.nn.Parameter(
        pt.concat(
        [
            y_mean_emp,
            pt.zeros(n_x, 1),
            y_log_std_emp,
            pt.zeros(n_x, 1),
        ],
        axis=1
        )
    )

    def loss_fun(params: pt.Tensor) -> float:
        """
        Compute the loss for a given set of ramp parameters.

        Args:
            params: pt.Tensor, shape (n_x, 4)

        Returns:
            loss: float, the loss
        """
        # (n_x, n_points)
        y_mean = params[:, 0, None] + params[:, 1, None] * x_trend_pt
        y_log_sdev = params[:, 2, None] + params[:, 3, None] * x_noise_pt
        lhood = gaussian_log_likelihood(x=y_pt, mean=y_mean, log_std=y_log_sdev)

        # (1, ) <- (n_x, n_points)
        return -lhood.sum()
    
    params, _ = optimize_adam(theta=params, loss_fn=loss_fun)


    y_mean = params[:, 0, None] + params[:, 1, None] * x_trend_pt
    y_log_sdev = params[:, 2, None] + params[:, 3, None] * x_noise_pt
    gauss_lhood = gaussian_log_likelihood(x=y_pt, mean=y_mean, log_std=y_log_sdev)
    
    return gauss_lhood.sum(dim=1).detach().cpu().numpy()