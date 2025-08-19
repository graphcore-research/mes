from typing import Callable
import numpy as np
import torch as pt

from boplay.acq_funs.ves_base import optimize_adam

from boplay.acq_funs.pytorch_settings import PT_DTYPE, PT_DEVICE

pt_params = {
    "dtype": PT_DTYPE,
    "device": PT_DEVICE,
}

half_log_pi_const = float(-0.5 * np.log(2 * np.pi))

def gaussian_log_likelihood(
    *,
    x: pt.Tensor,
    mean: pt.Tensor,
    log_std: pt.Tensor,
) -> pt.Tensor:
    """
    Compute the Gaussian log likelihood for data points, each row
    of x is a dataset with a mean and log_std.

    Args:
        x: pt.Tensor, shape (n_x, n_points)
        mean: pt.Tensor, shape (n_x,)
        log_std: pt.Tensor, shape (n_x,)

    Returns:
        lhood: pt.Tensor, shape (n_x, n_points)
    """
    assert x.shape == mean.shape == log_std.shape, (
        f"x.shape: {x.shape}, mean.shape: {mean.shape}, log_std.shape: {log_std.shape}"
    )
    sdev = log_std.exp().clip(1e-6)
    lhood = half_log_pi_const - log_std - 0.5 * (x - mean)**2 / sdev**2
    
    return lhood


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
    x_trend_pt = pt.tensor(x_trend, **pt_params)
    x_noise_pt = pt.tensor(x_noise, **pt_params)
    y_pt = pt.tensor(y_data, **pt_params)

    y_mean_emp = y_pt.mean(axis=1)[:, None]
    y_log_std_emp = pt.log(y_pt.std(axis=1)[:, None])

    # (n_x, 4): initialized constant mean and constant std
    params = pt.nn.Parameter(
        pt.concat(
        [
            y_mean_emp,
            pt.zeros(n_x, 1, **pt_params),
            y_log_std_emp,
            pt.zeros(n_x, 1, **pt_params),
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