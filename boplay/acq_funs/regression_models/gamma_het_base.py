from typing import Callable
import numpy as np
import torch as pt

from boplay.acq_funs.ves_base import optimize_adam


def gamma_log_likelihood(*, x: pt.Tensor, k: pt.Tensor, theta: pt.Tensor) -> pt.Tensor:
    """
    Compute the log likelihood of a Gamma distribution.

    Args:
        x: pt.Tensor, shape (n_x, n_points)
        k: pt.Tensor, shape (n_x, n_points)
        theta: pt.Tensor, shape (n_x, n_points)

    Returns:
        log_likelihood: pt.Tensor, shape (n_x, n_points)
    """
    assert len(x.shape) == 2, "x must be a 2D tensor"
    # k = k[:, None]÷
    # theta = theta[:, None]
    log_likelihood = (k - 1) * pt.log(x) - x / theta - k * pt.log(theta) - pt.lgamma(k)
    return log_likelihood


def fit_gamma_het_model(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    trend_basis_fun: Callable = None,
    k_basis_fun: Callable = None,
    beta_basis_fun: Callable = None,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.

    Args:
        x_data: np.ndarray, shape (n_x, n_points)
        y_data: np.ndarray, shape (n_x, n_points)
        trend_basis_fun: Callable, the basis function for the trend
        k_basis_fun: Callable, the basis function for the shape parameter
        beta_basis_fun: Callable, the basis function for the scale parameter

    Returns:
        mse: np.ndarray, shape (n_x,)
    """
    if trend_basis_fun is None:
        trend_basis_fun = lambda x: x
    if k_basis_fun is None:
        k_basis_fun = lambda x: x
    if beta_basis_fun is None:
        beta_basis_fun = lambda x: x

    n_x, n_points = x_data.shape

    # transform the input if necessary
    x_trend = trend_basis_fun(x_data)
    k_basis = k_basis_fun(x_data)
    beta_basis = beta_basis_fun(x_data)

    # ensure the shapes are unchanged
    assert x_data.shape == x_trend.shape == k_basis.shape == beta_basis.shape, (
        f"x_data.shape: {x_data.shape}, "
        f"x_trend.shape: {x_trend.shape}, "
        f"k_basis.shape: {k_basis.shape}, "
        f"beta_basis.shape: {beta_basis.shape}"
    )

    # (n_x, n_points)
    x_trend_pt = pt.tensor(x_trend, dtype=pt.float32)
    k_basis_pt = pt.tensor(k_basis, dtype=pt.float32)
    beta_basis_pt = pt.tensor(beta_basis, dtype=pt.float32)
    y_pt = pt.tensor(y_data, dtype=pt.float32)

    noise_vals = y_pt - x_trend_pt

    # make sure they're all (slightly) positive
    noise_vals_min = noise_vals.min(axis=1)[0] - 1e-6
    noise_vals_min = noise_vals_min.clip(min=None, max=0)

    noise_vals = noise_vals - noise_vals_min[:, None]

    # (n_x, 4): initialized constant mean and constant std
    noise_mean_emp = y_pt.mean(axis=1)[:, None]
    params = pt.nn.Parameter(
        pt.concat(
            [
                pt.ones(n_x, 1),
                pt.zeros(n_x, 1),
                noise_mean_emp,
                pt.zeros(n_x, 1),
            ],
            axis=1
        )
    )

    def loss_fun(params: pt.Tensor) -> float:
        """
        Compute the loss for a given set of parameters.

        Args:
            params: pt.Tensor, shape (n_x, 4)

        Returns:
            loss: float, the loss
        """

        # (n_x, n_points)
        k = params[:, 0, None] + params[:, 1, None] * k_basis_pt
        theta = params[:, 2, None] + params[:, 3, None] * beta_basis_pt

        k = k.clamp(min=1e-6, max=10)
        theta = theta.clamp(min=1e-6, max=100)

        lhood = gamma_log_likelihood(x=noise_vals, k=k, theta=theta)

        # (1, ) <- (n_x, n_points)
        return -lhood.sum()
    
    params, _ = optimize_adam(theta=params, loss_fn=loss_fun)

    # evaluate to get final log lieklihoods and final acquisition values
    # (n_x, n_points)
    k = params[:, 0, None] + params[:, 1, None] * k_basis_pt
    theta = params[:, 2, None] + params[:, 3, None] * beta_basis_pt
    gamma_lhood = gamma_log_likelihood(x=noise_vals, k=k,theta=theta)

    # (n_x, )
    return gamma_lhood.sum(dim=1).detach().cpu().numpy()