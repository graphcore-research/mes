from typing import Callable
import numpy as np
import torch as pt

from boplay.acq_funs.ves_base import optimize_adam

from boplay.acq_funs.pytorch_settings import PT_DTYPE, PT_DEVICE

pt_params = {
    "dtype": PT_DTYPE,
    "device": PT_DEVICE,
}


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
    k_min: float = 0.01,
    k_max: float = 10.0,
    lr: float = 1e-2,
    wd: float = 0.,
    max_iters: int = 200,
    make_heatmap: bool = False,
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
    x_trend_pt = pt.tensor(x_trend, **pt_params)
    k_basis_pt = pt.tensor(k_basis, **pt_params)
    beta_basis_pt = pt.tensor(beta_basis, **pt_params)
    y_pt = pt.tensor(y_data, **pt_params)

    noise_vals = y_pt - x_trend_pt

    # push all the values up to be positive
    noise_vals_min = noise_vals.min(axis=1).values[:, None]
    noise_vals = noise_vals - noise_vals_min + 1e-6

    # (n_x, 4): initialized constant mean and constant std
    noise_mean_emp = y_pt.mean(axis=1)[:, None]
    params = pt.nn.Parameter(
        pt.concat(
            [
                pt.zeros(n_x, 1, **pt_params),
                pt.ones(n_x, 1, **pt_params),
                pt.zeros(n_x, 1, **pt_params),
                noise_mean_emp,
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
        m_k = params[:, 0, None]
        c_k = params[:, 1, None]
        m_theta = params[:, 2, None]
        c_theta = params[:, 3, None]

        k = m_k * k_basis_pt + c_k
        theta = m_theta * beta_basis_pt + c_theta

        k = k.clamp(min=k_min, max=k_max)
        theta = theta.clamp(min=1e-6, max=100)

        lhood = gamma_log_likelihood(x=noise_vals, k=k, theta=theta)

        # (1, ) <- (n_x, n_points)
        return -lhood.sum()

    params, _ = optimize_adam(theta=params, loss_fn=loss_fun, lr=lr, wd=wd, max_iters=max_iters)

    # evaluate to get final log likelihoods and final acquisition values
    # (n_x, n_points)
    m_k = params[:, 0, None]
    c_k = params[:, 1, None]
    m_theta = params[:, 2, None]
    c_theta = params[:, 3, None]

    k = m_k * k_basis_pt + c_k
    theta = m_theta * beta_basis_pt + c_theta

    gamma_lhood = gamma_log_likelihood(x=noise_vals, k=k,theta=theta)

    if not make_heatmap:
        # (n_x, )
        return gamma_lhood.sum(dim=1).detach().cpu().numpy()

    else:
        def make_heatmap(*, row_idx: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            Make a heatmap of the Gamma log likelihood for a given row of the data.
            """
            assert len(x.shape) == 1, "x must be a 1D array"
            assert len(y.shape) == 1, "y must be a 1D array"
            assert x.shape[0] == y.shape[0], "x and y must have the same length"

            trend_basis = trend_basis_fun(x)
            k_basis = k_basis_fun(x)
            theta_basis = beta_basis_fun(x)

            noise_vals = y - trend_basis

            mask = noise_vals <= 0

            # go to pytorch world
            noise_vals = pt.tensor(noise_vals, **pt_params)
            k_basis = pt.tensor(k_basis, **pt_params)
            theta_basis = pt.tensor(theta_basis, **pt_params)

            # (float, float, float, float)
            m_k, c_k, m_theta, c_theta = params[row_idx, :]

            # (n_x,)
            k = m_k * k_basis + c_k
            theta = m_theta * theta_basis + c_theta

            k = k.clamp(min=k_min, max=k_max)
            theta = theta.clamp(min=1e-6, max=100)

            # (n_x,)
            gamma_lhood = gamma_log_likelihood(
                x=noise_vals[None, :],
                k=k[None, :],
                theta=theta[None, :]
            )

            # and then come back to numpy
            gamma_lhood_np = gamma_lhood.detach().cpu().numpy().flatten()

            # negative values have probabilty=0, log(prob) = -inf
            # Gamma distribution only generates positive values
            gamma_lhood_np[mask] = -np.inf

            return gamma_lhood_np
        
        return make_heatmap