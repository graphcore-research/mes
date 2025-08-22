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
        mean: pt.Tensor, shape (n_x, n_points)
        log_std: pt.Tensor, shape (n_x, n_points)

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
        noise_basis_fun: Callable, the basis function for the noise
        make_heatmap: bool, whether to make a heatmap of the Gaussian log likelihood

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
            pt.zeros(n_x, 1, **pt_params),
            y_mean_emp,
            pt.zeros(n_x, 1, **pt_params),
            y_log_std_emp,
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
        m_trend = params[:, 0, None]
        c_trend = params[:, 1, None]
        m_stdev = params[:, 2, None]
        c_stdev = params[:, 3, None]

        y_trend = m_trend * x_trend_pt + c_trend
        epsilon_log_sdev = m_stdev * x_noise_pt + c_stdev
        lhood = gaussian_log_likelihood(x=y_pt, mean=y_trend, log_std=epsilon_log_sdev)

        # (1, ) <- (n_x, n_points)
        return -lhood.sum()

    params, _ = optimize_adam(theta=params, loss_fn=loss_fun, lr=lr, wd=wd, max_iters=max_iters)

    m_trend = params[:, 0, None]
    c_trend = params[:, 1, None]
    m_stdev = params[:, 2, None]
    c_stdev = params[:, 3, None]

    y_trend = m_trend * x_trend_pt + c_trend
    epsilon_log_sdev = m_stdev * x_noise_pt + c_stdev
    gauss_lhood = gaussian_log_likelihood(x=y_pt, mean=y_trend, log_std=epsilon_log_sdev)


    gauss_lhood_vals = gauss_lhood.sum(dim=1).detach().cpu().numpy()

    if not make_heatmap:
        return gauss_lhood_vals

    else:
        def make_heatmap(*, row_idx: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            Make a heatmap of the Gaussian log likelihood for a given row of the data.
            """
            assert len(x.shape) == 1, "x must be a 1D array"
            assert len(y.shape) == 1, "y must be a 1D array"
            assert x.shape[0] == y.shape[0], "x and y must have the same length"

            trend_basis = trend_basis_fun(x)
            noise_basis = noise_basis_fun(x)

            trend_basis = pt.tensor(trend_basis, **pt_params)
            noise_basis = pt.tensor(noise_basis, **pt_params)

            m_trend, c_trend, m_stdev, c_stdev = params[row_idx, :]
            y_trend = m_trend * trend_basis + c_trend
            noise_log_sdev = m_stdev * noise_basis + c_stdev

            # go to pytorch world
            y = pt.tensor(y, **pt_params)
            y_trend = pt.tensor(y_trend, **pt_params)
            noise_log_sdev = pt.tensor(noise_log_sdev, **pt_params)

            # and then come back to numpy
            gauss_lhood = gaussian_log_likelihood(x=y, mean=y_trend, log_std=noise_log_sdev)
            return gauss_lhood.detach().cpu().numpy(), params
        
        return make_heatmap
