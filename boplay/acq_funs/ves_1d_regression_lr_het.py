from functools import partial

import numpy as np
import torch as pt

from boplay.acq_funs.ves_1d_regression_base import ves_1d_regression_base



def fit_lr_het_models(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.

    Args:
        x_data: np.ndarray, shape (n_x, n_points)
        y_data: np.ndarray, shape (n_x, n_points)

    Returns:
        mse: np.ndarray, shape (n_x,)
    """
    n_x, n_points = x_data.shape

    # (n_x, n_points)
    x_pt = pt.tensor(x_data, dtype=pt.float32)
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

    const = float(-0.5 * np.log(2 * np.pi))

    def Gaussian_log_likelihood(params: pt.Tensor) -> pt.Tensor:
        """
        Compute the Gaussian Likelihood over data points.

        Args:
            params: pt.Tensor, shape (n_x, 4)

        Returns:
            lhood: pt.Tensor, shape (n_x,)
        """
        # (n_x, n_points)
        y_mean = params[:, 0, None] + params[:, 1, None] * x_pt
        y_log_sdev = params[:, 2, None] + params[:, 3, None] * x_pt
        y_sdev = y_log_sdev.exp()

        # (n_x, n_points)
        y_sdev = y_sdev.clip(min=1e-6)

        # (n_x, n_points)
        lhood = const - pt.log(y_sdev) - 0.5 * (y_pt - y_mean)**2 / y_sdev**2

        # (n_x, ) <- (n_x, n_points)
        lhood = lhood.sum(dim=1)

        return lhood

    # (n_x, )
    def loss_fun(params: pt.Tensor) -> float:
        """
        Compute the loss for a given set of ramp parameters.

        Args:
            params: pt.Tensor, shape (n_x, 4)

        Returns:
            loss: float, the loss
        """
        return -Gaussian_log_likelihood(params).mean()
    
    opt = pt.optim.Adam([params], lr=5e-2)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        L = loss_fun(params)
        L.backward()
        opt.step()
    
    return -Gaussian_log_likelihood(params).detach().cpu().numpy()


ves_1d_regression_lr_het = partial(ves_1d_regression_base, model_fit_fun=fit_lr_het_models)