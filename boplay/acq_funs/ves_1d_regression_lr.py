import numpy as np
from sklearn.linear_model import LinearRegression
from functools import partial

from boplay.acq_funs.ves_1d_regression_base import ves_1d_regression_base



def fit_lr_models(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> np.ndarray:
    """
    Fit a linear regression model to each row of the data and return the MSE.

    Args:
        x_data: np.ndarray, shape (n_x, n_points)
        y_data: np.ndarray, shape (n_x, n_points)

    Returns:
        mse: np.ndarray, shape (n_x,)
    """
    n_x, n_points = x_data.shape

    mse = np.zeros(n_x)

    for i in range(n_x):
        x = x_data[i, :, None]
        y = y_data[i, :, None]

        # fit a linear regression model from scipy.stats
        model = LinearRegression()
        model.fit(x, y)
        mse[i] = model.score(x, y)

    return mse





ves_1d_regression_lr = partial(ves_1d_regression_base, model_fit_fun=fit_lr_models)

