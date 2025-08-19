import numpy as np

from .gamma_het_base import fit_gamma_het_model



def fit_gamma_0_0(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
) -> np.ndarray:
    """
    Fit a linear regression model with homoskedastic noise to each
    row of the data and return the Gaussian log likelihood.
    """
    return fit_gamma_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        k_basis_fun=lambda x: np.zeros_like(x),
        beta_basis_fun=lambda x: np.zeros_like(x),
    )


def fit_gamma_0_1(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.
    """
    return fit_gamma_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        k_basis_fun=lambda x: np.zeros_like(x),
        beta_basis_fun=lambda x: x,
    )


def fit_gamma_0_2(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.
    """
    return fit_gamma_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        k_basis_fun=lambda x: np.zeros_like(x),
        beta_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
    )
