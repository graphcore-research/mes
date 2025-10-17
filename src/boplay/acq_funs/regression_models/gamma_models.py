import numpy as np

from .gamma_het_base import fit_gamma_het_model

LR = 3e-2
WD = 0.0
MAX_ITERS = 200


def fit_gamma_0_0(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = LR,
    wd: float = WD,
    max_iters: int = MAX_ITERS,
    make_heatmap: bool = False,
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
        lr=lr,
        wd=wd,
        max_iters=max_iters,
        make_heatmap=make_heatmap,
    )


def fit_gamma_0_1(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = LR,
    wd: float = WD,
    max_iters: int = MAX_ITERS,
    make_heatmap: bool = False,
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
        lr=lr,
        wd=wd,
        max_iters=max_iters,
        make_heatmap=make_heatmap,
    )


def fit_gamma_0_2(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = LR,
    wd: float = WD,
    max_iters: int = MAX_ITERS,
    make_heatmap: bool = False,
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
        lr=lr,
        wd=wd,
        max_iters=max_iters,
        make_heatmap=make_heatmap,
    )


def fit_exp_0(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = LR,
    wd: float = WD,
    max_iters: int = MAX_ITERS,
    make_heatmap: bool = False,
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
        k_min=1,
        k_max=1,
        lr=lr,
        wd=wd,
        max_iters=max_iters,
        make_heatmap=make_heatmap,
    )


def fit_exp_1(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = LR,
    wd: float = WD,
    max_iters: int = MAX_ITERS,
    make_heatmap: bool = False,
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
        k_min=1,
        k_max=1,
        lr=lr,
        wd=wd,
        max_iters=max_iters,
        make_heatmap=make_heatmap,
    )


def fit_exp_2(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = LR,
    wd: float = WD,
    max_iters: int = MAX_ITERS,
    make_heatmap: bool = False,
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
        k_min=1,
        k_max=1,
        lr=lr,
        wd=wd,
        max_iters=max_iters,
        make_heatmap=make_heatmap,
    )
