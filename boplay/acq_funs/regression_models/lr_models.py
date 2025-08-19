import numpy as np

from .lr_het_base import fit_lr_het_model



def fit_lr_1_0(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = 1e-2,
    wd: float = 0.0,
    make_heatmap: bool = False,
) -> np.ndarray:
    """
    Fit a linear regression model with homoskedastic noise to each
    row of the data and return the Gaussian log likelihood.
    """
    return fit_lr_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: x,
        noise_basis_fun=lambda x: np.zeros_like(x),
        lr=lr,
        wd=wd,
        make_heatmap=make_heatmap,
    )


def fit_lr_1_1(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = 1e-2,
    wd: float = 0.0,
    make_heatmap: bool = False,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.
    """
    return fit_lr_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: x,
        noise_basis_fun=lambda x: x,
        lr=lr,
        wd=wd,
        make_heatmap=make_heatmap,
    )


def fit_lr_1_2(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = 1e-2,
    wd: float = 0.0,
    make_heatmap: bool = False,
) -> np.ndarray:
    """
    Fit a linear regression model with heteroskedastic noise to each
    row of the data and return the Gaussian log likelihood.
    """
    return fit_lr_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: x,
        noise_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        lr=lr,
        wd=wd,
        make_heatmap=make_heatmap,
    )


def fit_lr_2_0(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = 1e-2,
    wd: float = 0.0,
    make_heatmap: bool = False,
) -> np.ndarray:
    return fit_lr_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        noise_basis_fun=lambda x: np.zeros_like(x),
        lr=lr,
        wd=wd,
        make_heatmap=make_heatmap,
    )


def fit_lr_2_1(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = 1e-2,
    wd: float = 0.0,
    make_heatmap: bool = False,
) -> np.ndarray:
    return fit_lr_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        noise_basis_fun=lambda x: x,
        lr=lr,
        wd=wd,
        make_heatmap=make_heatmap,
    )


def fit_lr_2_2(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_min: float,
    lr: float = 1e-2,
    wd: float = 0.0,
    make_heatmap: bool = False,
) -> np.ndarray:
    return fit_lr_het_model(
        x_data=x_data,
        y_data=y_data,
        trend_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        noise_basis_fun=lambda x: np.clip(x, a_min=x_min, a_max=None),
        lr=lr,
        wd=wd,
        make_heatmap=make_heatmap,
    )

