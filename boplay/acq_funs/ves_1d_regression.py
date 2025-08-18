from functools import partial

import numpy as np

from boplay.acq_funs.regression_models.lr import fit_lr_sklearn
from boplay.acq_funs.regression_models.lr_models import (
    fit_lr_1_0,
    fit_lr_1_1,
    fit_lr_1_2,
    fit_lr_2_0,
    fit_lr_2_1,
    fit_lr_2_2,
)

from boplay.acq_funs.ves_1d_regression_base import ves_1d_regression_base



ves_1d_regression_lr = partial(ves_1d_regression_base, model_fit_fun=fit_lr_sklearn)



def ves_1_0(
    *,
    y_best: float,
    **kwargs,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using a linear regression model
    with heteroskedastic noise to predict y* values from y_n1 values.
    """
    return ves_1d_regression_base(
        y_best=y_best,
        model_fit_fun=fit_lr_1_0,
        model_fit_fun_kwargs=dict(y_best=y_best),
        **kwargs,
    )


def ves_1_1(
    *,
    y_best: float,
    **kwargs,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using a linear regression model
    with heteroskedastic noise to predict y* values from y_n1 values.
    """
    return ves_1d_regression_base(
        y_best=y_best,
        model_fit_fun=fit_lr_1_1,
        model_fit_fun_kwargs=dict(y_best=y_best),
        **kwargs,
    )


def ves_1_2(
    *,
    y_best: float,
    **kwargs,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using a linear regression model
    with heteroskedastic noise to predict y* values from y_n1 values.
    """
    return ves_1d_regression_base(
        y_best=y_best,
        model_fit_fun=fit_lr_1_2,
        model_fit_fun_kwargs=dict(y_best=y_best),
        **kwargs,
    )


def ves_2_0(
    *,
    y_best: float,
    **kwargs,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using a linear regression model
    with heteroskedastic noise to predict y* values from y_n1 values.
    """
    return ves_1d_regression_base(
        y_best=y_best,
        model_fit_fun=fit_lr_2_0,
        model_fit_fun_kwargs=dict(y_best=y_best),
        **kwargs,
    )


def ves_2_1(
    *,
    y_best: float,
    **kwargs,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using a linear regression model
    with heteroskedastic noise to predict y* values from y_n1 values.
    """
    return ves_1d_regression_base(
        y_best=y_best,
        model_fit_fun=fit_lr_2_1,
        model_fit_fun_kwargs=dict(y_best=y_best),
        **kwargs,
    )


def ves_2_2(
    *,
    y_best: float,
    **kwargs,
) -> np.ndarray:
    """
    Variational Entropy Search acquisition function using a linear regression model
    with heteroskedastic noise to predict y* values from y_n1 values.
    """
    return ves_1d_regression_base(
        y_best=y_best,
        model_fit_fun=fit_lr_2_2,
        model_fit_fun_kwargs=dict(y_best=y_best),
        **kwargs,
    )