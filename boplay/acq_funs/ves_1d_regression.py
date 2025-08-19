"""
Variational Entropy Search acquisition functions that use 1D regression.

This file imports all of the regression models and makes them into
acquisition functions.
"""

from typing import Callable

import numpy as np

from boplay.acq_funs.regression_models import (
    fit_lr_sklearn,
    fit_lr_1_0,
    fit_lr_1_1,
    fit_lr_1_2,
    fit_lr_2_0,
    fit_lr_2_1,
    fit_lr_2_2,
    fit_gamma_0_0,
    fit_gamma_0_1,
    fit_gamma_0_2,
    fit_exp_0,
    fit_exp_1,
    fit_exp_2,
)
from boplay.acq_funs.ves_1d_regression_base import ves_1d_regression_base


def acq_fun_maker(model_fit_fun: Callable) -> Callable:
    """
    Make an acquisition function that uses a given model fit function.
    """
    def acq_fun(
        *,
        y_best: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Acquisition function that uses a given model fit function.
        """
        return ves_1d_regression_base(
            y_best=y_best,
            model_fit_fun=model_fit_fun,
            model_fit_fun_kwargs=dict(x_min=y_best),
            **kwargs,
        )
    return acq_fun


ves_1d_regression_lr = acq_fun_maker(fit_lr_sklearn)
ves_lr_0_0 = acq_fun_maker(fit_lr_1_0)
ves_lr_0_1 = acq_fun_maker(fit_lr_1_1)
ves_lr_0_2 = acq_fun_maker(fit_lr_1_2)
ves_lr_2_0 = acq_fun_maker(fit_lr_2_0)
ves_lr_2_1 = acq_fun_maker(fit_lr_2_1)
ves_lr_2_2 = acq_fun_maker(fit_lr_2_2)
ves_gamma_0_0 = acq_fun_maker(fit_gamma_0_0)
ves_gamma_0_1 = acq_fun_maker(fit_gamma_0_1)
ves_gamma_0_2 = acq_fun_maker(fit_gamma_0_2)
ves_exp_0 = acq_fun_maker(fit_exp_0)
ves_exp_1 = acq_fun_maker(fit_exp_1)
ves_exp_2 = acq_fun_maker(fit_exp_2)