from functools import partial

from boplay.acq_funs.regression_models.lr import fit_lr_models
from boplay.acq_funs.regression_models.lr_het import fit_lr_het_models

from boplay.acq_funs.ves_1d_regression_base import ves_1d_regression_base



ves_1d_regression_lr = partial(ves_1d_regression_base, model_fit_fun=fit_lr_models)

ves_1d_regression_lr_het = partial(ves_1d_regression_base, model_fit_fun=fit_lr_het_models)

