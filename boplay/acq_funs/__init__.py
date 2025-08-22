from .ei import expected_improvement
from .random import random_search
from .ves_ramp import ves_exponential_ramp
from .ves_mc_expo import ves_mc_exponential
from .ves_mc_gamma import ves_mc_gamma
from .ves_mc_gaussian import ves_mc_gaussian
from .ves_gamma import ves_gamma
from .ves_1d_regression import (
    ves_1d_regression_lr,
    ves_lr_0_0,
    ves_lr_0_1,
    ves_lr_0_2,
    ves_lr_2_0,
    ves_lr_2_1,
    ves_lr_2_2,
    ves_gamma_0_0,
    ves_gamma_0_1,
    ves_gamma_0_2,
    ves_exp_0,
    ves_exp_1,
    ves_exp_2,
)



ACQ_FUNCS = {
    "ves_ramp": ves_exponential_ramp,
    "expected_improvement": expected_improvement,
    "random_search": random_search,
    "ves_mc_exponential": ves_mc_exponential,
    "ves_mc_gamma": ves_mc_gamma,
    "ves_mc_gaussian": ves_mc_gaussian,
    "ves_gamma": ves_gamma,
    "ves_1d_regression_lr": ves_1d_regression_lr,
    "ves_lr_0_0": ves_lr_0_0,
    "ves_lr_0_1": ves_lr_0_1,
    "ves_lr_0_2": ves_lr_0_2,
    "ves_lr_2_0": ves_lr_2_0,
    "ves_lr_2_1": ves_lr_2_1,
    "ves_lr_2_2": ves_lr_2_2,
    "ves_gamma_0_0": ves_gamma_0_0,
    "ves_gamma_0_1": ves_gamma_0_1,
    "ves_gamma_0_2": ves_gamma_0_2,
    "ves_exp_0": ves_exp_0,
    "ves_exp_1": ves_exp_1,
    "ves_exp_2": ves_exp_2,
}
