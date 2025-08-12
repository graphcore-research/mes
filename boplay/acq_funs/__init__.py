from .ei import expected_improvement
from .random import random_search
from .ves_ramp import ves_exponential_ramp
from .ves_mc_expo import ves_mc_exponential
from .ves_mc_gamma import ves_mc_gamma


ACQ_FUNCS = {
    "ves_ramp": ves_exponential_ramp,
    "expected_improvement": expected_improvement,
    "random_search": random_search,
    "ves_mc_exponential": ves_mc_exponential,
    "ves_mc_gamma": ves_mc_gamma,
}
