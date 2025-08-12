from .ei import expected_improvement
from .random import random_search
from .ves_ramp import ves_exponential_ramp

ACQ_FUNCS = {
    "ves_ramp": ves_exponential_ramp,
    "expected_improvement": expected_improvement,
    "random_search": random_search,
}
