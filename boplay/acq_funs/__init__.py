from .ei import expected_improvement
from .random import random_search

ACQ_FUNCS = {
    "expected_improvement": expected_improvement,
    "random_search": random_search,
}
