from functools import partial

from boplay.bo_algorithm import BayesianOptimization
from boplay.benchmark_data import make_benchmark_data
from boplay.kernels import KERNELS  
from boplay.acq_funs import random_search


def test_bo_deterministic():
    """
    Test the BO algorithm for a synthetic function with no noise.
    """
    benchmark = make_benchmark_data(
        n_x=50,
        n_y=1,
    )


    kernel = partial(KERNELS[benchmark.kernel], **benchmark.kernel_params)

    bo = BayesianOptimization(
        x_grid=benchmark.x,
        y_true=benchmark.y[0, :],
        kernel=kernel,
        acq_fun=random_search,
        n_init=4,
        n_final=10,
        seed=0,
        y_noise_std=0.0,
    )

    bo.run()


def test_bo_stochastic():
    """
    Test the BO algorithm for a synthetic function with no noise.
    """
    benchmark = make_benchmark_data(
        n_x=50,
        n_y=1,
    )

    kernel = partial(KERNELS[benchmark.kernel], **benchmark.kernel_params)

    bo = BayesianOptimization(
        x_grid=benchmark.x,
        y_true=benchmark.y[0, :],
        kernel=kernel,
        acq_fun=random_search,
        y_noise_std=0.5,
        n_init=4,
        n_final=10,
        seed=0,
    )

    bo.run()
