from pathlib import Path

from boplay.acq_funs.random import random_search
from boplay.benchmark_data import Benchmark
from boplay.bo_algorithm import BayesianOptimization
from boplay.kernels import KERNELS
from boplay.plotting.plot_gp_1d import plot_bo_history_1d


DATA_DIR = Path(__file__).parent.parent / "data"


benchmark = Benchmark.load(DATA_DIR / "1d_dataset.json")

x_grid = benchmark.x

kernel = lambda x1, x2: KERNELS[benchmark.kernel_type](
    x1, x2, **benchmark.kernel_params
)


y_true = benchmark.y[1, :]
bo = BayesianOptimization(
    x_grid=x_grid,
    y_true=y_true,
    kernel=kernel,
    acq_fun=random_search,
    n_init=4,
    n_final=50,
    seed=0,
)

bo.run()

plot_bo_history_1d(
    x_grid=x_grid,
    x_train=bo.x_train,
    y_train=bo.y_train,
    y_true=y_true,
    state_history=bo.state_history,
    animation_gif=Path(__file__).parent / "bo_history.gif",
)
