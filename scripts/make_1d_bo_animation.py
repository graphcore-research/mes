from pathlib import Path

from boplay.acq_funs import ves_lr_0_0
from boplay.benchmark_data import make_benchmark_data
from boplay.bo_algorithm import BayesianOptimization
from boplay.kernels import KERNELS
from boplay.plotting.plot_gp_1d import plot_bo_history_1d


if __name__ == "__main__":
    # 1. Create a dataset of synthetic 1D functions
    benchmark = make_benchmark_data(
        n_x=201,
        n_y=2,
        kernel_type="se",
        kernel_params={"len_scale": 3.0, "sigma_f": 1.0},
        x_min=[0],
        x_max=[100],
    )

    # 2. Extract variables needed for the BO method
    kernel = lambda x1, x2: KERNELS[benchmark.kernel](
        x1, x2, **benchmark.kernel_params
    )
    x_grid = benchmark.x
    y_true = benchmark.y[1, :]

    # 3. Instanitiate BO object and run
    bo = BayesianOptimization(
        x_grid=x_grid,
        y_true=y_true,
        kernel=kernel,
        acq_fun=ves_lr_0_0,
        n_init=4,
        n_final=50,
        seed=0,
        verbose=True,
    )
    bo.run()

    # 4. animate the steps of the BO algorithm
    plot_bo_history_1d(
        x_grid=x_grid,
        x_train=bo.x_train,
        y_train=bo.y_train,
        y_true=y_true,
        state_history=bo.state_history,
        animation_gif=Path(__file__).parent / "bo_history.gif",
    )
