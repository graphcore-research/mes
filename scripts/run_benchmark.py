import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product, chain
from multiprocessing import Pool, cpu_count, RLock, current_process
from functools import partial

from boplay.acq_funs import ACQ_FUNCS
from boplay.kernels import KERNELS
from boplay.bo_algorithm import BayesianOptimization
from boplay.benchmark_data import make_benchmark_data, Benchmark
from tqdm import tqdm, trange


DATA_DIR = Path(__file__).parent.parent / "data"

# Hyperparameter sweep
wds = np.logspace(-2, 1, num=10)  # 1e-2 -> 10
lrs = np.logspace(-4, -0.5, num=10)  # 1e-4 -> 5e-2
acq_fun_params_list = [
    dict(lr=lr, wd=wd, max_iters=max_iters)
    for lr, wd, max_iters in product(lrs, wds, [50])
]  # default params
acq_types = [
    "ves_mc_gamma",
    "ves_gamma",
    "expected_improvement",
    "random_search",
    "ves_ramp",
]
kernel_types = ["matern-3/2", "matern-5/2"]
len_scales = [10, 25]
n_dims = [2, 4]

# n_x increases with dimensionality. E.g. in 2D n_x = 200 => 200**2 points.
n_total_samples = 100
n_y = 25  # Run quick
n_init, n_final = 4, 25


def _make_benchmark_from_hps(kernel, len_scale, n_dim, acq_fun_params):
    x_min = np.asarray([0 for _ in range(n_dim)])
    x_max = np.asarray([100 for _ in range(n_dim)])
    n_x = int(n_total_samples ** (1 / n_dim))
    return make_benchmark_data(
        n_x=n_x,
        n_y=n_y,
        kernel_type=kernel,
        kernel_params={"len_scale": len_scale, "sigma_f": 1.0},
        acq_fun_params=acq_fun_params,
        x_min=x_min,
        x_max=x_max,
    )


def _make_kernel(benchmark: Benchmark):
    kernel_func = KERNELS[benchmark.kernel]
    return partial(kernel_func, **benchmark.kernel_params)


def _fit(acq_type: str, benchmark: Benchmark) -> list[BayesianOptimization]:
    """
    Fit a bayesian for each of the synthetic functions in benchmark.

    Seed 0 enforces reproduceability.
    """
    wid = current_process()._identity[0]
    x_grid = benchmark.x
    acq_fun_params = benchmark.acq_fun_params
    n_test_funs, _ = benchmark.y.shape
    bos = []
    for i in trange(n_test_funs, leave=False, position=wid, desc=f"job: {wid}"):
        y_true = benchmark.y[i, :]
        acq_func = ACQ_FUNCS[acq_type]
        kernel = _make_kernel(benchmark)
        bo = BayesianOptimization(
            x_grid=x_grid,
            y_true=y_true,
            kernel=kernel,
            acq_fun=acq_func,
            n_init=n_init,
            n_final=n_final,
            seed=0,
            acq_fun_params=acq_fun_params,
        )
        bo.run()
        bos.append(bo)
    return bos


def _process_hyperparams(params):
    """Process a single hyperparameter combination."""
    acq_type, kernel_type, len_scale, n_dim, acq_fun_params = params
    benchmark = _make_benchmark_from_hps(kernel_type, len_scale, n_dim, acq_fun_params)
    bos = _fit(acq_type, benchmark)

    # Each row of dataframe is a single BO algorithm
    # We use batch (B) as the number of synthetic functions (n_y)
    # Iteration size (T) = n_final - n_init
    # y_true_max: Ground truth y_max, best possible point on the curve
    # final_y_max: Result from final convergence. (y_max_history[:, -1])
    # y_max_history: Full dataset of convergence over T iterations steps for all functions.
    # steps: range(n_init, n_final)
    y_true_max = np.max(benchmark.y, axis=1)
    rows = []
    for i, bo in enumerate(bos):
        steps, y_max_history = zip(*bo.y_max_history)
        final_y_max = y_max_history[-1]
        row = {
            "acq_func": acq_type,
            "kernel_type": kernel_type,
            "len_scale": len_scale,
            "n_dim": n_dim,
            "run_id": i,
            "y_true_max": y_true_max[i],  # (B,)
            "final_y_max": final_y_max,  # (B,)
            "y_max_history": y_max_history,  # (B, T)
            "steps": steps,  # (B, T)
            **acq_fun_params,  # lr, wd
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    param_combinations = list(
        product(acq_types, kernel_types, len_scales, n_dims, acq_fun_params_list)
    )
    n_sweeps = len(param_combinations)
    lock = RLock()
    with Pool(
        processes=cpu_count(), initializer=tqdm.set_lock, initargs=(lock,)
    ) as pool:
        # Run sweep in parallel, futures are evaluated lazily.
        # We don't care what order the futures return in.
        futures = tqdm(
            pool.imap_unordered(_process_hyperparams, param_combinations),
            total=n_sweeps,
            desc="Total runs",
        )
        results = list(chain.from_iterable(futures))  # flatten & evaluate

    # Run no multiprocessing
    # futures = tqdm(
    #     map(_process_hyperparams, param_combinations),
    #     total=n_sweeps,
    # )
    # results = list(chain.from_iterable(futures))  # flatten & evaluate

    results_df = pd.DataFrame(results)
    results_df.to_json(DATA_DIR / "benchmark_df.json")
