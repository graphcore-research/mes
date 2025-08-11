import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial

from boplay.acq_funs import ACQ_FUNCS
from boplay.bo_algorithm import BayesianOptimization
from boplay.kernels import KERNELS
from boplay.benchmark_data import make_benchmark_data, Benchmark
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data"

# Hyperparameter sweep
acq_types = ACQ_FUNCS.keys()
kernel_types = KERNELS.keys()
len_scales = [1, 5, 10, 25, 50, 100]
n_dims = [1, 2, 4, 8]

# n_x increases with dimensionality. E.g. in 2D n_x = 200 => 200**2 points.
n_total_samples = 256
n_y = 100  # Run quick
n_init, n_final = 4, 50


def _make_benchmark_from_hps(kernel, len_scale, n_dim):
    x_min = np.asarray([0 for _ in range(n_dim)])
    x_max = np.asarray([100 for _ in range(n_dim)])
    n_x = int(n_total_samples ** (1 / n_dim))
    return make_benchmark_data(
        n_x=n_x,
        n_y=n_y,
        kernel_type=kernel,
        kernel_params={"len_scale": len_scale, "sigma_f": 1.0},
        x_min=x_min,
        x_max=x_max,
    )


def _make_kernel(benchmark: Benchmark):
    kernel_func = KERNELS[benchmark.kernel]
    return lambda x1, x2: kernel_func(x1, x2, **benchmark.kernel_params)


def _fit(acq_type: str, benchmark: Benchmark) -> list[BayesianOptimization]:
    """
    Fit a bayesian for each of the synthetic functions in benchmark.

    Seed 0 enforces reproduceability.
    """
    x_grid = benchmark.x
    n_test_funs, _ = benchmark.y.shape
    bos = []
    for i in range(n_test_funs):
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
        )
        bo.run()
        bos.append(bo)
    return bos


def _process_hyperparams(params):
    """Process a single hyperparameter combination."""
    acq_type, kernel_type, len_scale, n_dim = params
    benchmark = _make_benchmark_from_hps(kernel_type, len_scale, n_dim)
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
        }
        rows.append(row)
    return rows


if __name__ == "__main__":
    param_combinations = list(product(acq_types, kernel_types, len_scales, n_dims))
    n_sweeps = len(param_combinations)
    with Pool(processes=cpu_count()) as pool:
        all_results = list(
            tqdm(
                pool.imap(_process_hyperparams, param_combinations),
                total=n_sweeps,
            )
        )

    # Flatten the results
    results = []
    for result_batch in all_results:
        results.extend(result_batch)

    results_df = pd.DataFrame(results)
    results_df.to_json(DATA_DIR / "benchmark_df.json")
