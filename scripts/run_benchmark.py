import numpy as np
from pathlib import Path
from itertools import product
from typing import Callable
import json

from boplay.acq_funs import ACQ_FUNCS
from boplay.bo_algorithm import BayesianOptimization
from boplay.kernels import KERNELS
from boplay.benchmark_data import make_benchmark_data, Benchmark


DATA_DIR = Path(__file__).parent.parent / "data"

# Hyperparameter sweep
acq_types = ACQ_FUNCS.keys()
kernel_types = KERNELS.keys()
len_scales = [1, 5, 10, 25, 50, 100]
n_dims = [1, 2]

# n_x increases with dimensionality. E.g. in 2D n_x = 200 => 200**2 points.
n_total_samples = 201
n_y = 1000

def _make_benchmark_from_hps(kernel, len_scale, n_dim):
    x_min = np.asarray([-1000 for _ in range(n_dim)])
    x_max = np.asarray([1000 for _ in range(n_dim)])
    n_x = int(n_total_samples ** (1 / n_dim))
    return make_benchmark_data(
        n_x=n_x,
        n_y=1000,
        kernel_type=kernel,
        kernel_params={"len_scale": len_scale, "sigma_f": 1.0},
        x_min=x_min,
        x_max=x_max,
    )


def _make_kernel(benchmark: Benchmark):
    kernel_func = KERNELS[benchmark.kernel]
    return lambda x1, x2: kernel_func(x1, x2, **benchmark.kernel_params)


def _one_optimizer_fit(acq_type: str, benchmark: Benchmark):
    x_grid = benchmark.x
    y_true = benchmark.y[1, :]
    acq_func = ACQ_FUNCS[acq_type]
    kernel = _make_kernel(benchmark)
    return BayesianOptimization(
        x_grid=x_grid,
        y_true=y_true,
        kernel=kernel,
        acq_fun=acq_func,
        n_init=4,
        n_final=50,
        seed=0,
    )


results = []

for acq_type, kernel_type, len_scale, n_dim in product(
    acq_types, kernel_types, len_scales, n_dims
):
    print(f"Running: {acq_type}, {kernel_type}, len_scale={len_scale}, n_dim={n_dim}")

    benchmark = _make_benchmark_from_hps(kernel_type, len_scale, n_dim)
    bo = _one_optimizer_fit(acq_type, benchmark)
    bo.run()

    # Save results with metadata
    result = {
        "acq_func": acq_type,
        "kernel_type": kernel_type,
        "len_scale": len_scale,
        "n_dim": n_dim,
        "state_history": bo.state_history,
        "y_max_history": bo.y_max_history,
        "y_true_max": bo.y_true_max,
    }
    results.append(result)

# Save all results to file
output_file = DATA_DIR / "hyperparameter_sweep_results.json"
with open(output_file, "w") as f:
    json.dump(
        results,
        f,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        indent=2,
    )

print(f"Results saved to {output_file}")
