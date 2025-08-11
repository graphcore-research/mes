import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from boplay.acq_funs import ACQ_FUNCS
from boplay.bo_algorithm import BayesianOptimization
from boplay.kernels import KERNELS
from boplay.benchmark_data import make_benchmark_data, Benchmark
from tqdm import trange, tqdm


DATA_DIR = Path(__file__).parent.parent / "data"

# Hyperparameter sweep
acq_types = ACQ_FUNCS.keys()
kernel_types = KERNELS.keys()
len_scales = [1, 5, 10, 25, 50, 100]
n_dims = [1, 2, 4, 8]

n_sweeps = len(acq_types) * len(kernel_types) * len(len_scales) * len(n_dims)
print(f"Running {n_sweeps} sweeps")

# n_x increases with dimensionality. E.g. in 2D n_x = 200 => 200**2 points.
n_total_samples = 256
n_y = 100  # Run quick


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
    x_grid = benchmark.x
    n_test_funs, _ = benchmark.y.shape
    bos = []
    for i in trange(n_test_funs, leave=False):
        y_true = benchmark.y[i, :]
        acq_func = ACQ_FUNCS[acq_type]
        kernel = _make_kernel(benchmark)
        bo = BayesianOptimization(
            x_grid=x_grid,
            y_true=y_true,
            kernel=kernel,
            acq_fun=acq_func,
            n_init=4,
            n_final=50,
            seed=0,
        )
        bo.run()
        bos.append(bo)
    return bos


# Keep original pickle format for full reproducibility
pickle_results = {
    "meta": {"n_total_samples": n_total_samples, "n_y": n_y},
    "data": {},
    "y_max_history": {},
}

# Also collect structured data for pandas DataFrame
pd_results = []

for acq_type, kernel_type, len_scale, n_dim in tqdm(
    product(acq_types, kernel_types, len_scales, n_dims), total=n_sweeps
):
    benchmark = _make_benchmark_from_hps(kernel_type, len_scale, n_dim)
    bos = _fit(acq_type, benchmark)

    # Store in original format for pickle
    name = f"{acq_type}-{kernel_type}-{n_dim}D-{len_scale}_len_scale"
    pickle_data = {"name": name, "bos": bos, "benchmark": benchmark}
    pickle_results["data"][name] = pickle_data

    # Create one row per test function (bo run)
    y_true_max = np.max(benchmark.y, axis=1)
    for i, bo in enumerate(bos):
        steps, y_max_history = zip(*bo.y_max_history)
        final_y_max = y_max_history[-1]

        pd_data = {
            "acq_func": acq_type,
            "kernel_type": kernel_type,
            "len_scale": len_scale,
            "n_dim": n_dim,
            "run_id": i,
            "y_true_max": y_true_max[i],
            "final_y_max": final_y_max,
            "y_max_history": y_max_history,
            "steps": steps,
        }
        pd_results.append(pd_data)


results_df = pd.DataFrame(pd_results)
results_df.to_json(DATA_DIR / "benchmark_df.json")

with open(DATA_DIR / "benchmark_data.pickle", "wb") as f:
    pickle.dump(pd_results, f)
