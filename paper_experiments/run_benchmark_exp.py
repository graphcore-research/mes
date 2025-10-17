from pathlib import Path
from functools import partial
from datetime import datetime
import subprocess as sp

from multiprocessing import cpu_count
from itertools import product

from tqdm.contrib.concurrent import process_map

import pandas as pd
# from tqdm import tqdm

from boplay.benchmark_data import Benchmark
from boplay.acq_funs import ACQ_FUNCS
from boplay.kernels import KERNELS
from boplay.bo_algorithm import BayesianOptimization

from make_datasets import DATA_DIR


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_benchmark(
    *,
    benchmark: Benchmark,
    y_idx: int,
    acq_fun_name: str,
    filename: Path,
    n_init: int = 4,
    n_final: int = 100,
    y_noise_std: float = 0.0,
    seed: int = 0,
) -> None:
    """
    Run a benchmark experiment and save the results to a file.
    """
    # make the kernel
    kernel = partial(KERNELS[benchmark.kernel], **benchmark.kernel_params)

    # make the acq fun
    acq_fun = ACQ_FUNCS[acq_fun_name]

    # make the bo
    bo = BayesianOptimization(
        x_grid=benchmark.x,
        y_true=benchmark.y[y_idx, :],
        kernel=kernel,
        acq_fun=acq_fun,
        n_init=n_init,
        n_final=n_final,
        seed=seed,
        y_noise_std=y_noise_std,
    )

    # run the bo
    bo.run()

    # unpack and save the results
    (
        steps,
        y_max_history,
        y_max_diff,
        y_rec_diff_mean,
        y_rec_diff_max,
        y_max_var,
    ) = zip(*bo.y_max_history)
    final_y_max = y_max_history[-1]
    row = {
        "acq_func": acq_fun_name,
        "kernel_type": benchmark.kernel,
        "len_scale": benchmark.kernel_params["len_scale"],
        "n_dim": benchmark.x.shape[1],
        "run_id": y_idx,
        "y_true_max": benchmark.y[y_idx, :].max(),  # (B,)
        "final_y_max": final_y_max,  # (B,)
        "y_max_history": y_max_history,  # (B, T)
        "y_max_diff": y_max_diff,  # (B, T)
        "y_rec_diff_mean": y_rec_diff_mean,  # (B, T)
        "y_rec_diff_max": y_rec_diff_max,  # (B, T)
        "y_max_var": y_max_var,  # (B, T)
        "steps": steps,  # (B, T)
        "y_noise_std": y_noise_std,
    }

    pd.DataFrame(row).to_json(filename, orient="records")
    # print(f"Saved {filename}")
    return filename


def bench_wrapper(params) -> Path:
    """
    Run a benchmark experiment.
    """
    # unpack params, load the benchmark dataset
    y_idx, benchmark_file, y_noise_std, acq_fun, results_dir = params
    benchmark = Benchmark.load(DATA_DIR / benchmark_file)

    # Compose a result file name for this single run
    kernel_type = benchmark.kernel.replace("/", "_")
    filename = (
        f"{kernel_type}_{benchmark.kernel_params['len_scale']}_"
        f"{benchmark.x.shape[1]}_{acq_fun}_{y_idx}_{y_noise_std}.json"
    )
    filename = results_dir / filename

    if filename.exists():
        # Skip if the file already exists
        print(f"Skipping {filename} because it already exists")
        return filename

    else:
        # Run the benchmark and save the results
        print(f"Running {filename}")
        return run_benchmark(
            benchmark=benchmark,
            y_idx=y_idx,
            acq_fun_name=acq_fun,
            y_noise_std=y_noise_std,
            filename=filename,
        )


def make_results_dirname():
    """
    Make a results directory for the current run.
    """
    run_id = len(list(RESULTS_DIR.glob("*")))
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    commit_hash = (
        sp.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )
    results_dir = RESULTS_DIR / f"{run_id}_{time}_{commit_hash}"
    return results_dir


if __name__ == "__main__":
    # results_dir = RESULTS_DIR / "0"
    results_dir = make_results_dirname()

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    # PARAMETER SWEEPS
    benchmark_files = ["mat52_2d_short.json"]
    acq_funs = [
        # "probability_of_improvement",
        # "expected_improvement",
        # "random_search",
        # "ves_gamma_0_0",
        # "ves_gamma_0_1",
        # "ves_gamma_0_2",
        # "ves_mc_gamma",
        # "ves_exp_0",
        # "ves_exp_1",
        # "ves_exp_2",
        # "ves_mc_exponential",
        "ves_lr_0_0",
        "ves_lr_0_1",
        "ves_lr_0_2",
        "ves_lr_2_0",
        "ves_lr_2_1",
        "ves_lr_2_2",
        # "ves_mc_gaussian",
    ]
    y_noise_std_levels = [0.0, 0.3]
    n_y = 100

    param_combinations = list(
        product(
            range(n_y),
            benchmark_files,
            y_noise_std_levels,
            acq_funs,
            [results_dir],
        )
    )

    print(f"Running {len(param_combinations)} experiments")

    if 1 == 11:
        for params in param_combinations:
            bench_wrapper(params)

    elif 1 == 1:
        _ = process_map(
            bench_wrapper,
            param_combinations,
            max_workers=cpu_count(),
        )
