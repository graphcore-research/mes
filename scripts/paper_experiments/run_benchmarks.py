import subprocess as sp
from datetime import datetime
from functools import partial
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from tqdm.contrib.concurrent import process_map

import pandas as pd

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

    Parameters
    ----------
    benchmark : Benchmark
        The benchmark data set to run the experiment on.
    y_idx : int
        The index of the y-value to run the experiment on.
    acq_fun_name : str
        The name of the acquisition function to use.
    filename : Path
        The path to save the results to.
    n_init : int
        The number of initial points to sample.
    n_final : int
        The number of final points to sample.
    y_noise_std : float
        The noise standard deviation to add to the function values.
    seed : int
        The random seed to use for the experiment.

    Returns
    -------
    None
        The results are saved to the file specified by the filename parameter.
    """
    # 1. make the kernel function and the acquisition function
    kernel = partial(KERNELS[benchmark.kernel], **benchmark.kernel_params)
    acq_fun = ACQ_FUNCS[acq_fun_name]

    # 2. instantiate the BO object and run BO
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
    bo.run()

    # 3. unpack and save the results
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
        "y_true_max": benchmark.y[y_idx, :].max(),  # (1,)
        "final_y_max": final_y_max,  # (1,)
        "y_max_history": y_max_history,  # (T,)
        "y_max_diff": y_max_diff,  # (T,)
        "y_rec_diff_mean": y_rec_diff_mean,  # (T,)
        "y_rec_diff_max": y_rec_diff_max,  # (T,)
        "y_max_var": y_max_var,  # (T,)
        "steps": steps,  # (T,)
        "y_noise_std": y_noise_std,
    }

    pd.DataFrame(row).to_json(filename, orient="records")
    print(f"Saved {filename}")


def bench_wrapper(params: list[Any]) -> Path:
    """
    Take the list of params for one run, unpack them, load the necessary
    files and call run_benchmark() and return the results file path.

    Parameters
    ----------
    params : list
        A list of parameters for one run.

    Returns
    -------
    Path
        The path to the result single JSON file with results for this one run.
    """
    # 1. unpack params, load the benchmark dataset
    y_idx, benchmark_file, y_noise_std, acq_fun, results_dir = params

    # 2. load the pregenerated benchmark data
    benchmark = Benchmark.load(DATA_DIR / benchmark_file)

    # 3. Compose a result file name for this single run
    kernel_type = benchmark.kernel.replace("/", "_")
    filename = (
        f"{kernel_type}_{benchmark.kernel_params['len_scale']}_"
        f"{benchmark.x.shape[1]}_{acq_fun}_{y_idx}_{y_noise_std}.json"
    )
    filename = results_dir / filename

    # 4. run the benchmark and save results (or skip if already saved)
    if filename.exists():
        # 4.1 Skip if the file already exists (e.g. restarting a batch of experiments)
        print(f"Skipping {filename} because it already exists")
        return

    # 4.2 Run the benchmark and save the results to a new file
    print(f"Running {filename}")
    run_benchmark(
        benchmark=benchmark,
        y_idx=y_idx,
        acq_fun_name=acq_fun,
        y_noise_std=y_noise_std,
        filename=filename,
    )


def make_results_dir() -> Path:
    """
    Make a uniquely named results directory for the current run using
    the number of prior runs + 1, current time and the git hash. E.g.

    {RESULTS_DIR}/1234_2025-10-20_12-34-56_txvf65.json
    """
    run_id = len(list(RESULTS_DIR.glob("*")))
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    commit_hash = (
        sp.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )
    results_dir = RESULTS_DIR / f"{run_id}_{time}_{commit_hash}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    # 1. make a new results directory to store all the JSON files
    # NOTE: replace this folder with a prior results folder to re-run all
    # the missing results/restart an old batch of experiments.
    results_dir = make_results_dir()

    # 2. define the varying parameters for the experiments to sweep over
    benchmark_files = ["mat52_2d_short.json"]
    acq_funs = [
        # "probability_of_improvement",
        "expected_improvement",
        "random_search",
        "ves_gamma_0_0",
        # "ves_gamma_0_1",
        # "ves_gamma_0_2",
        # "ves_mc_gamma",
        "ves_exp_0",
        # "ves_exp_1",
        # "ves_exp_2",
        # "ves_mc_exponential",
        "ves_lr_0_0",
        # "ves_lr_0_1",
        # "ves_lr_0_2",
        # "ves_lr_2_0",
        # "ves_lr_2_1",
        # "ves_lr_2_2",
        # "ves_mc_gaussian",
    ]
    y_noise_std_levels = [0.0, 0.3]
    test_fun_indices = range(100)

    # 3. make a list of all the individual parameter combinations to run
    param_combinations = list(
        product(
            test_fun_indices,
            benchmark_files,
            y_noise_std_levels,
            acq_funs,
            [results_dir],
        )
    )
    print(f"Running {len(param_combinations)} experiments")

    # 4. run all the individual parameter settings in parallel
    _ = process_map(
        bench_wrapper,
        param_combinations,
        max_workers=cpu_count(),
    )
