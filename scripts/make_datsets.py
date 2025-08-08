"""
Make benchmark datasets. These are simply GP geenrated functions.

The datasets are saved as JSON in the REPO/data/ directory. The datasets are saved as a
Benchmark dataclass with the following fields:
    - x: np.ndarray, shape (n_x, x_dim)  # x-locations
    - y: np.ndarray, shape (n_y, n_x)    # each row is one test fun, a y-vector for the n_x locs
    - n_x: int                           # number of x-locations
    - n_y: int                           # number of y-vectors to generated   
    - kernel_type: str                   # generating kernel type, "se" for squared exponential
    - kernel_params: dict                # generating kernel parameters
    - x_min: np.ndarray, shape (x_dim,)  # min of each dimension
    - x_max: np.ndarray, shape (x_dim,)  # max of each dimension
"""

from pathlib import Path

from boplay.benchmark_data import make_benchmark_data, Benchmark

import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # make 1D dataset
    benchmark_new = make_benchmark_data(
        n_x=201,
        n_y=1000,
        kernel_type="se",
        kernel_params={"len_scale": 3.0, "sigma_f": 1.0},
        x_min=[0],
        x_max=[100],
    )

    benchmark_new.save(DATA_DIR / "1d_dataset.json")
    benchmark = Benchmark.load(DATA_DIR / "1d_dataset.json")

    # plot the dataset
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(10):
        ax.plot(benchmark.x, benchmark.y[i, :], color="gray", alpha=0.5)

    fig.savefig(DATA_DIR / "1d_dataset.png")

