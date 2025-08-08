import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from boplay.kernels import KERNELS


@dataclass
class Benchmark:
    """ A benchmark data set for BO methods """ 
    x : np.ndarray
    y : np.ndarray
    n_x: int
    n_y: int
    kernel: str
    kernel_params: dict
    x_min: np.ndarray
    x_max: np.ndarray

    def save(self, path: str | Path) -> None:
        """Save the dataclass to a JSON file."""
        data = asdict(self)
        data["x"] = data["x"].tolist()
        data["y"] = data["y"].tolist()
        data["x_min"] = np.asarray(data["x_min"]).tolist()
        data["x_max"] = np.asarray(data["x_max"]).tolist()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Benchmark":
        """Load from a JSON file into a Config instance."""
        with open(path) as f:
            data = json.load(f)
        
        data["x"] = np.asarray(data["x"])
        data["y"] = np.asarray(data["y"])
        data["x_min"] = np.asarray(data["x_min"])
        data["x_max"] = np.asarray(data["x_max"])

        return cls(**data)


def _make_x_grid(*, n: int, x_min: list[float], x_max: list[float]) -> np.ndarray:
    """ Make a full dense grid of points in the given ranges """
    
    x_min = np.asarray(x_min).reshape(-1)
    x_max = np.asarray(x_max).reshape(-1)

    assert len(x_min) == len(x_max), "x_min and x_max must have the same length"

    # list of 1-dim grids with n points, one for each axis
    x_grid_axis = [np.linspace(lo, hi, n) for lo, hi in zip(x_min, x_max)]

    # list if d-dim grids with n**d points each 
    x_grid = np.meshgrid(x_grid_axis)

    # flatten into list of n**d column vectors
    x_grid = [x_grid_i.reshape(-1, 1) for x_grid_i in x_grid]

    # stack column vectors into a big x matrix
    x_grid = np.hstack(x_grid)

    assert x_grid.shape[1] == len(x_min), "x_grid must have the same number of columns as x_min"
    assert x_grid.shape[0] == n**len(x_min), "x_grid must have the same number of rows as n**len(x_min)"

    return x_grid


def _make_y_data(
    *,
    x_grid: np.ndarray,
    n_y: int,
    kernel: callable,
) -> tuple[np.ndarray, np.ndarray]:
    """ Make a random y vectors from a kernel matrix """
    assert len(x_grid.shape) == 2, "x_grid must be a matrix"

    mean_vector = np.zeros(x_grid.shape[0])
    cov_matrix = kernel(x_grid, x_grid)
    y_grid = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_y).T

    assert y_grid.shape[0] == x_grid.shape[0], "y_grid must have the same number of rows as x_grid"
    assert y_grid.shape[1] == n_y, "y_grid must have n_y columns"

    return y_grid


def make_benchmark_data(
    *,
    n_x: int=100,
    n_y: int=1000,
    kernel_type: str="se",
    x_min: np.ndarray=[0],
    x_max: np.ndarray=[100],
    kernel_params: dict = {"len_scale": 10.0, "sigma_f": 1.0},
    seed:int = 0,
) -> Benchmark:
    """ Make a benchmark data set for BO methods """
    np.random.seed(seed)

    x_grid = _make_x_grid(n=n_x, x_min=x_min, x_max=x_max)

    # get the kernel function
    kernel_func = KERNELS[kernel_type]
    kernel = lambda x1, x2: kernel_func(x1, x2, **kernel_params)

    # sample y-values over the grid
    y_grid = _make_y_data(x_grid=x_grid, n_y=n_y, kernel=kernel)

    return Benchmark(
        x=x_grid,
        y=y_grid,
        n_x=n_x,
        n_y=n_y,
        kernel=kernel_type,
        kernel_params=kernel_params,
        x_min=x_min,
        x_max=x_max
    )
