import numpy as np

from boplay.benchmark_data import _make_x_grid


def test_grid_x():
    x_grid = _make_x_grid(n=11, x_min=[0], x_max=[1])

    assert (x_grid == np.linspace(0, 1, 11).reshape(11, 1)).all()
