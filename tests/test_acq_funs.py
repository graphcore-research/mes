from functools import partial
from typing import Callable
import unittest

import numpy as np

from boplay.acq_funs import ACQ_FUNCS
from boplay.gp import GaussianProcess
from boplay.kernels import se_kernel


class TestAcqFuns(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # make a grid of x-values
        x_grid = np.linspace(0, 100, 101).reshape(-1, 1)
        self.n_x = x_grid.shape[0]

        # make a train set of x-values and y-values
        idx_train = np.array([0, 50, 75])
        x_train = x_grid[idx_train, :]
        y_train = np.asarray([0.5, 0.0, 1]).reshape(-1, 1)

        # make a kernel
        kernel = partial(se_kernel, len_scale=10.0, sigma_f=1.0)

        # make a Gaussian process
        gp = GaussianProcess(x_train=x_train, y_train=y_train, kernel=kernel)

        # get the mean and covariance of the Gaussian process
        y_mean, y_cov = gp.predict(x_test=x_grid)

        # get the best observed value
        y_best = max(y_train)
    
        self.x_grid = x_grid
        self.y_mean = y_mean
        self.y_cov = y_cov
        self.y_best = y_best
        self.idx_train = idx_train

    def _validate_acq_funs(self, acq_fun_name: str, acq_fun: Callable) -> None:
        """
        Execute and validate the output of a given acquisition function.

        This simply serves as a sanity check that the acquisition function/API is working.
        """
        acq_fun_vals = acq_fun(
            x_grid=self.x_grid,
            y_mean=self.y_mean,
            y_cov=self.y_cov,
            y_best=self.y_best,
            idx_train=self.idx_train,
        )

        assert acq_fun_vals.shape == (self.n_x,), (
            f"{acq_fun_name}: shape mismatch {acq_fun_vals.shape} != ({self.n_x},)"
        )
        assert np.isnan(acq_fun_vals).sum() == 0, f"{acq_fun_name}: nans in acq fun values"
        assert np.isinf(acq_fun_vals).sum() == 0, f"{acq_fun_name}: infs in acq fun values"
    
    @classmethod
    def build_acq_fun_tests(cls):
        """
        Construct a test function for each acquisition function.
        """
        def make_test(name, fun):
            def test(self):
                self._validate_acq_funs(acq_fun_name=name, acq_fun=fun)
            return test

        for name, fun in ACQ_FUNCS.items():
            setattr(cls, f"test_{name}", make_test(name, fun))


TestAcqFuns.build_acq_fun_tests()
