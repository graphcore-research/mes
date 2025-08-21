from typing import Callable


import numpy as np
import unittest

from boplay.acq_funs import regression_models


fun_names = [f for f in dir(regression_models) if f.startswith("fit_")]


class Test_fit_funs(unittest.TestCase):
    """
    Test the model fit functions.

    This is a sanity check that the model fit functions are working.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(0)
        self.n_rows = 4
        self.n_points = 61

        # deterministc dataset, y-values are above the ReLU trend line
        x_data = np.tile(np.linspace(-3, 3, self.n_points)[None, :], (self.n_rows, 1))
        x_min = 0
        trend = np.clip(x_data, x_min, None)

        y_data = trend + self.rng.uniform(0, 1, trend.shape)
        self.safe_dataset = {
            "x_data": x_data,
            "x_min": x_min,
            "y_data": y_data,
        }

        # stochastic dataset, y-values can be below the ReLU trend line
        # (n_x, n_points)
        y_data = trend + self.rng.uniform(-1, 1, trend.shape)
        self.noisy_dataset = {
            "x_data": x_data,
            "x_min": x_min,
            "y_data": y_data,
        }

    def _validate_fit_fun(self, model_fit_fun: Callable, dataset: dict):
        """
        Validate a model fit function.
        """
        scores = model_fit_fun(**dataset)
        assert scores.shape == (self.n_rows,), f"scores.shape: {scores.shape} != ({self.n_rows},)"
        assert np.isnan(scores).sum() == 0, f"scores: {scores}"
        assert np.isinf(scores).sum() == 0, f"scores: {scores}"

    @classmethod
    def build_fit_fun_tests(cls):
        def make_tests(fun: Callable) -> Callable:
            def test_deterministic(self):
                self._validate_fit_fun(model_fit_fun=fun, dataset=self.safe_dataset)
            
            def test_stochastic(self):
                self._validate_fit_fun(model_fit_fun=fun, dataset=self.noisy_dataset)

            return test_deterministic, test_stochastic


        for fun_name in fun_names:
            fit_fun = getattr(regression_models, fun_name)
            test_det, test_sto = make_tests(fit_fun)
            setattr(cls, f"test_{fun_name}_deterministic", test_det)
            setattr(cls, f"test_{fun_name}_stochastic", test_sto)


Test_fit_funs.build_fit_fun_tests()