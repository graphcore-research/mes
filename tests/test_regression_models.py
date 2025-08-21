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
        self.x_data = self.rng.normal(0, 1, 100).reshape(4, 25)
        self.y_data = np.sin(self.x_data) + self.rng.normal(0, 1, 100).reshape(4, 25)

    def _validate_fit_fun(self, model_fit_fun):
        """
        Validate a model fit function.
        """
        scores = model_fit_fun(x_data=self.x_data, y_data=self.y_data, x_min=-1000)
        assert scores.shape == (self.x_data.shape[0],), f"scores.shape: {scores.shape} != ({self.x_data.shape[0]},)"
        assert np.isnan(scores).sum() == 0, f"scores: {scores}"
        assert np.isinf(scores).sum() == 0, f"scores: {scores}"

    @classmethod
    def build_fit_fun_tests(cls):
        def make_test(fun):
            def test(self):
                self._validate_fit_fun(model_fit_fun=fun)
            return test

        for fun_name in fun_names:
            fun = getattr(regression_models, fun_name)
            setattr(cls, f"test_{fun_name}", make_test(fun))


Test_fit_funs.build_fit_fun_tests()