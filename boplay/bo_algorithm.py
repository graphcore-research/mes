import numpy as np
import time

from boplay.gp import GaussianProcess


class BayesianOptimization:
    """
    Bayesian optimization algorithm.

    provided some ground truth data, the algorithm will sequentially
    select points to evaluate and update a model.

    Args:
        x_grid: np.ndarray, shape (n_x, x_dim)
        y_true: np.ndarray, shape (n_x, 1)
        kernel: callable, kernel function
        acq_fun: callable, acquisition function
        n_init: int, number of initial points
        n_final: int, number of final points
        seed: int, random seed

    Returns:
        y_max_history: list, list of (n_train, y_max) pairs
    """

    def __init__(
        self,
        *,
        x_grid: np.ndarray,
        y_true: np.ndarray,
        kernel: callable,
        acq_fun: callable,
        y_noise_std: float = 0.0,
        n_init:int=4,
        n_final:int=100,
        seed:int=0,
        verbose:bool=False,
    ) -> None:
        # benchmark ground truth data, x_grid is known to the algorithm,
        # y_true is not known to the algorithm and is collected one value
        # at a time by the algorithm.
        self.x_grid = np.asarray(x_grid)
        self.y_true = np.asarray(y_true).reshape(-1, 1)
        self.y_true_max = float(np.max(self.y_true))
        self.y_noise_std = y_noise_std


        # algorithm parameters (constants)
        self.kernel = kernel
        self.acq_fun = acq_fun
        self.n_init = n_init
        self.n_final = n_final
        self.seed = seed
        self.verbose = verbose

        # pre-generate a stream of noise values
        np.random.seed(self.seed)
        self.y_noise_values = np.random.normal(size=(n_final, 1)) * self.y_noise_std

        # algorithm state (mutable)
        self.x_train = None
        self.y_train = None
        self.idx_train = None
        self.model = None
        self.y_best = None
        self.acq_fun_vals = None

        self.y_max_history = []

        self.state_history = []

    def _update_model(self) -> None:
        self.model = GaussianProcess(
            x_train=self.x_train,
            y_train=self.y_train,
            kernel=self.kernel,
            y_noise_std=self.y_noise_std,
        )
        start_time = time.time()
        self.y_mean, self.y_cov = self.model.predict(x_test=self.x_grid)
        self.predict_time = time.time() - start_time
        self.y_best = np.max(self.y_train)

    def _select_next_point(self) -> int:
        # compute mean, cov, acq fun and save as attributes
        start_time = time.time()
        self.acq_fun_vals = self.acq_fun(
            x_grid=self.x_grid,
            y_mean=self.y_mean,
            y_cov=self.y_cov,
            y_noise_std=self.y_noise_std,
            y_best=self.y_best,
            idx_train= [] if self.y_noise_std > 1e-3 else self.idx_train,
        )
        self.acq_fun_time = time.time() - start_time

        # if there is noise in the y-values, we dont need to mask off old points, we 
        # can re-visit old points.
        if self.y_noise_std > 1e-3:
            return np.argmax(self.acq_fun_vals)
        
        # set the acq fun vals for the points we have already evaluated to -inf
        acq_fun_vals_masked = self.acq_fun_vals.copy()
        for idx in self.idx_train:
            acq_fun_vals_masked[idx] = -np.inf

        # select the point with the highest acq fun value
        return np.argmax(acq_fun_vals_masked)

    def _get_initial_points(self) -> list[int]:
        """
        Get a list of indices of the initial points.
        """
        return np.random.choice(self.x_grid.shape[0], size=self.n_init, replace=False)

    def _update_history(self) -> None:
        sample_funcs = self.model.sample_posterior(
            x_test=self.x_grid,
            n_samples=1000,
        )
        y_max_var = np.var(np.max(sample_funcs, axis=1))
        y_max_diff = self.y_true_max - np.max(self.y_train)

        idx_recomend_mean = np.argmax(self.y_mean)
        idx_recomend_max = np.argmax(self.y_train)
        y_rec_diff_mean = self.y_true_max - self.y_true[idx_recomend_mean]
        y_rec_diff_max = self.y_true_max - self.y_true[idx_recomend_max]
        
        self.state_history.append(
            {
                "n_train": len(self.y_train),
                "y_mean": self.y_mean,
                "y_sd": np.sqrt(np.clip(np.diag(self.y_cov), 1e-6, None)),
                "acq_fun_vals": self.acq_fun_vals,
                "acq_fun_time": self.acq_fun_time,
                "predict_time": self.predict_time,
                "y_max_var": y_max_var,
                "y_max_diff": y_max_diff,
                "y_rec_diff_mean": y_rec_diff_mean,
                "y_rec_diff_max": y_rec_diff_max,
            }
        )
        # keep these for easy access/plotting convergence curves
        self.y_max_history.append(
            [
                len(self.y_train),
                float(np.max(self.y_train)),
                float(y_max_diff),
                float(y_rec_diff_mean),
                float(y_rec_diff_max),
                float(y_max_var),
            ]
        )

    def run(self) -> None:
        # initialize the algorithm with a few random points
        idx_init = self._get_initial_points().tolist()
        self.idx_train = idx_init
        self.x_train = self.x_grid[self.idx_train]
        self.y_train = self.y_true[self.idx_train] + self.y_noise_values[:len(self.idx_train)]
        self._update_model()

        for _ in range(self.n_init, self.n_final):
            # fit a model to the points we have so far

            # select the next point
            idx_new = self._select_next_point()
            self._update_history()

            # update the algorithm training data
            self.idx_train.append(idx_new)

            x_new = self.x_grid[idx_new]
            y_new = self.y_true[idx_new] + self.y_noise_values[len(self.idx_train)-1]
            self.x_train = np.vstack((self.x_train, x_new))
            self.y_train = np.vstack((self.y_train, y_new))
            self._update_model()

            n, y_max = len(self.y_train), np.max(self.y_train)
            y_max_diff = self.y_true_max - y_max

            acq_fun_time = round(1e3 * self.acq_fun_time, 5)
            predict_time = round(1e3 * self.predict_time, 5)

            if self.verbose:
                print((
                    f"Iteration {n}, "
                    f"y_max: {y_max}, "
                    f"y_max_diff: {y_max_diff}, "
                    f"acq_fun_time: {acq_fun_time} ms, "
                    f"predict_time: {predict_time} ms, "
                )[:-2])

        self._update_history()