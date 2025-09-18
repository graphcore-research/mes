import numpy as np


class GaussianProcess:
    """1D Gaussian Process regression with Cholesky inference.

    Attributes:
        X_train: 1D numpy array of training inputs.
        y_train: 1D numpy array of training targets.
        kernel: Callable kernel function k(x, x').
        noise: Scalar noise level added to the diagonal.
    """

    def __init__(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        kernel: callable,
        y_noise_std: float = 1e-4,
    ) -> None:
        """Initializes the GP with training data and a kernel.

        Args:
            X_train: d-dimensional numpy array of training inputs.
            y_train: 1-dimensional numpy array of training targets.
            kernel: A callable kernel function k(x, x').
            y_noise_std: Gaussian noise level (default is 1e-6).
        """
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train).reshape(-1, 1)

        assert len(x_train.shape) == 2, "x must be a matrix"

        assert x_train.shape[0] == y_train.shape[0], "x and y must have the same number of rows"

        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel
        self.y_noise_std = y_noise_std
        self.x_dim = x_train.shape[1]
        self._fit()

    def _fit(self):
        """Computes the Cholesky decomposition of the kernel matrix."""
        k = self.kernel(self.x_train, self.x_train)
        k += self.y_noise_std**2 * np.eye(len(self.x_train))
        self.L = np.linalg.cholesky(k)
        self.alpha = np.linalg.solve(
            self.L.T, np.linalg.solve(self.L, self.y_train)
        )

    def predict(self, *, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts the mean and variance at test inputs.

        Args:
            X_test: 1D numpy array of test inputs.

        Returns:
            Tuple of (mean, variance) arrays at the test inputs.
        """

        x_test = x_test.reshape(-1, self.x_dim)
        k_s = self.kernel(self.x_train, x_test)
        k_ss = self.kernel(x_test, x_test)

        mean = k_s.T @ self.alpha
        v = np.linalg.solve(self.L, k_s)
        cov = k_ss - v.T @ v

        return mean, cov

    def sample_posterior(
        self,
        *,
        x_test: np.ndarray,
        n_samples: int = 1,
        seed: int = 0,
    ) -> np.ndarray:
        """Draws samples from the posterior at test inputs.

        Args:
            x_test: (n_test, x_dim) numpy array of test inputs.
            n_samples: Number of posterior samples to draw.

        Returns:
            Samples: A (n_samples, n_test) array of function samples.
        """
        np.random.seed(seed)
        mean, cov = self.predict(x_test=x_test)
        
        cov += 1e-8 * np.eye(len(x_test))  # stability
        samples = np.random.multivariate_normal(
            mean.reshape(-1), cov, size=n_samples
        )
        return samples
    
    def sample_prior(
        self,
        *,
        x_test: np.ndarray,
        n_samples: int = 1,
        seed: int = 0,
    ) -> np.ndarray:
        """Draws samples from the prior at test inputs.

        Args:
            x_test: (n_test, x_dim) numpy array of test inputs.
            n_samples: Number of prior samples to draw.

        Returns:
            Samples: A (n_samples, n_test) array of function samples.
        """
        np.random.seed(seed)
        mean = np.zeros(len(x_test))
        cov = self.kernel(x_test, x_test)

        cov += 1e-8 * np.eye(len(x_test))  # stability
        samples = np.random.multivariate_normal(
            mean.reshape(-1), cov, size=n_samples
        )
        return samples
