import numpy as np
from scipy.special import digamma, gammaln, polygamma


def estimate_gamma_params(
    *,
    x: np.ndarray,
    max_iters: int= 200,
    k_min: float=0.01,
    k_max: float = 10.0,
    lr: float = 1.,
    wd: float = 0.
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the parameters of a Gamma distribution.

    Args:
        x: np.ndarray, shape (n_rows, n_cols)
        max_iters: int, maximum number of iterations
        k_min: float, minimum value for k

    Returns:
        k: np.ndarray, shape (n_rows, )
        theta: np.ndarray, shape (n_rows, )
    """
    assert x.ndim == 2, "x must be a 2D array [B, N]"
    assert max_iters >= 0, "max_iters must be non-negative"
    assert k_min > 0, "k_min must be positive"

    x_mean = x.mean(axis=1)
    x_var = x.var(axis=1, ddof=1)
    k = (x_mean**2 / x_var).clip(min=k_min)
    s = np.log(x_mean) - np.log(x).mean(axis=1)
    for _ in range(max_iters):
        grad = (np.log(k) - digamma(k) - s) / (1.0 / k - polygamma(1, k) + 1e-8)
        grad += k**2 * wd
        k -= grad * lr
        k = k.clip(min=k_min, max=k_max)
    theta = x_mean / k
    return k, theta


def gamma_log_likelihood(
    *,
    x: np.ndarray,
    k: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute row-wise log-likelihoods of Gamma(k, theta) for x.

    Args:
        x: np.ndarray, shape (n_rows, n_cols)
        k: np.ndarray, shape (n_rows, )
        theta: np.ndarray, shape (n_rows, )

    Returns:
        np.ndarray, shape (n_rows, ncols)
    """

    assert x.ndim == 2, "x must be a 2D array [B, N]"
    assert k.shape == (x.shape[0],), "k must have shape [B]"
    assert theta.shape == (x.shape[0],), "theta must have shape [B]"

    k = k[:, None]
    theta = theta[:, None]

    log_likelihood = (k - 1.0) * np.log(x) - x / theta - k * np.log(theta) - gammaln(k)
    return log_likelihood
