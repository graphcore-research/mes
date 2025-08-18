import numpy as np
import torch as pt


half_log_pi_const = float(-0.5 * np.log(2 * np.pi))


def gaussian_log_likelihood(
    *,
    x: pt.Tensor,
    mean: pt.Tensor,
    log_std: pt.Tensor,
) -> pt.Tensor:
    """
    Compute the Gaussian log likelihood for data points, each row
    of x is a dataset with a mean and log_std.

    Args:
        x: pt.Tensor, shape (n_x, n_points)
        mean: pt.Tensor, shape (n_x,)
        log_std: pt.Tensor, shape (n_x,)

    Returns:
        lhood: pt.Tensor, shape (n_x, n_points)
    """
    assert x.shape == mean.shape == log_std.shape, (
        f"x.shape: {x.shape}, mean.shape: {mean.shape}, log_std.shape: {log_std.shape}"
    )
    sdev = log_std.exp().clip(1e-6)
    lhood = half_log_pi_const - log_std - 0.5 * (x - mean)**2 / sdev**2
    
    return lhood
