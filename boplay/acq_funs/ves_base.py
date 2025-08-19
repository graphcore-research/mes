"""
This is the base acquisition function for all Max-Value Variational Entropy search
(VES) implementations. All implementations use the same samples of various y-values
and learn to fit probability distributions to those y-values
 - exponential/wedge with constant shape
 - exponential/wedge with shrinking shape for larger y_n1 values
 - gamma (humpy wedge) is the ICML 2025 proposed method
 - truncated normal
 - many more!

Individual VES implementations are defined in their own files and must contain
the following functions:
 - initialize_params: initialize the parameters for the mapping from y_n1 -> ymax distro parameters
 - compute_distro_params: y_n1 + params -> compute the ymax distributionparameters
 - compute_log_likelihood: ymax + distro_params -> compute the log likelihood of the ymax values

Then an optimizer can be used to find the y_n1 -> ymax parameter mapping that maximizes
the log likelihood of all y-max values.
"""

from typing import Callable
import numpy as np
import torch as pt

from boplay.acq_funs.mes_utils import sample_yn1_ymax

from boplay.acq_funs.pytorch_settings import PT_DTYPE, PT_DEVICE

pt_params = {
    "dtype": PT_DTYPE,
    "device": PT_DEVICE,
}


def optimize_adam(
    *,
    theta: pt.Tensor,
    loss_fn: Callable,
    max_iters: int = 200,
    tol: float=1e-9,
    lr: float=1e-2,
    wd: float = 0.
) -> tuple[pt.Tensor, float]:
    """
    Optimize the given parameters using Adam.

    Args:
        theta: pt.Tensor, shape (n_x, 4)
        loss_fn: Callable, the loss function to optimize
        max_iters: int, the maximum number of iterations
        tol: float, the tolerance for the optimization
        lr: float, the learning rate for the optimization

    Returns:
        theta: pt.Tensor, shape (n_x, 4)
        final_loss: float, the optimized loss
    """
    opt = pt.optim.Adam([theta], lr=lr, amsgrad=True, weight_decay=wd)
    prev_loss = float("inf")
    L = None

    for _ in range(max_iters):
        opt.zero_grad(set_to_none=True)
        L = loss_fn(theta)
        L.backward()
        opt.step()

        # Early stopping
        if abs(prev_loss - L.item()) < tol:
            break
        prev_loss = L.item()

    return theta.detach(), L.item()


def ves_base(
    *,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    y_best: float,
    n_yn1: int=10,
    n_ymax: int=30,
    batch_size: int=1e9,
    initialize_params: Callable,
    compute_distro_params: Callable,
    compute_log_likelihood: Callable,
    idx_train: np.ndarray,
) -> np.ndarray:
    """
    Compute the acquisition function for the base case.

    NOTE: this function has numpy arrays as input and output, and torch tensors
    for the middle part.

    Args:
        y_mean: np.ndarray, shape (n_x, 1)
        y_cov: np.ndarray, shape (n_x, 1)
        y_best: float
        n_yn1: int
        n_ymax: int
        batch_size: int
        initialize_params: Callable
        compute_distro_params: Callable
        compute_log_likelihood: Callable
        idx_train: np.ndarray, shape (n_x,)
    Returns:
        np.ndarray, shape (n_x,)
    """
    y_mean = np.array(y_mean).reshape(-1, 1)

    # Part 1/3: Sample y_n1 and y_max
    # NUMPY ARRAYS
    y_n1_samples, _, y_max_samples, _ = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=batch_size,
    )

    y_best_n1 = np.clip(y_n1_samples, min=y_best)
    y_max_shifted = y_max_samples - y_best_n1[:, :, None]

    # make sure they aren't exactly zero
    y_max_shifted = np.clip(y_max_shifted, min=1e-8)

    # NUMPY ARRAYS
    params_initial = initialize_params(
        y_mean=y_mean,
        y_cov=y_cov,
        y_n1_samples=y_n1_samples,
        y_max_samples=y_max_samples,
        y_max_shifted=y_max_shifted,
        y_best=y_best,
    )

    # NUMPY ARRAYS
    # remove the training points from the parameters and y_n1_samples
    # and y_max_shifted, we dont want the acq_fun for past points.
    idx_test = np.setdiff1d(np.arange(y_mean.shape[0]), idx_train)
    params_initial = params_initial[idx_test]
    y_n1_samples = y_n1_samples[idx_test]
    y_max_shifted = y_max_shifted[idx_test]

    # TORCH TENSORS FROM HERE ONWARDS
    params_initial = pt.nn.Parameter(pt.tensor(params_initial, **pt_params))
    y_n1_samples = pt.tensor(y_n1_samples, **pt_params)
    y_max_shifted = pt.tensor(y_max_shifted, **pt_params)

    # get the optimal parameters
    def loss_fn(params: pt.Tensor) -> float:
        """
        Compute the loss for a given set of ramp parameters.
        """
        distro_params = compute_distro_params(
            y_n1_samples=y_n1_samples,
            params=params,
        )
        log_likelihoods = compute_log_likelihood(
            y_max_shifted=y_max_shifted,
            distro_params=distro_params,
        )

        loss = -log_likelihoods.nanmean()
        if loss.isnan():
            import pdb; pdb.set_trace()
            print(loss)
        return loss

    # optimize the parameters
    # params_optimal, _ = optimize_lbfgs(theta=params_initial,loss_fn=loss_fn)
    params_optimal, _ = optimize_adam(theta=params_initial, loss_fn=loss_fn)

    # (n_x_test, n_yn1):  compute the distro parameters
    distro_params = compute_distro_params(
        y_n1_samples=y_n1_samples,
        params=params_optimal,
    )

    # (n_x_test, n_yn1): for each parameter, compute the log likelihood
    log_likelihoods = compute_log_likelihood(
        y_max_shifted=y_max_shifted,
        distro_params=distro_params,
    )

    # (n_x_test,): take the mean over the y_n1 dimension (MC averaging over y_n1)
    acq_fun_vals = log_likelihoods.mean(axis=1)

    # (n_x,): initilize a vector of all acq_fun values to a minimal value
    acq_fun_min = acq_fun_vals.min().item()
    acq_fun_max = acq_fun_vals.max().item()
    acq_fun_lo = acq_fun_min - (acq_fun_max - acq_fun_min) * 0.1
    acq_fun_vals_np = acq_fun_lo * np.ones(y_mean.shape[0], dtype=np.float64)

    # (n_x,) <- (n_x_test,)
    acq_fun_vals_np[idx_test] = acq_fun_vals.detach().cpu().numpy()

    return acq_fun_vals_np
