import numpy as np
from scipy.stats import norm


def gaussian_bin_centers(n:int, mu:float=0.0, sigma:float=1.0) -> list[float]:
    """ Get the quantiles of the Gaussian """
    ps = (np.arange(n) + 0.5) / n   # bin centers in [0, 1]
    return norm.ppf(ps, loc=mu, scale=sigma)


def sample_yn1_ymax(
    *,
    y_mean: np.ndarray,
    y_cov: np.ndarray,
    n_yn1: int=10,
    n_ymax: int=30,
    batch_size: int=1e9,
    noise: float=1e-9,
) -> np.ndarray:
    """
    Given the mean and covariance of all the function values at all the x-locations,
    for each x-location, sample y_{n+1} values, for each y_n1, we then sample n_ymax
    full functions (vector of n_x y values) from a model fit to data that includes
    the new (x, y_n1) point. The output is matrix of (n_x, n_yn1) values for y_n1 and
    a tensor of (n_x, n_yn1, n_ymax) sampled y_max values.

    Given a single x-location, x, and a single y_n1 value and a single y-sample vector
    of y-vals at for whole x-grid, the way to update the y-sample to include (x, y_n1)
    is by the following formula:
        Given y_sample ~ N(y_mean, y_cov) where y_mean and y_cov are form the GP with n
        (x, y) data points, and we have a new (x, y_n1) point, we want to update the
        y_sample to include the new (x, y_n1) point.

            y_sample_new = y_sample + (y_n1 - y_sample[x]) * y_cov[x, :] / y_cov[x, x]
        
        where the shapes are:
            y_sample_new:         (n_x, ) vector
            (y_n1 - y_sample[x]): scalar
            y_cov[x, :]:          (n_x, ) vector
            y_cov[x, x]:          scalar
        
    This function is the above computation that has been vectorised over 3 axes:
        (1) n_x x-locations  (supports minibatching)
        (2) n_yn1 sampled y_n1 values
        (3) n_ymax y-sampled vectors

    Args:
        y_mean: np.ndarray, shape (n_x, 1)
        y_cov: np.ndarray, shape (n_x, n_x)
        n_yn1: int, number of y_n1 values to sample for each x-location
        n_ymax: int, number of functions to sample from the model
        batch_size: int, number of x-locations to process in each batch
        noise: float, noise to add to the covariance matrix

    Returns:
        y_n1_output: np.ndarray, shape (n_x, n_yn1)
        y_funcs_output: np.ndarray, shape (n_x, n_yn1, n_ymax, n_x)
        y_max_output: np.ndarray, shape (n_x, n_yn1, n_ymax)
    """
    n_x = y_mean.shape[0]       # total number of x -locations
    bs = min(batch_size, n_x)   # batch size

    batch_idx_subsets = np.array_split(np.arange(n_x), n_x // bs)
     
    # (n_x, n_x) square matrices
    y_cov += noise * np.eye(y_cov.shape[0])
    chol_k = np.linalg.cholesky(y_cov)
    y_sd = np.sqrt(np.diag(y_cov))[:, None]

    # (n_ymax, n_x) matrix, each row is one vector of y-values for the n_x
    # locations generated from the model using current data
    z_ymax = np.random.normal(size=(n_ymax, y_mean.shape[0]))
    y_funcs = y_mean.T + z_ymax @ chol_k.T  # (n_ymax, n_x)

    # (n_yn1,) vector of z-scores for the y_n1 values to be computed later
    z_yn1 = gaussian_bin_centers(n_yn1).reshape(1, n_yn1)

    # output tensors to be filled with each minibatch and concat at the end
    y_n1_output = []       # (n_x, n_yn1)
    y_funcs_output = []    # (n_x, n_yn1, n_ymax, n_x)
    y_max_output = []      # (n_x, n_yn1, n_ymax)

    for batch_idx in batch_idx_subsets:

        # generate y_{n+1} values for each x in this batch
        y_mean_b = y_mean[batch_idx, :]
        y_sd_b = y_sd[batch_idx, :]
        y_n1_b = y_mean_b + y_sd_b * z_yn1  # (bs, n_yn1)

        # (bs, n_x) each row is the delta to adjust a sample fun for one x in batch
        fn_delta = y_cov[batch_idx, :] / np.diag(y_cov)[batch_idx, None]

        # (bs, n_ymax), get the y-values from the sample funs at x locs in this batch
        y_funcs_bx = y_funcs[:, batch_idx].T

        # (bs, n_yn1, n_ymax) <- (bs, n_yn1, 1) - (bs, 1, n_ymax)
        # for each x in batch, get diff between
        #     (1) sampled funcs at x and
        #     (2) sampled y_n1 vals at x
        y_diffs =  y_n1_b[:,:, None] - y_funcs_bx[:, None, :]

        # (bs, n_yn1, n_ymax, n_x)
        y_funcs_b = (
            y_funcs[None, None, :, :] +       # ( 1,     1, n_ymax, n_x)
            (
                y_diffs[:, :, :, None] *      # (bs, n_yn1, n_ymax,   1)
                fn_delta[:, None, None, :]    # (bs,     1,      1, n_x)
            )
        )

        y_n1_output.append(y_n1_b)
        y_funcs_output.append(y_funcs_b)
        y_max_output.append(y_funcs_b.max(axis=3))

    y_n1_output = np.concatenate(y_n1_output, axis=0)
    y_funcs_output = np.concatenate(y_funcs_output, axis=0)
    y_max_output = np.concatenate(y_max_output, axis=0)

    return y_n1_output, y_funcs_output, y_max_output
