import numpy as np

from boplay.kernels import KERNELS
from boplay.acq_funs.mes_utils import sample_yn1_ymax

from boplay.gp import GaussianProcess


def test_sampling_y_max():
    # make some fake data
    x_train = np.linspace(0, 100, 6).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train / 100).reshape(-1, 1)

    # Define a GP model with an SE kernel
    kernel = lambda x1, x2: KERNELS["se"](x1, x2, len_scale=10.0, sigma_f=1.0)
    model = GaussianProcess(x_train=x_train, y_train=y_train, kernel=kernel)
    
    # get the mean and cov of predictions
    n_x_test = 101
    x_test = np.linspace(0, 1, n_x_test).reshape(-1, 1)
    y_mean, y_cov = model.predict(x_test=x_test)

    # sample y_n1 and y_max values
    n_yn1 = 10
    n_ymax = 30
    y_n1_samples, y_funcs_samples, y_max_samples, _ = sample_yn1_ymax(
        y_mean=y_mean,
        y_cov=y_cov,
        n_yn1=n_yn1,
        n_ymax=n_ymax,
        batch_size=n_x_test,
    )

    # ensure all tensors have the correct shape
    assert y_n1_samples.shape == (n_x_test, n_yn1)
    assert y_funcs_samples.shape == (n_x_test, n_yn1, n_ymax, n_x_test)
    assert y_max_samples.shape == (n_x_test, n_yn1, n_ymax)

    # ensure each sampled function passes through the correct y_n1 at x_idx
    for x_idx in range(n_x_test):
        for j in range(n_yn1):
            y_n1_ij = y_n1_samples[x_idx, j]
            y_funcs_ij = y_funcs_samples[x_idx, j, :, :]

            # make sure each sampled function passes through y_n1 at x_idx
            assert max(y_n1_ij - y_funcs_ij[:, x_idx]) < 1e-9
