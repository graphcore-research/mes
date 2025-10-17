import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity


def _optimal_kde_bandwidth(data: np.ndarray) -> tuple[float, float]:
    """Finds the optimal bandwidth for KDE using leave-one-out CV.

    Args:
        data: A 1D numpy array of data points.
        bandwidths: A list or array of candidate bandwidths.

    Returns:
        A tuple containing:
            - best_bandwidth: The bandwidth that maximizes the log-likelihood.
            - best_score: The corresponding average log-likelihood.
    """
    data = np.asarray(data).reshape(-1, 1)
    loo = LeaveOneOut()
    best_score = -np.inf
    best_bandwidth = None

    bw_max = np.max(data) - np.min(data)
    bw_min = bw_max / 1000
    bandwidths = np.linspace(np.log(bw_min), np.log(bw_max), 100)
    bandwidths = np.exp(bandwidths)

    for bw in bandwidths:
        log_likelihood = 0.0
        for train_idx, test_idx in loo.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]

            kde = KernelDensity(kernel="gaussian", bandwidth=bw)
            kde.fit(train_data)
            log_likelihood += kde.score_samples(test_data)[0]

        avg_log_likelihood = log_likelihood / len(data)

        if avg_log_likelihood > best_score:
            best_score = avg_log_likelihood
            best_bandwidth = bw

    return best_bandwidth, best_score


def fit_kde(x_train: np.ndarray, bw: float | None = None) -> KernelDensity:
    """
    Fit a KDE to the data and return the KDE object with the entropy computed
    """
    x_train = np.asarray(x_train).reshape(-1, 1)
    if bw is None:
        best_bw, _ = _optimal_kde_bandwidth(x_train)
    else:
        best_bw = bw
    kde = KernelDensity(kernel="gaussian", bandwidth=best_bw)
    kde.fit(x_train)
    kde.density = lambda x: np.exp(kde.score_samples(x.reshape(-1, 1)))

    # compute and store the entropy as an attribute of the KDE object
    kde.entropy = -np.mean(kde.score_samples(x_train))
    return kde
