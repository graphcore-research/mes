import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from boplay.kde import _optimal_kde_bandwidth


if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.normal(0, 1, size=100) * 1.0

    # Bandwidth candidates
    bandwidths = np.linspace(0.1, 1.0, 20)
    best_bw, _ = _optimal_kde_bandwidth(data)

    # Fit KDE with optimal bandwidth
    kde = KernelDensity(kernel="gaussian", bandwidth=best_bw)
    kde.fit(data.reshape(-1, 1))

    # Evaluate KDE on a grid
    x_plot = np.linspace(-4, 4, 1000).reshape(-1, 1)
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)

    entropy = -np.mean(kde.score_samples(data.reshape(-1, 1)))

    # Plot histogram and KDE curve
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, density=True, alpha=0.5, label="Histogram")
    plt.plot(x_plot[:, 0], dens, label=f"KDE {entropy:.3f}", linewidth=2)
    plt.title(f"KDE with Optimal Bandwidth = {best_bw:.2f}")
    plt.xlabel("Data")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("kde.png")
