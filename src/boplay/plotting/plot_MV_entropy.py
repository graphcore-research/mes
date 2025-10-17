import numpy as np
import matplotlib.pyplot as plt

from ..gp import GaussianProcess1D
from ..kernels import KERNELS
from ..kde import fit_kde



def plot_gp_and_kde(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_samples:int=100,
    n_x:int=101,
    y_min:float=-4,
    y_max:float=6,
    kde_dens_max:float=5
) -> tuple[plt.figure, float]:
    """
    Plot the GP and KDE for a 1D regression problem.

    Args:
        x_train: 1D numpy array of training inputs.
        y_train: 1D numpy array of training targets.
        filename: Path to save the figure.
        n_samples: Number of sample functions from the model/posterior.
        n_x: Number of points to plot for the x-axis.

    Returns:
        fig: The figure object.
        entropy: The entropy of the KDE.
    """

    # points and ranges for (x, y) plotting
    x_min, x_max = np.min(x_train), np.max(x_train)
    dx = x_max - x_min

    # add some padding to the plot
    x_min -= 0.2 * dx
    x_max += 0.2 * dx
    x_plot = np.linspace(x_min, x_max, n_x)

    # Model and predict
    gp = GaussianProcess1D(x_train, y_train, kernel=se_kernel)
    y_plot, y_cov = gp.predict(x_plot)
    y_sd = np.sqrt(np.diag(y_cov))

    # Sample random functions/walks consistent with the data points
    np.random.seed(10)
    y_samples = gp.sample_posterior(x_plot, n_samples=n_samples)
    y_star_samples = np.max(y_samples, axis=0)

    # KDE
    kde = fit_kde(y_star_samples)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Subplot 1/2: plot the GP and the training data
    ax = axes[0]
    ax.fill_between(x_plot, y_plot - y_sd, y_plot + y_sd, alpha=0.2)
    label_samples = "sampled functions"
    label_peaks = "sample peaks"
    for y_sample in y_samples.T:
        ax.plot(x_plot, y_sample, 'r-', alpha=0.1, label=label_samples)
        i_top = np.argmax(y_sample)
        ax.scatter(x_plot[i_top], y_sample[i_top], color='b', alpha=0.2, s=10, zorder=10, label=label_peaks)
        label_samples, label_peaks = None, None

    ax.plot(x_plot, y_plot, label='GP mean', color='k')
    ax.plot(x_train, y_train, 'o', label='Training data')
    ax.set_ylim(y_min, y_max)
    ax.legend()

    # Subplot 2/2: vertically plot the KDE and the sampled y_max values
    ax = axes[1]
    y_kde_points = np.linspace(y_min, y_max, 101)
    y_kde_densities = kde.density(y_kde_points)
    ax.plot(y_kde_densities, y_kde_points, label=f'KDE {kde.entropy:.3f}')
    ax.scatter([0] * n_samples, y_star_samples, color='b', alpha=0.2, s=10, zorder=10)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.2, kde_dens_max)
    ax.legend()

    return fig, kde.entropy


