from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MES_VI.plot_MV_entropy import plot_gp_and_kde


ROOT = Path(__file__).parent
frames_dir = ROOT / 'frames'
frames_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    # make some data points
    x_train = np.asarray([1, 5, 7])
    y_train = np.asarray([0.5, 0, 0.5])

    # plot the data + KDE model on original data
    fig_0, entropy_0 = plot_gp_and_kde(x_train, y_train, n_samples=100, n_x=41)
    fig_0.savefig(ROOT / "original_data.png")

    # make synthetic new points
    x_new = np.array([6])
    y_new_vals = np.linspace(-2, 6, 31)
    entropies = []
    for i, y_new_i in enumerate(y_new_vals):

        # original + synthetic data
        x_train_i = np.concatenate([x_train, x_new])
        y_train_i = np.concatenate([y_train, np.array([y_new_i])])

        # plot the data + KDE model on original data
        fig_i, entropy_i = plot_gp_and_kde(
            x_train_i,
            y_train_i,
            n_samples=100,
            n_x=41,
        )
        fig_i.savefig(frames_dir / "{i:04}.png")

        


    
    
    entropies = entropy_0 - np.asarray(entropies)

    ei_vals = np.clip(y_new_vals - 0.5, 0, 1000)

    fig, ax = plt.subplots()
    ax.plot(y_new_vals, entropies, label="Entropy difference")
    ax.plot(y_new_vals, ei_vals, label="Expected improvement")
    ax.vlines(x=0.5, ymin = 0, ymax=max(entropies), label="y data max")
    fig.savefig(frames_dir.parent / 'entropy_vs_y_new.png')
