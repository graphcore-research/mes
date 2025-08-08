from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from boplay.plotting.plot_MV_entropy import plot_gp_and_kde
from boplay.plotting.animate import animate_files

# Folders
ROOT_DIR = Path(__file__).parent
FRAMES_DIR = ROOT_DIR / 'frames'
FRAMES_DIR.mkdir(exist_ok=True)

# Image filenames
GP_0_PNG = ROOT_DIR / "original_data.png"
Y_NEW_BENEFIT_PNG = ROOT_DIR / 'entropy_vs_y_new.png'
ANIMATION_GIF = ROOT_DIR / "animation.gif"

# Computational params
NUM_RANDOM_WALKS = 100
NUM_X_GRID = 41
X_NEW = 3
Y_NEW_MIN = -2
Y_NEW_MAX = 3


if __name__ == '__main__':
    # make some data points
    x_train = np.asarray([1, 5, 7])
    y_train = np.asarray([0.5, 0, 0.5])

    # plot the data + KDE model on original data
    fig_0, entropy_0 = plot_gp_and_kde(x_train, y_train, n_samples=100, n_x=41)
    fig_0.savefig(GP_0_PNG)
    print(f"Saved image: {GP_0_PNG}")

    # make synthetic new points
    x_new = np.array([X_NEW])
    y_new_vals = np.linspace(Y_NEW_MIN, Y_NEW_MAX, 31)
    entropies = []
    for i, y_new_i in enumerate(y_new_vals):

        # original + synthetic data
        x_train_i = np.concatenate([x_train, x_new])
        y_train_i = np.concatenate([y_train, np.array([y_new_i])])

        # plot the data + KDE model on original data
        fig_i, entropy_i = plot_gp_and_kde(
            x_train_i,
            y_train_i,
            n_samples=NUM_RANDOM_WALKS,
            n_x=NUM_X_GRID,
        )
        entropies.append(entropy_i)
        fig_i.savefig(FRAMES_DIR / f"{i:04}.png")
        plt.close(fig_i)
        print(f"Saved image: {FRAMES_DIR}/{i:04}.png")

    
    animate_files(
        frames_dir=FRAMES_DIR,
        output_filename=ANIMATION_GIF,
    )
    print(f"Saved animation: {ANIMATION_GIF}")


    # make a plot of synthetic y-value vs "benefit" of that y-value
    ei_vals = np.clip(y_new_vals - max(y_train), 0, 1000)
    entropies = entropy_0 - np.asarray(entropies)

    fig, ax = plt.subplots()
    ax.plot(y_new_vals, entropies, label="Entropy Reduction")
    ax.plot(y_new_vals, ei_vals, label="Expected improvement")
    ax.set_xlabel("new y value")
    ax.set_xlabel("benefit of new y-value")
    ax.vlines(x=0.5, ymin = 0, ymax=max(entropies), label="y data max", color="g")
    ax.legend()
    fig.savefig(Y_NEW_BENEFIT_PNG)
    print(f"Saved y benefit: {Y_NEW_BENEFIT_PNG}")


