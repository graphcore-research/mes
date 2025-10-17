import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from boplay.plotting.animate import animate_files


ROOT = Path(__file__).parent.parent.parent


def plot_gp_and_acq_fun(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    acq_fun_vals: np.ndarray,
    ax: plt.Axes,
    y_min: float = -4,
    y_max: float = 6,
) -> None:
    """
    Plot the GP and the acquisition function on the given axes.
    """
    assert x_train.shape[0] == y_train.shape[0]
    assert x_grid.shape[0] == acq_fun_vals.shape[0]
    assert x_grid.shape[1] == 1
    assert x_train.shape[1] == 1

    x_train = x_train.reshape(-1)
    x_grid = x_grid.reshape(-1)
    y_train = y_train.reshape(-1)
    y_mean = y_mean.reshape(-1)
    y_sd = y_sd.reshape(-1)
    acq_fun_vals = acq_fun_vals.reshape(-1)

    # Line 1/2: plot the GP and the training data
    ax.plot(x_grid, y_true, label="True function", color="b")
    ax.fill_between(x_grid, y_mean - y_sd, y_mean + y_sd, alpha=0.2)
    ax.plot(x_grid, y_mean, label="GP mean", color="k")
    ax.plot(x_train, y_train, "o", label="Training data")
    ax.set_ylim(y_min, y_max)

    # Line 2/2: plot the acquisition function and its peak
    acq_fun_vals = acq_fun_vals - np.min(acq_fun_vals)
    acq_fun_vals = acq_fun_vals / (np.max(acq_fun_vals) + 1e-8)
    acq_fun_vals = acq_fun_vals + y_min

    ax.plot(x_grid, acq_fun_vals, label="Acquisition function", color="r")

    acq_fun_vals_max = np.max(acq_fun_vals)
    acq_fun_vals_max_idx = np.argmax(acq_fun_vals)
    ax.plot(
        x_grid[acq_fun_vals_max_idx],
        acq_fun_vals_max,
        "o",
        label="Acquisition function max",
        color="r",
    )

    ax.legend()
    ax.set_title(f"Iteration {len(x_train)}")


def plot_bo_history_1d(
    x_grid: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    state_history: list[dict],
    animation_gif: Path,
    tmp_dir: Path = None,
) -> None:
    if tmp_dir is None:
        tmp_dir = ROOT / "pics" / f"{animation_gif.stem}_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)

    for file in tmp_dir.glob("*.png"):
        file.unlink()

    for state in state_history:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        n_train = state["n_train"]
        y_mean = state["y_mean"].reshape(-1)
        y_sd = state["y_sd"].reshape(-1)
        acq_fun_vals = state["acq_fun_vals"].reshape(-1)
        x_train_n = x_train[:n_train]
        y_train_n = y_train[:n_train]

        plot_gp_and_acq_fun(
            x_train=x_train_n,
            y_train=y_train_n,
            y_true=y_true,
            x_grid=x_grid,
            y_mean=y_mean,
            y_sd=y_sd,
            acq_fun_vals=acq_fun_vals,
            ax=ax,
        )
        fig.savefig(tmp_dir / f"bo_state_{n_train:03d}.png")
        plt.close(fig)

    animate_files(tmp_dir, animation_gif)
