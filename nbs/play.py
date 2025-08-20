# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])

SAVE_DIR = Path().cwd()  # mes/paper_plots
DATA_DIR = Path().cwd().parent / "data"  # mes/data
assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"


def load_benchmark_data(data_dir):
    """Load and process benchmark data from JSON file."""
    df = pd.read_json(data_dir / "benchmark_df.json")

    # Calculate regret from y_true_max and y_max_history
    df["regret"] = df.apply(
        lambda x: x["y_true_max"] - np.array(x["y_max_history"]), axis=1
    )

    # Cleanup unnecessary columns
    df = df.drop(columns=["y_true_max", "final_y_max", "y_max_history"])

    # Explode history into separate rows
    df = df.explode(["regret", "steps"], ignore_index=True)
    df["regret"] = pd.to_numeric(df["regret"], errors="raise")

    return df


def aggregate_regret_data(df):
    """Group and aggregate regret data by experimental conditions."""
    regret_df = df.groupby(["kernel_type", "n_dim", "acq_func", "steps", "lr", "wd"])
    regret_df = regret_df["regret"].agg(["mean", "std", "count"])
    return regret_df


def _plot_contour(acq_data, acq_func, ax, i, vmin, vmax, value_col="mean"):
    """Helper function to create a single contour plot."""
    # Create pivot table for contour plotting
    lr_values = acq_data.index.get_level_values("lr").unique()
    wd_values = acq_data.index.get_level_values("wd").unique()

    # Create meshgrid
    lr_grid = np.array(sorted(lr_values))
    wd_grid = np.array(sorted(wd_values))
    LR, WD = np.meshgrid(lr_grid, wd_grid)

    # Fill Z values
    Z = np.zeros_like(LR)
    for j, wd in enumerate(wd_grid):
        for k, lr in enumerate(lr_grid):
            try:
                Z[j, k] = acq_data.loc[(lr, wd), value_col]
            except KeyError:
                Z[j, k] = np.nan

    # Create contour plot with slightly more levels for smoother visual appearance
    contourf = ax.contourf(LR, WD, Z, levels=15, alpha=1.0, vmin=vmin, vmax=vmax)

    # Configure axes
    ax.set_xlabel("Learning Rate")
    if i == 0:  # Only label y-axis for first subplot since they share y
        ax.set_ylabel("Weight Decay")
    ax.set_title(acq_func.replace("_", " ").title())
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    # Adjust tick label positioning to prevent overlap
    ax.tick_params(axis='x', pad=5)  # Add padding to x-axis tick labels
    if i == 0:
        ax.tick_params(axis='y', pad=5)  # Add padding to y-axis tick labels only for first subplot

    return contourf


def _plot_contours(
    available_sweep_acqs,
    data,
    final_steps,
    baseline_acqs,
    kernel_type,
    n_dim,
    save_dir,
    value_col="mean",
):
    """Helper function to setup figure, plot all contour subplots and finish the figure."""
    # Create subplots with shared y-axis
    n_plots = len(available_sweep_acqs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    # Get global min/max for consistent colorbar scale including baselines
    all_z_values = []
    for acq_func in available_sweep_acqs:
        final_step = final_steps[acq_func]
        acq_data = data.loc[(acq_func, final_step)]
        all_z_values.extend(acq_data[value_col].values)

    # Include baseline values in the scale calculation
    for baseline_acq in baseline_acqs:
        if baseline_acq in data.index.get_level_values("acq_func"):
            final_step_baseline = final_steps[baseline_acq]
            baseline_data = data.loc[(baseline_acq, final_step_baseline)]
            baseline_value = baseline_data[value_col].iloc[0]
            all_z_values.append(baseline_value)

    vmin, vmax = np.min(all_z_values), np.max(all_z_values)

    # Plot all contours
    contourf = None
    for i, acq_func in enumerate(available_sweep_acqs):
        ax = axes[i]
        # Get data for this acquisition function at final step
        final_step = final_steps[acq_func]
        acq_data = data.loc[(acq_func, final_step)]

        # Create contour plot using helper function
        contourf = _plot_contour(acq_data, acq_func, ax, i, vmin, vmax, value_col)

    plt.tight_layout(pad=2.0)

    # Add single colorbar for all subplots (outside the plot area)
    cbar = plt.colorbar(contourf, ax=axes, aspect=10, pad=0.02, fraction=0.08)
    cbar.set_label("Mean Regret")

    # Add baseline annotations to colorbar
    _add_baseline_annotations(cbar, data, final_steps, baseline_acqs, value_col)
    plt.suptitle(f"{kernel_type.upper()} Kernel Regret ({n_dim}D)", y=1.02, fontsize=14)

    # Save figure
    filename = f"contour_{kernel_type.replace('/', '_')}_{n_dim}d.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.show()
    plt.close()


def _add_baseline_annotations(cbar, data, final_steps, baseline_acqs, value_col="mean"):
    """Helper function to add baseline annotations to the colorbar."""
    if not baseline_acqs:
        return

    for i, baseline_acq in enumerate(baseline_acqs):
        if baseline_acq in data.index.get_level_values("acq_func"):
            final_step_baseline = final_steps[baseline_acq]
            baseline_data = data.loc[(baseline_acq, final_step_baseline)]
            baseline_value = baseline_data[value_col].iloc[0]

            # Use abbreviated names for common acquisition functions
            acq_name = baseline_acq.replace("expected_improvement", "EI").replace(
                "random_search", "RS"
            )

            # Position all annotations at the same horizontal offset
            x_offset = 1.3

            # Add arrow pointing to baseline position on colorbar
            cbar.ax.annotate(
                f"{acq_name}: {baseline_value:.3f}",
                xy=(1.0, baseline_value),
                xycoords=("axes fraction", "data"),
                xytext=(x_offset, baseline_value),
                textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )


def plot_sweep_contours(
    regret_df,
    save_dir,
    sweep_acqs,
    baseline_acqs,
    value_col="mean",
):
    """
    Simple contour plotting for (lr, wd) sweeps per acquisition function.

    Assumptions
    -----------
    Your data is a pandas DataFrame named `regret_df` with a MultiIndex that includes
    these levels (names must match):
        'kernel_type', 'n_dim', 'acq_func', 'lr', 'wd', 'steps'
    And it contains a numeric column with the metric to plot (default 'mean').

    What it does
    ------------
    - For selected acquisition functions (e.g., ['ves_gamma', 'ves_mv_gamma']),
      it makes *one subplot per acq_func* showing a contour of (lr, wd) → value at
      the final step.
    - For baseline acq_funcs that don't use lr/wd (e.g., EI, random search)
      it computes the *final values* and shows them as annotations on the colorbar,
      but **does not** draw contours for them
    - Saves one figure per (kernel_type, n_dim).

    Usage
    -----
    plot_sweep_contours(
        regret_df,
        save_dir="plots",
        sweep_acqs=["ves_gamma", "ves_mv_gamma"],
        baseline_acqs=["expected_improvement", "random_search"],
        value_col="mean",
    )
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Get unique combinations of kernel_type and n_dim
    kernel_dims = regret_df.index.droplevel(["acq_func", "steps", "lr", "wd"]).unique()

    for kernel_type, n_dim in kernel_dims:
        # Filter data for this kernel_type and n_dim
        data = regret_df.loc[(kernel_type, n_dim)]

        # Get the final step for each acquisition function
        final_steps = data.groupby("acq_func").apply(
            lambda x: x.index.get_level_values("steps").max()
        )

        # Plot contours for sweep acquisition functions and finish the figure
        _plot_contours(
            sweep_acqs,
            data,
            final_steps,
            baseline_acqs,
            kernel_type,
            n_dim,
            save_dir,
            value_col,
        )


df = load_benchmark_data(DATA_DIR)
regret_df = aggregate_regret_data(df)

available_acq_funcs = regret_df.index.get_level_values("acq_func").unique()
print("Available acquisition functions:", ", ".join(available_acq_funcs))

print("Example data for matern-3/2, 4D:")
print(regret_df.loc[("matern-3/2", 4)])


plot_sweep_contours(
    regret_df,
    save_dir=str(SAVE_DIR / "contour_plots"),
    sweep_acqs=["ves_gamma", "ves_mc_gamma"],  # Use what's available in your data
    baseline_acqs=["expected_improvement", "random_search"],  # Multiple baselines
    value_col="mean",
)

# %%
