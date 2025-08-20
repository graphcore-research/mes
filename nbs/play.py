# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid"])

# %%
SAVE_DIR = Path().cwd()  # mes/paper_plots
DATA_DIR = Path().cwd().parent / "data"  # mes/data
assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"

# %% [markdown]
# ## Plotting HPs
#
# We'll want to make different figures for different kernel types, and dimensionality?

# %%
df = pd.read_json(DATA_DIR / "benchmark_df.json")

df["regret"] = df.apply(
    lambda x: x["y_true_max"] - np.array(x["y_max_history"]), axis=1
)
df = df.drop(columns=["y_true_max", "final_y_max", "y_max_history"])  # cleanup rows
df = df.explode(
    ["regret", "steps"], ignore_index=True
)  # explode history into seperate rows
df["regret"] = pd.to_numeric(df["regret"], errors="raise")
df

# %%
regret_df = df.groupby(["kernel_type", "n_dim", "acq_func", "steps", "lr", "wd"])
regret_df = regret_df["regret"].agg(["mean", "std", "count"])
regret_df  # multi-index of key: (kernel_type, n_dim, acq_func, steps)

# %%
regret_df.index.get_level_values("acq_func").unique()

# %% [markdown]
# This gives us a multi-index dataframe with unique keys of `(kernel_type, n_dim, acq_func, steps)` and mean, std, count regret for each row.
#
# To index into a single kernel / dimension, we can use:

# %%
print(regret_df.loc[("matern-3/2", 4)])


# %%
def plot_sweep_contours(
    regret_df,
    save_dir="plots",
    sweep_acqs=["ves_gamma", "ves_mv_gamma"],
    baseline_acq="expected_improvement",
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
    - For baseline acq_func that don't use lr/wd a.k.a ei
      it computes the *lowest final value* and shows those numbers on the figure,
      but **does not** draw contours for them, this should be a baseline
    - Saves one figure per (kernel_type, n_dim).

    Usage
    -----
    plot_sweep_contours(
        regret_df,
        save_dir="plots",
        sweep_acqs=["ves_gamma", "ves_mv_gamma"],
        baseline_acq="expected_improvement",
        value_col="mean",
    )
    """
    from pathlib import Path

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

        # Filter sweep acquisition functions that exist in data
        available_sweep_acqs = [
            acq for acq in sweep_acqs if acq in data.index.get_level_values("acq_func")
        ]

        if not available_sweep_acqs:
            print(f"No sweep acquisition functions found for {kernel_type}, {n_dim}")
            continue

        # Create subplots with shared y-axis
        n_plots = len(available_sweep_acqs)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
        if n_plots == 1:
            axes = [axes]

        # Get global min/max for consistent colorbar scale
        all_z_values = []
        for acq_func in available_sweep_acqs:
            final_step = final_steps[acq_func]
            acq_data = data.loc[(acq_func, final_step)]
            all_z_values.extend(acq_data[value_col].values)

        vmin, vmax = np.min(all_z_values), np.max(all_z_values)

        # Plot contours for sweep acquisition functions
        for i, acq_func in enumerate(available_sweep_acqs):
            ax = axes[i]

            # Get data for this acquisition function at final step
            final_step = final_steps[acq_func]
            acq_data = data.loc[(acq_func, final_step)]

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
            contourf = ax.contourf(
                LR, WD, Z, levels=15, alpha=1.0, vmin=vmin, vmax=vmax
            )

            ax.set_xlabel("Learning Rate")
            if i == 0:  # Only label y-axis for first subplot since they share y
                ax.set_ylabel("Weight Decay")
            ax.set_title(acq_func.replace("_", " ").title())
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

        # Add baseline information if available
        if baseline_acq in data.index.get_level_values("acq_func"):
            final_step_baseline = final_steps[baseline_acq]
            baseline_data = data.loc[(baseline_acq, final_step_baseline)]
            baseline_value = baseline_data[value_col].iloc[0]

            # Create a small colored box showing baseline value
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            # Get color for baseline value from the same colormap
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("viridis")
            baseline_color = cmap(norm(baseline_value))

        plt.tight_layout()

        # Add single colorbar for all subplots (outside the plot area)
        cbar = plt.colorbar(contourf, ax=axes, aspect=10, pad=0.02, fraction=0.08)
        cbar.set_label('Mean Regret')

        # Add baseline info next to colorbar if available
        if baseline_acq in data.index.get_level_values("acq_func"):
            # Add arrow pointing to EI position on colorbar using data coordinates
            cbar_ax = cbar.ax
            cbar_ax.annotate(
                f"EI: {baseline_value:.3f}",
                xy=(1.0, baseline_value),
                xycoords=("axes fraction", "data"),
                xytext=(1.3, baseline_value),
                textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )
        plt.suptitle(f"{kernel_type.upper()} Kernel Regret ({n_dim}D)", y=1.02, fontsize=14)

        # Save figure
        filename = f"contour_{kernel_type.replace('/', '_')}_{n_dim}d.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()


# %%
# Example usage with available acquisition functions
available_acq_funcs = regret_df.index.get_level_values("acq_func").unique()
print("Available acquisition functions:", available_acq_funcs)

# Test the plotting function with available acquisition functions
plot_sweep_contours(
    regret_df,
    save_dir=str(SAVE_DIR / "contour_plots"),
    sweep_acqs=["ves_gamma", "ves_mc_gamma"],  # Use what's available in your data
    baseline_acq="expected_improvement",
    value_col="mean",
)

# %%
