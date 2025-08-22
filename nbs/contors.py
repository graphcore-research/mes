# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])

SAVE_DIR = Path().cwd()  # mes/nbs
DATA_DIR = Path().cwd().parent / "data"  # mes/data
assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"

# %%
# Split into baseline vs sweep acquisition functions

df = pd.read_json(DATA_DIR / "benchmark_df_sweep_3.json")
df["regret"] = df.apply(
    lambda x: x["y_true_max"] - x["final_y_max"], axis=1
)
df = df.drop(columns=["steps", "y_max_history", "y_true_max", "final_y_max", "max_iters"])
df.query('kernel_type == "matern-5/2" and n_dim == 4 and wd == 0').head()

# %%

# Check every config has the same number of runs
group_cols = [c for c in df.columns if c not in ["run_id", "regret"]]
expected = df["run_id"].nunique()
n_runs = df.groupby(group_cols)["run_id"].nunique()
assert n_runs.eq(expected).all(), (
    f"Configs with != {expected} runs:\n{n_runs[n_runs != expected]}"
)

regret_df = (
    df.assign(regret=df.groupby(group_cols)["regret"].transform("mean"))
    .drop(columns="run_id")
    .drop_duplicates(subset=group_cols)
    .reset_index(drop=True)
)


regret_df.query('kernel_type == "matern-5/2" and n_dim == 4 and wd == 0').head()

# %%
# All EI and RS are the same
regret_df.query('kernel_type == "matern-5/2" and n_dim == 4 and acq_func == "expected_improvement"').head()

# %%
# Ok let's split up our runs into baselines (which don't vary with lr and wd) and sweeps
baseline_acq = ["expected_improvement", "random_search"]
base_df = regret_df.query("acq_func in @baseline_acq").copy()

# Baseline regrets do not vary with lr/wd within a fixed config. We should assert this.
fixed_cols = [c for c in regret_df.columns if c not in ["lr", "wd", "regret"]]
_nuniq = base_df.groupby(fixed_cols).regret.nunique()
assert _nuniq.eq(1).all(), (
    "Baseline regrets vary with lr/wd for some configs:\n"
    f"{_nuniq[_nuniq > 1]}"
)

# Ok great, so let's drop our superfulous lr and wd.
# base_df is nice and simple!
base_df = base_df.drop(columns=["lr", "wd"]).drop_duplicates()
base_df

# %%
# Here's our sweep:
sweep_acq = ["ves_gamma", "ves_mc_gamma"]
sweep_df = regret_df.query("acq_func in @sweep_acq").copy()
sweep_df.query('kernel_type == "matern-5/2" and n_dim == 4').head()
# %%

# Contour Plot Spec (for future self)
# - Figures: one per (acq_func, n_dim), excluding baselines.
# - Subplots: one per kernel_type within the figure.
# - Axes: X=lr, Y=wd; label axes accordingly. lr may use log scale.
# - Surface: contour of regret over the lr–wd grid for that kernel.
# - Colorbars: one per subplot, labeled 'regret'; overlay baseline methods as
#   horizontal lines with labels on the colorbar.
# - Color scale: DO NOT share vmin/vmax across subplots; each subplot sets its
#   own color limits based on its data.
# - Data completeness: require a full lr–wd grid per subplot; raise an error if
#   any required grid points are missing (no silent interpolation/filling).
# - Highlights: mark the best (lowest regret) grid point on each subplot.
# - Titles: subplot title = kernel_type; figure suptitle = "{acq_func} | n_dim={n_dim}".
# - Filters: apply fixed params consistently (e.g., len_scale, max_iters) across plots.
# - Ordering: keep a consistent kernel_type order and axis tick order across figures.
# - Input: use df (averaged over run_id) with columns: acq_func, kernel_type,
#   n_dim, lr, wd, regret (plus fixed params).

# %%

# Minimal implementation: one figure per (acq_func, n_dim) with subplots per kernel.
# - Per-subplot color scale (no sharing)
# - Require complete lr–wd grid per kernel (raise if missing)
# - Mark best (lowest) regret per subplot

def _grid_from_kernel_df(kdf: pd.DataFrame):
    """Return sorted lr, wd and a dense regret matrix; raise if grid incomplete."""
    lr_vals = np.sort(kdf["lr"].unique())
    wd_vals = np.sort(kdf["wd"].unique())

    # Build pivot; ensure full grid
    pivot = kdf.pivot(index="wd", columns="lr", values="regret").reindex(
        index=wd_vals, columns=lr_vals
    )
    if pivot.isna().any().any():
        missing = np.argwhere(np.asarray(pivot.isna()))
        raise ValueError(
            f"Incomplete lr–wd grid for kernel {kdf['kernel_type'].iloc[0]}: found NaNs"
        )
    Z = pivot.values.astype(float)
    return lr_vals, wd_vals, Z


def plot_contours_minimal(sweep_df: pd.DataFrame, acq_func: str, n_dim: int):
    """Plot minimal contour figures for a single (acq_func, n_dim).

    - One subplot per kernel_type present in sweep_df for this acq/dim.
    - X axis is lr (log scale), Y axis is wd (linear here for simplicity).
    - Individual color scales per subplot, with a colorbar each.
    - Best (min regret) point marked.
    """
    sdf = sweep_df.query("acq_func == @acq_func and n_dim == @n_dim").copy()
    if sdf.empty:
        raise ValueError(f"No sweep data for acq_func={acq_func}, n_dim={n_dim}")

    kernel_types = sorted(sdf["kernel_type"].unique())
    n_plots = len(kernel_types)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for i, (kt, ax) in enumerate(zip(kernel_types, axes)):
        kdf = sdf.query("kernel_type == @kt")
        lr_vals, wd_vals, Z = _grid_from_kernel_df(kdf)

        # Per-subplot color limits
        vmin, vmax = float(Z.min()), float(Z.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9

        LR, WD = np.meshgrid(lr_vals, wd_vals)
        cf = ax.contourf(
            LR, WD, Z, levels=16, vmin=vmin, vmax=vmax, cmap="viridis"
        )
        cbar = plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.08)
        cbar.set_label("regret")

        # Axes formatting
        ax.set_title(kt)
        ax.set_xlabel("lr")
        if i == 0:
            ax.set_ylabel("wd")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Mark best point
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        ax.plot(lr_vals[min_idx[1]], wd_vals[min_idx[0]], "wo", ms=5, mec="k")

    pretty = acq_func.replace("_", " ").title()
    fig.suptitle(f"{pretty} | n_dim={n_dim}")
    out = SAVE_DIR / f"contours_min_{acq_func}_{n_dim}d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Example minimal call (adjust if needed)
try:
    plot_contours_minimal(sweep_df, acq_func="ves_gamma", n_dim=4)
except Exception as e:
    print("Skipping minimal contour plot:", e)
