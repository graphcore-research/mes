# %%
# fmt: off
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid"])

SAVE_DIR = Path().cwd()  # mes/nbs
DATA_DIR = Path().cwd().parent / "data"  # mes/data
assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"

# %%
# Define these as use fuul
n_dim = 4
acq_func = "ves_gamma"
kt = "matern-3/2"

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
# base_df is nice and simple! I've replaced the names since it'll be easier later
base_df = base_df.drop(columns=["lr", "wd"]).drop_duplicates()
base_df.acq_func = base_df.acq_func.replace({
    "expected_improvement": "EI",
    "random_search": "RS"
})
base_df

# %%
# Here's our sweep:
sweep_acq = ["ves_gamma", "ves_mc_gamma"]
sweep_df = regret_df.query("acq_func in @sweep_acq").copy()
sweep_df.query('kernel_type == "matern-5/2" and n_dim == 4').head()
# %%
# Great, so for each aquisition function x n_dim we want to make a plot:
sdf = sweep_df.query("acq_func == @acq_func and n_dim == @n_dim").copy()
# And for each kernel type a subplot
kdf = sdf.query("kernel_type == @kt").copy()
kdf.head()
# %%
# Now some pivot magic to make a 2D matrix of wd x lr
pivot = kdf.pivot(index="wd", columns="lr", values="regret")
assert not pivot.isna().any().any(), "Missing points in the lr-wd grid"
pivot
# %%
# It'd be cool if we could just plug this straight in to matplotlib, but we need a 1d XY values for each point and a 2d matrix of values.

def _plot_single(kdf, ax, title):
    pivot = kdf.pivot(index="wd", columns="lr", values="regret")
    assert not pivot.isna().any().any(), "Missing points in the lr-wd grid"

    X = pivot.columns.values # lr
    Y = pivot.index.values   # wd
    Z = pivot.values         # regret
    assert Z.shape == (Y.shape + X.shape), f"Shape mismatch: {Y.shape + X.shape=} != {Z.shape=}"

    # Plot the contour and make a colorbar
    cf = ax.contourf(X, Y, Z)
    cbar = plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.08)
    cbar.set_label("Regret")

    # Matplotlib
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Weight Decay")
    ax.set_yscale("log") # breaks for some reason
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=5)
    ax.set_title(title)

    # Get best point
    _min_idx_1d, min_val = Z.argmin(), Z.min()
    _y_idx, _x_idx = np.unravel_index(_min_idx_1d, Z.shape) # for some reason it gives the flattened argmin
    ax.plot(X[_x_idx], Y[_y_idx], "wo", ms=10, mec="k")

    return cbar, min_val

n_plots = 1
fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 5, 4))
fig.suptitle(f"{acq_func.replace('_', ' ').upper()} -- {n_dim}D", fontsize=14)

ax = axs # coz there's only 1
cbar, min_val = _plot_single(kdf, ax, title=f"{kt.upper().replace("-", " ")}")
print(min_val)
plt.show()
# %%
# Ok awesome, now let's have a look at those baselines. As a reminder:
base_df

# %%
# So we want to find the ones for specific subplot:

def _add_baseline_annotations(
    cbar,
    baseline_values: dict[str, int], # {"EI": 0.2, "RS": 0.4}
    fontsize=8,  # smaller text
    x_offset=1.5,  # tighter to the bar
    lw=1,  # thinner arrows
    box_alpha=0.8,  # lighter box
):
    for label, bv in baseline_values.items():
        # clip to colorbar range for positioning
        clipped_value = np.clip(bv, cbar.vmin, cbar.vmax)
        arrow_y, coords_type = clipped_value, "data"

        cbar.ax.annotate(
            f"{label}: {bv:.2f}",
            xy=(1.0, arrow_y),
            xycoords=("axes fraction", coords_type),
            xytext=(x_offset, arrow_y),
            textcoords=("axes fraction", coords_type),
            arrowprops=dict(arrowstyle="->", color="black", lw=lw),
            fontsize=fontsize,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=box_alpha),
        )

kdf_base = base_df.query("kernel_type == @kt and n_dim == @n_dim").copy()
kdf_vals: dict[str, int] = {row.acq_func: row.regret for row in kdf_base.itertuples()} # type: ignore
_add_baseline_annotations(cbar, kdf_vals)
# plt.show()
fig

# %%
