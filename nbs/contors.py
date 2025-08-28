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

# df = pd.read_json(DATA_DIR / "benchmark_df_sweep.json")
df = pd.read_json(DATA_DIR / "benchmark_df_big.json")
df["regret"] = df.apply(
    lambda x: x["y_true_max"] - x["final_y_max"], axis=1
)
df = df.drop(columns=["steps", "y_max_history", "y_true_max", "final_y_max", "max_iters"])
df.query('kernel_type == "matern-5/2" and n_dim == 4 and wd == 0').head()

# %%
cols2group = [c for c in df.columns if c not in {"run_id", "regret"}]
# Check every config has the same number of runs
expected = df["run_id"].nunique()
n_runs = df.groupby(cols2group)["run_id"].nunique()
assert n_runs.eq(expected).all(), (
    f"Configs with != {expected} runs:\n{n_runs[n_runs != expected]}"
)
group_cols = [c for c in df.columns if c not in ["run_id", "regret"]]

mean_df = df.groupby(cols2group).agg('mean', 'count').drop(columns="run_id")
mean_df
# %%
regret_df = (
    mean_df.assign(regret=mean_df.groupby(group_cols)["regret"].transform("mean"))
    .reset_index()
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

def _plot_single(kdf, ax, title, setxlabel=True, setylabel=True):
    pivot = kdf.pivot(index="wd", columns="lr", values="regret")
    assert not pivot.isna().any().any(), "Missing points in the lr-wd grid"

    X = pivot.columns.values # lr
    Y = pivot.index.values   # wd
    Z = pivot.values         # regret
    assert Z.shape == (Y.shape + X.shape), f"Shape mismatch: {Y.shape + X.shape=} != {Z.shape=}"

    # Plot the contour and make a colorbar
    cf = ax.contourf(X, Y, Z, levels=15)
    cbar = plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.08)
    cbar.set_label("Regret")

    # Matplotlib
    ax.set_xscale("log")
    ax.set_yscale("log")
    if setxlabel:
        ax.set_xlabel("Learning Rate")
    if setylabel:
        ax.set_ylabel("Weight Decay")
    # ax.set_yscale("log") # breaks for some reason
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
print(base_df)

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

from itertools import product

acq_funcs = ["ves_gamma", "ves_mc_gamma"] # , "ves_mc_gamma"
kts = ["matern-3/2", "matern-5/2"]
n_dims = [2, 4]

prod = list(product(acq_funcs, kts, n_dims))
n_plots = len(prod)
row_stride = len(acq_funcs) * len(kts)
n_rows = n_plots // row_stride

fig, axs = plt.subplots(row_stride, n_rows, figsize=(n_rows * 5, row_stride * 4), sharex=True, sharey=True)
axs = axs.flatten()

def cdiv(x, y): return (x + y - 1) // y

for i, (acq_func, kt, n_dim) in enumerate(prod):
    ax = axs[i]
    sdf = sweep_df.query("acq_func == @acq_func and n_dim == @n_dim").copy()
    kdf = sdf.query("kernel_type == @kt").copy()
    title = f"{acq_func.upper().replace('_', ' ')} -- {kt.upper().replace('-', ' ')} ({n_dim}D)"
    cbar, min_val = _plot_single(kdf, ax, title=title, setylabel=i % row_stride == 0, setxlabel=(i // row_stride) == (n_rows - 1))
    kdf_base = base_df.query("kernel_type == @kt and n_dim == @n_dim").copy()
    kdf_vals: dict[str, int] = {row.acq_func: row.regret for row in kdf_base.itertuples()} # type: ignore
    _add_baseline_annotations(cbar, kdf_vals)

plt.show()

# %%

acq_funcs = ["ves_gamma", "ves_mc_gamma"]
kts = ["matern-3/2", "matern-5/2"]
n_dims = [2, 4]

fig = plt.figure(figsize=(len(n_dims)*5, len(acq_funcs)*len(kts)*4))
subfigs = fig.subfigures(nrows=len(acq_funcs), ncols=1)

for acq_func, subfig in zip(acq_funcs, subfigs):
    subfig.suptitle(acq_func.upper().replace('_', ' '), y=0.98, fontsize=14, fontweight='bold')
    axs = subfig.subplots(nrows=len(kts), ncols=len(n_dims), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    for r, kt in enumerate(kts):
        for c, n_dim in enumerate(n_dims):
            ax = axs[r, c]
            sdf = sweep_df.query("acq_func == @acq_func and n_dim == @n_dim").copy()
            kdf = sdf.query("kernel_type == @kt").copy()

            title = f"{kt.upper().replace('-', ' ')} ({n_dim}D)"
            cbar, min_val = _plot_single(
                kdf, ax,
                title=title,
                setylabel=(c == 0),                 # left col only
                setxlabel=(r == len(kts) - 1)       # bottom row only
            )

            kdf_base = base_df.query("kernel_type == @kt and n_dim == @n_dim").copy()
            kdf_vals = {row.acq_func: row.regret for row in kdf_base.itertuples()}  # type: ignore
            _add_baseline_annotations(cbar, kdf_vals)

plt.show()

# %%

fig = plt.figure(figsize=(len(n_dims)*5, len(acq_funcs)*len(kts)*4), constrained_layout=True)

# Small gap between the two subfigures (the sections)
gs = fig.add_gridspec(nrows=len(acq_funcs), ncols=1, hspace=0.02)
subfigs = [fig.add_subfigure(gs[i, 0]) for i in range(len(acq_funcs))]

# (optional) shrink global pads a touch

for acq_func, subfig in zip(acq_funcs, subfigs):
    subfig.suptitle(acq_func.upper().replace('_', ' '), fontsize=14, fontweight='bold')

    axs = subfig.subplots(nrows=len(kts), ncols=len(n_dims), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    for r, kt in enumerate(kts):
        for c, n_dim in enumerate(n_dims):
            ax = axs[r, c]
            sdf = sweep_df.query("acq_func == @acq_func and n_dim == @n_dim").copy()
            kdf = sdf.query("kernel_type == @kt").copy()
            title = f"{kt.upper().replace('-', ' ')} ({n_dim}D)"
            cbar, min_val = _plot_single(kdf, ax, title=title,
                                         setylabel=(c == 0),
                                         setxlabel=(r == len(kts) - 1))
            kdf_base = base_df.query("kernel_type == @kt and n_dim == @n_dim").copy()
            kdf_vals = {row.acq_func: row.regret for row in kdf_base.itertuples()}  # type: ignore
            _add_baseline_annotations(cbar, kdf_vals)

# %%
