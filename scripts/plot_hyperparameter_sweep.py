import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_results():
    """Load hyperparameter sweep results from JSON file."""
    results_file = DATA_DIR / "hyperparameter_sweep_results.json"
    with open(results_file, "r") as f:
        return json.load(f)


def plot_convergence_comparison(results):
    """Plot convergence curves comparing different acquisition functions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Group by n_dim and plot separately
    for dim_idx, n_dim in enumerate([1, 2]):
        ax = axes[dim_idx]
        
        # Group by acquisition function
        acq_results = defaultdict(list)
        for result in results:
            if result["n_dim"] == n_dim:
                acq_results[result["acq_func"]].append(result)
        
        for acq_func, acq_data in acq_results.items():
            convergence_curves = []
            for data in acq_data:
                y_max_history = np.array(data["y_max_history"])
                regret = data["y_true_max"] - y_max_history[:, 1]
                convergence_curves.append(regret)
            
            # Calculate mean and std
            min_length = min(len(curve) for curve in convergence_curves)
            curves_array = np.array([curve[:min_length] for curve in convergence_curves])
            mean_regret = np.mean(curves_array, axis=0)
            std_regret = np.std(curves_array, axis=0)
            
            iterations = np.arange(len(mean_regret))
            ax.plot(iterations, mean_regret, label=f"{acq_func}", linewidth=2)
            ax.fill_between(iterations, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Simple Regret")
        ax.set_yscale("log")
        ax.set_title(f"Convergence Comparison ({n_dim}D)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add kernel comparison plots
    for kernel_idx, kernel_type in enumerate(["rbf", "matern_3_2"]):
        ax = axes[2 + kernel_idx]
        
        # Group by acquisition function for this kernel
        acq_results = defaultdict(list)
        for result in results:
            if result["kernel_type"] == kernel_type and result["n_dim"] == 1:
                acq_results[result["acq_func"]].append(result)
        
        for acq_func, acq_data in acq_results.items():
            convergence_curves = []
            for data in acq_data:
                y_max_history = np.array(data["y_max_history"])
                regret = data["y_true_max"] - y_max_history[:, 1]
                convergence_curves.append(regret)
            
            if convergence_curves:
                min_length = min(len(curve) for curve in convergence_curves)
                curves_array = np.array([curve[:min_length] for curve in convergence_curves])
                mean_regret = np.mean(curves_array, axis=0)
                std_regret = np.std(curves_array, axis=0)
                
                iterations = np.arange(len(mean_regret))
                ax.plot(iterations, mean_regret, label=f"{acq_func}", linewidth=2)
                ax.fill_between(iterations, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Simple Regret")
        ax.set_yscale("log")
        ax.set_title(f"Kernel: {kernel_type} (1D)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_lengthscale_analysis(results):
    """Plot performance vs length scale for different acquisition functions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Create DataFrame for easier analysis
    data_rows = []
    for result in results:
        y_max_history = np.array(result["y_max_history"])
        final_regret = result["y_true_max"] - y_max_history[-1, 1]
        
        data_rows.append({
            "acq_func": result["acq_func"],
            "kernel_type": result["kernel_type"],
            "len_scale": result["len_scale"],
            "n_dim": result["n_dim"],
            "final_regret": final_regret,
            "convergence_rate": np.mean(np.diff(y_max_history[:10, 1])) if len(y_max_history) > 10 else 0
        })
    
    df = pd.DataFrame(data_rows)
    
    # Plot for each dimension and kernel
    plot_configs = [
        (1, "rbf", 0, 0),
        (1, "matern_3_2", 0, 1),
        (2, "rbf", 1, 0),
        (2, "matern_3_2", 1, 1)
    ]
    
    for n_dim, kernel, row, col in plot_configs:
        ax = axes[row, col]
        subset = df[(df["n_dim"] == n_dim) & (df["kernel_type"] == kernel)]
        
        for acq_func in subset["acq_func"].unique():
            acq_subset = subset[subset["acq_func"] == acq_func]
            ax.semilogx(acq_subset["len_scale"], acq_subset["final_regret"], 
                       "o-", label=acq_func, linewidth=2, markersize=6)
        
        ax.set_xlabel("Length Scale")
        ax.set_ylabel("Final Simple Regret")
        ax.set_yscale("log")
        ax.set_title(f"{kernel} kernel ({n_dim}D)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lengthscale_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_acquisition_function_heatmap(results):
    """Create heatmap showing performance across different hyperparameters."""
    # Create pivot table for heatmap
    data_rows = []
    for result in results:
        if result["n_dim"] == 1:  # Focus on 1D for clarity
            y_max_history = np.array(result["y_max_history"])
            final_regret = result["y_true_max"] - y_max_history[-1, 1]
            
            data_rows.append({
                "acq_func": result["acq_func"],
                "kernel_type": result["kernel_type"],
                "len_scale": result["len_scale"],
                "final_regret": final_regret
            })
    
    df = pd.DataFrame(data_rows)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, acq_func in enumerate(df["acq_func"].unique()):
        ax = axes[idx]
        subset = df[df["acq_func"] == acq_func]
        
        # Create pivot table
        pivot = subset.pivot_table(values="final_regret", 
                                 index="kernel_type", 
                                 columns="len_scale", 
                                 aggfunc="mean")
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", 
                   ax=ax, cbar_kws={"label": "Final Simple Regret"})
        ax.set_title(f"{acq_func} Performance")
        ax.set_xlabel("Length Scale")
        ax.set_ylabel("Kernel Type")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "performance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_acquisition_function_evolution(results, max_examples=4):
    """Plot evolution of acquisition function values during optimization."""
    # Filter for 1D results with interesting cases
    filtered_results = [r for r in results if r["n_dim"] == 1 and r["len_scale"] == 10][:max_examples]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(filtered_results):
        if idx >= 4:
            break
            
        ax = axes[idx]
        state_history = result["state_history"]
        
        # Plot acquisition function evolution at key iterations
        iterations_to_plot = [0, len(state_history)//4, len(state_history)//2, -1]
        
        for iter_idx in iterations_to_plot:
            if iter_idx < len(state_history):
                state = state_history[iter_idx]
                x_grid = np.linspace(0, len(state["acq_fun_vals"]), len(state["acq_fun_vals"]))
                
                alpha = 0.4 + 0.6 * (iterations_to_plot.index(iter_idx) / len(iterations_to_plot))
                ax.plot(x_grid, state["acq_fun_vals"], 
                       alpha=alpha, label=f"Iter {state['n_train']}")
        
        ax.set_xlabel("Grid Index")
        ax.set_ylabel("Acquisition Function Value")
        ax.set_title(f"{result['acq_func']} - {result['kernel_type']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "acquisition_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_summary_statistics(results):
    """Generate and save summary statistics."""
    data_rows = []
    for result in results:
        y_max_history = np.array(result["y_max_history"])
        final_regret = result["y_true_max"] - y_max_history[-1, 1]
        
        # Calculate convergence rate (improvement in first 10 iterations)
        early_improvement = (y_max_history[min(10, len(y_max_history)-1), 1] - 
                           y_max_history[0, 1]) if len(y_max_history) > 1 else 0
        
        data_rows.append({
            "acq_func": result["acq_func"],
            "kernel_type": result["kernel_type"],
            "len_scale": result["len_scale"],
            "n_dim": result["n_dim"],
            "final_regret": final_regret,
            "early_improvement": early_improvement,
            "total_iterations": len(y_max_history)
        })
    
    df = pd.DataFrame(data_rows)
    
    # Group by acquisition function and compute statistics
    summary = df.groupby("acq_func").agg({
        "final_regret": ["mean", "std", "min", "max"],
        "early_improvement": ["mean", "std"],
        "total_iterations": "mean"
    }).round(4)
    
    print("Summary Statistics by Acquisition Function:")
    print(summary)
    
    # Save to file
    summary.to_csv(PLOTS_DIR / "summary_statistics.csv")
    
    return df


def main():
    """Main function to run all analyses."""
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} experimental results")
    
    print("Generating plots...")
    plot_convergence_comparison(results)
    plot_lengthscale_analysis(results)
    plot_acquisition_function_heatmap(results)
    plot_acquisition_function_evolution(results)
    
    print("Generating summary statistics...")
    df = generate_summary_statistics(results)
    
    print(f"All plots saved to {PLOTS_DIR}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()