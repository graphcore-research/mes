# Variational Entropy Search is Just 1D Regression

This code is purely for running toy experiments for the research paper "Variational Entropy Search is Just 1D Regression" published at the NeurIPS 2025 [Frontiers in Probabalistic Inferece](https://fpineurips.framer.website) workshop.

Collections of toy experiments for approximating Max Value entropy Search with variational methods. This simply boils down to fitting a 1D regression model for each x location and measuring the model likelihood, or "goodness of fit" or loss function, and the x location with highest likelihood/lowest loss is chosen for expensive black box evaluation.

We build on this ICML 2025 spotlight paper: [A Unified Framework for Entropy Search and Expected Improvement in Bayesian Optimization](https://arxiv.org/pdf/2501.18756)

# Setup
```
# install uv first: see https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the other dependencies
uv venv
uv pip install -e .
uv run pytest
```

# Usage
You can run a single 1D test function experiment using the command
```
cd scripts
python make_1d_bo_animation.py    # runs BO and parses the results into an animation
```

To run a sweep of methods over a sweep of test functions to reproduce results in the paper
```
cd paper_experiments
python make_datasets.py           # creates a collection of synthetic test functions
python run_benchmarks.py          # run a sweep of BO methods on the test functions
plot_results.ipynb                # jupyter notbook to plot convergence curves
```

![Distribution of Max Values gitven 3 points](scripts/original_data.png)

![Max value varying for one synthetic point](scripts/animation.gif)

![Expected Improvement](scripts/bo_history.gif)