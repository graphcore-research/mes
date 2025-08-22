# Running Experiments for the main paper

1. Make the datasets: `python make_dataset.py`, this saves new JSON files in `benchmark_datasets`

2. Open `run_benchmark_exp.py` and make a list of acquisition functions and benchmark datasets you want to run

3. Run all the experiments `python run_benchmark_exp.py`
  - each call to this script makes a new results sub directory
  - if you give it an  old directory, it will skip pre-existing results files


# Plotting Results

Open Jupyter notebook, choose your results DIR and run the cells