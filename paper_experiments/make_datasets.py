from pathlib import Path

from boplay.benchmark_data import make_benchmark_data


DATA_DIR = Path(__file__).parent / "benchmark_datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_X = 15

if __name__ == "__main__":
    mat52_2d_short = make_benchmark_data(
        n_x=N_X,
        n_y=1000,
        kernel_type="matern-5/2",
        kernel_params={"len_scale": 5.0, "sigma_f": 1.0},
        x_min=[0, 0],
        x_max=[100, 100],
    )
    mat52_2d_short.save(DATA_DIR / "mat52_2d_short.json")

    mat52_2d_long = make_benchmark_data(
        n_x=N_X,
        n_y=1000,
        kernel_type="matern-5/2",
        kernel_params={"len_scale": 10.0, "sigma_f": 1.0},
        x_min=[0, 0],
        x_max=[100, 100],
    )
    mat52_2d_long.save(DATA_DIR / "mat52_2d_long.json")

    se_2d_short = make_benchmark_data(
        n_x=N_X,
        n_y=1000,
        kernel_type="se",
        kernel_params={"len_scale": 5.0, "sigma_f": 1.0},
        x_min=[0, 0],
        x_max=[100, 100],
    )
    se_2d_short.save(DATA_DIR / "se_2d_short.json")

    se_2d_long = make_benchmark_data(
        n_x=N_X,
        n_y=1000,
        kernel_type="se",
        kernel_params={"len_scale": 10.0, "sigma_f": 1.0},
        x_min=[0, 0],
        x_max=[100, 100],
    )
    se_2d_long.save(DATA_DIR / "se_2d_long.json")
