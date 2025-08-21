from setuptools import setup, find_packages

setup(
    name="MES edge case checking",
    version="0.1.0",
    description="Playing with MES on toy settings",
    packages=find_packages("."),
    install_requires=[
        "imageio",
        "matplotlib",
        "numpy",
        "pandas",
        "pytest",
        "scienceplots",
        "scipy",
        "scikit-learn",
        "tqdm",
    ],
    python_requires=">=3.12",
    include_package_data=True,
)
