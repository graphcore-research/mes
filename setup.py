from setuptools import setup, find_packages

setup(
    name="MES edge case checking",
    version="0.1.0",
    description="Playing with MES on toy settings",
    packages=find_packages(include=["MES_VI"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
    python_requires=">=3.12",
    include_package_data=True,
)
