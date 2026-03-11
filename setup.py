from setuptools import setup, find_packages

setup(
    name="pinn_landslide",
    version="0.1.0",
    author="suyash-04",
    description="Physics-Informed Neural Network for Landslide Stability Analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
