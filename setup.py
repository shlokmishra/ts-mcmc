from setuptools import setup, find_packages
setup(
    name="ts_mcmc",
    packages=find_packages(),
    py_modules=["tree", "mcmc", "recorder"]
)